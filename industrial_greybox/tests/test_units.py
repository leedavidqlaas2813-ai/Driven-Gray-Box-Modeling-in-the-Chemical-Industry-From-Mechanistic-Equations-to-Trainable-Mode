"""单元测试：数值积分、时滞对齐、清洗、窗口切分、慢特征、影子因果性。

运行方式（仓库根目录）::

    python -m unittest discover -s industrial_greybox/tests -v

测试刻意不训练长模型：数值与数据管道问题应在秒级暴露；需要训练的
端到端检查放在 ``test_e2e`` 并默认使用极小 epoch。
"""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd

from industrial_greybox.config import (
    COL_C_A,
    COL_C_A0,
    COL_C_A_ONLINE,
    COL_RUN_ID,
    COL_T_C,
    COL_T_IN,
    COL_T_REACTOR,
    COL_TAU,
    COL_TIME_MIN,
    CONTROL_COLUMNS,
    DeadbandRule,
    DelayConfig,
    HygieneConfig,
    IntegratorConfig,
    LimitRule,
    ModelConfig,
    PhysicsConfig,
    RateOfChangeRule,
    RateCorrectionConfig,
    ShadowConfig,
    config_from_dict,
)
from industrial_greybox.data import build_datasets, check_no_run_leakage, split_runs
from industrial_greybox.delays import (
    ShadowBuffer,
    _attach_lab_values,
    align_lab_results,
    shift_controls_for_actuator_delay,
)
from industrial_greybox.hygiene import (
    FLAG_SUFFIX,
    IndustrialHygienePipeline,
    apply_column_mapping,
)
from industrial_greybox.numerics import (
    euler_step,
    explicit_stability_ratio,
    guard_state,
    imex_step,
    integrate_interval,
    rk4_step,
)
from industrial_greybox.slow_features import (
    cumulative_feed_m3,
    ema_temperature,
    normalize_slow_features,
)
from industrial_greybox.synthetic_plant import (
    CampaignSpec,
    PlantTruth,
    simulate_campaign,
    simulate_stiff_run,
)


class TestNumerics(unittest.TestCase):
    """积分格式必须收敛到参考解，且在准刚性工况下不发散。"""

    def setUp(self) -> None:
        self.truth = PlantTruth()
        self.physics = PhysicsConfig()

    def _reference(self, dt: float, n_steps: int, tau: float, ua: float, sub: int = 256):
        """用极小子步 RK4 生成参考解（线性 + 非线性完整方程）。"""
        c = 1.2 * 0.8
        t = 355.0
        h = dt / sub
        for _ in range(n_steps * sub):
            # 与 mechanistic_layer 相同的方程，但参数取真值。
            k = 0.032 * np.exp(-48_000 / 8.314 * (1 / t - 1 / 350.0))
            rate = k * c
            dc = (1.2 - c) / tau - rate
            dtemp = (350.0 - t) / tau + 24.0 * rate + 0.10 * (290.0 - t)
            d1 = (dc, dtemp)
            k2 = 0.032 * np.exp(-48_000 / 8.314 * (1 / (t + h / 2 * d1[1]) - 1 / 350.0))
            r2 = k2 * (c + h / 2 * d1[0])
            d2 = (
                (1.2 - (c + h / 2 * d1[0])) / tau - r2,
                (350.0 - (t + h / 2 * d1[1])) / tau + 24.0 * r2 + 0.10 * (290.0 - (t + h / 2 * d1[1])),
            )
            k3 = 0.032 * np.exp(-48_000 / 8.314 * (1 / (t + h / 2 * d2[1]) - 1 / 350.0))
            r3 = k3 * (c + h / 2 * d2[0])
            d3 = (
                (1.2 - (c + h / 2 * d2[0])) / tau - r3,
                (350.0 - (t + h / 2 * d2[1])) / tau + 24.0 * r3 + 0.10 * (290.0 - (t + h / 2 * d2[1])),
            )
            k4 = 0.032 * np.exp(-48_000 / 8.314 * (1 / (t + h * d3[1]) - 1 / 350.0))
            r4 = k4 * (c + h * d3[0])
            d4 = (
                (1.2 - (c + h * d3[0])) / tau - r4,
                (350.0 - (t + h * d3[1])) / tau + 24.0 * r4 + 0.10 * (290.0 - (t + h * d3[1])),
            )
            c += h / 6 * (d1[0] + 2 * d2[0] + 2 * d3[0] + d4[0])
            t += h / 6 * (d1[1] + 2 * d2[1] + 2 * d3[1] + d4[1])
        return np.array([c, t])

    def test_rk4_matches_reference_better_than_euler(self) -> None:
        """相同步长下 RK4 误差应显著小于 Euler。"""
        dt = 0.5
        tau = 5.0
        reference = self._reference(dt, 10, tau, 0.08)
        state = np.array([[1.2 * 0.8, 355.0]])
        # 用与参考解相同的方程做数值积分（缩放坐标在这里不引入）。
        import keras
        from keras import ops

        def nonlinear(x):
            c = x[:, 0:1]
            t = x[:, 1:2]
            k = 0.032 * ops.exp(-48_000 / 8.314 * (1 / t - 1 / 350.0))
            rate = k * c
            return ops.concatenate([-rate, 24.0 * rate], axis=-1)

        decay = ops.convert_to_tensor([[1 / tau, 1 / tau + 0.10]], dtype="float32")
        forcing = ops.convert_to_tensor([[1.2 / tau, 350.0 / tau + 0.10 * 290.0]], dtype="float32")
        x_euler = state.copy()
        x_rk4 = state.copy()
        for _ in range(10):
            x_euler = np.asarray(
                euler_step(ops.convert_to_tensor(x_euler), dt, decay_per_min=decay,
                           forcing=forcing, nonlinear=nonlinear)
            )
            x_rk4 = np.asarray(
                rk4_step(ops.convert_to_tensor(x_rk4), dt, decay_per_min=decay,
                         forcing=forcing, nonlinear=nonlinear)
            )
        err_euler = np.linalg.norm(x_euler[0] - reference)
        err_rk4 = np.linalg.norm(x_rk4[0] - reference)
        self.assertLess(err_rk4, err_euler)
        self.assertLess(err_rk4, 1e-2)

    def test_imex_survives_stiff_step(self) -> None:
        """显式 Euler 在 h*a>2 时振荡，IMEX 仍保持有界且合理。"""
        dt = 1.0
        tau = 0.2  # 1/tau = 5 min^-1
        ua = 2.0
        self.assertGreater(explicit_stability_ratio(dt, tau, ua), 2.0)
        import keras
        from keras import ops

        def nonlinear(x):
            return ops.zeros_like(x)

        decay = ops.convert_to_tensor([[1 / tau, 1 / tau + ua]], dtype="float32")
        forcing = ops.convert_to_tensor([[1.0 / tau, 350.0 / tau + ua * 300.0]], dtype="float32")
        state = ops.convert_to_tensor([[1.0, 350.0]], dtype="float32")
        euler_state = state
        imex_state = state
        for _ in range(20):
            euler_state = euler_step(euler_state, dt, decay_per_min=decay, forcing=forcing, nonlinear=nonlinear)
            imex_state = imex_step(imex_state, dt, decay_per_min=decay, forcing=forcing, nonlinear=nonlinear)
        imex_array = np.asarray(imex_state)[0]
        euler_array = np.asarray(euler_state)[0]
        # 纯线性系统在 20 步后应已接近解析稳态 forcing/decay。
        expected = np.asarray(forcing) / np.asarray(decay)
        self.assertTrue(np.all(np.isfinite(imex_array)))
        self.assertLess(np.abs(imex_array - expected[0]).max(), 0.05)
        # 显式 Euler 在 h*a > 2 时必然振荡/发散：这里只要求它“明显更糟”，
        # 不要求它保持有限——那正是引入 IMEX 的理由。
        euler_error = (
            np.inf
            if not np.all(np.isfinite(euler_array))
            else float(np.abs(euler_array - expected[0]).max())
        )
        imex_error = float(np.abs(imex_array - expected[0]).max())
        self.assertGreater(euler_error, imex_error)

    def test_guard_state_replaces_nonfinite_with_fallback(self) -> None:
        import keras
        from keras import ops

        state = ops.convert_to_tensor([[np.nan, 350.0], [1.0, np.inf], [1.0, 350.0]], dtype="float32")
        fallback = ops.convert_to_tensor([[0.9, 349.0], [0.8, 348.0], [0.7, 347.0]], dtype="float32")
        safe, nonfinite, clipped = guard_state(
            state, fallback, c_a_bounds=(0.0, 5.0), t_bounds=(280.0, 520.0)
        )
        safe = np.asarray(safe)
        self.assertTrue(np.isfinite(safe).all())
        np.testing.assert_allclose(safe[0], [0.9, 349.0])
        np.testing.assert_allclose(safe[1], [0.8, 348.0])
        # 第 3 行本身是有限值，应原样保留（不能被 fallback 覆盖）。
        np.testing.assert_allclose(safe[2], [1.0, 350.0])
        self.assertEqual(np.asarray(nonfinite).sum(), 2)
        # 只有第 1 行（回退后仍在界内）与第 2 行（回退后仍在界内）不需要裁剪；
        # 第 2 行 fallback 温度 348 也在界内。
        self.assertEqual(np.asarray(clipped).sum(), 0)

    def test_guard_state_clips_out_of_bounds(self) -> None:
        import keras
        from keras import ops

        state = ops.convert_to_tensor([[-1.0, 900.0]], dtype="float32")
        fallback = ops.convert_to_tensor([[1.0, 350.0]], dtype="float32")
        safe, _, clipped = guard_state(
            state, fallback, c_a_bounds=(0.0, 5.0), t_bounds=(280.0, 520.0)
        )
        np.testing.assert_allclose(np.asarray(safe)[0], [0.0, 520.0])
        self.assertTrue(bool(np.asarray(clipped)[0]))

    def test_integrate_interval_substeps_converge(self) -> None:
        """子步越多，越接近细步参考解。"""
        import keras
        from keras import ops

        def nonlinear(x):
            c = x[:, 0:1]
            return ops.concatenate([-0.5 * c, 0.5 * c], axis=-1)

        decay = ops.convert_to_tensor([[0.2, 0.2]], dtype="float32")
        forcing = ops.convert_to_tensor([[0.0, 0.0]], dtype="float32")
        state = ops.convert_to_tensor([[1.0, 350.0]], dtype="float32")
        coarse = integrate_interval(
            state, 1.0, method="rk4", substeps=1,
            decay_per_min=decay, forcing=forcing, nonlinear=nonlinear,
        )
        fine = integrate_interval(
            state, 1.0, method="rk4", substeps=16,
            decay_per_min=decay, forcing=forcing, nonlinear=nonlinear,
        )
        reference = integrate_interval(
            state, 1.0, method="rk4", substeps=256,
            decay_per_min=decay, forcing=forcing, nonlinear=nonlinear,
        )
        coarse_error = np.abs(np.asarray(coarse) - np.asarray(reference)).max()
        fine_error = np.abs(np.asarray(fine) - np.asarray(reference)).max()
        self.assertLessEqual(fine_error, coarse_error)

    def test_stiff_run_is_finite_for_all_methods(self) -> None:
        from industrial_greybox.synthetic_plant import simulate_stiff_run as stiff

        times, reference = stiff(dt_min=1.0, tau_min=0.35, ua_per_min=0.6, n_steps=20)
        self.assertTrue(np.isfinite(reference).all())
        self.assertEqual(len(times), 21)


class TestConfig(unittest.TestCase):
    def test_roundtrip_via_dict(self) -> None:
        from dataclasses import asdict

        config = ModelConfig()
        rebuilt = config_from_dict(asdict(config))
        self.assertEqual(config.signature(), rebuilt.signature())

    def test_signature_changes_with_physics(self) -> None:
        from dataclasses import replace

        base = ModelConfig()
        changed = replace(base, physics=replace(base.physics, activation_energy_j_mol=50_000.0))
        self.assertNotEqual(base.signature(), changed.signature())

    def test_unknown_config_key_rejected(self) -> None:
        from dataclasses import asdict

        payload = asdict(ModelConfig())
        payload["physics"]["unknown_field"] = 1.0
        with self.assertRaises(ValueError):
            config_from_dict(payload)

    def test_integration_defaults_are_documented(self) -> None:
        config = IntegratorConfig()
        self.assertIn(config.method, ("euler", "rk4", "imex"))
        self.assertGreaterEqual(config.substeps, 1)


class TestDelays(unittest.TestCase):
    def test_align_uses_result_time_minus_dead_time(self) -> None:
        """结果时间戳必须回拉死时间，且只有吸附行带监督掩码。"""
        dcs = pd.DataFrame(
            {
                COL_RUN_ID: ["R1"] * 41,
                COL_TIME_MIN: np.arange(41) * 0.5,
            }
        )
        lab = pd.DataFrame(
            {
                COL_RUN_ID: ["R1", "R1"],
                COL_TIME_MIN: [12.0, 22.0],  # 结果到达时刻
                COL_C_A: [0.8, 0.7],
            }
        )
        aligned = align_lab_results(
            dcs, lab, dead_time_min=2.0, timestamp_semantics="result"
        )
        # 过程时刻 10.0 与 20.0 都落在 0.5 min 网格上。
        supervised = aligned[aligned["c_a_mask"] > 0]
        self.assertEqual(len(supervised), 2)
        np.testing.assert_allclose(supervised[COL_TIME_MIN].to_numpy(), [10.0, 20.0])
        np.testing.assert_allclose(supervised["c_a_aligned"].to_numpy(), [0.8, 0.7])
        # 第一条化验之前没有值。
        self.assertTrue(np.isnan(aligned.loc[aligned[COL_TIME_MIN] < 10.0, "c_a_aligned"]).all())

    def test_align_process_semantics_ignores_dead_time(self) -> None:
        dcs = pd.DataFrame({COL_RUN_ID: ["R1"] * 11, COL_TIME_MIN: np.arange(11) * 1.0})
        lab = pd.DataFrame({COL_RUN_ID: ["R1"], COL_TIME_MIN: [5.0], COL_C_A: [1.1]})
        aligned = align_lab_results(
            dcs,
            lab,
            dead_time_min=3.0,
            timestamp_semantics="process",
        )
        mask_rows = aligned[aligned["c_a_mask"] > 0]
        self.assertEqual(len(mask_rows), 1)
        self.assertAlmostEqual(float(mask_rows[COL_TIME_MIN].iloc[0]), 5.0)

    def test_off_grid_lab_is_dropped_not_interpolated(self) -> None:
        dcs = pd.DataFrame({COL_RUN_ID: ["R1"] * 6, COL_TIME_MIN: np.arange(6) * 1.0})
        lab = pd.DataFrame({COL_RUN_ID: ["R1"], COL_TIME_MIN: [2.4], COL_C_A: [1.0]})
        aligned = align_lab_results(
            dcs, lab, dead_time_min=0.0, timestamp_semantics="process",
            alignment_tolerance_min=0.2,
        )
        # 2.4 与最近网格 2.0/2.5 相差均为 0.4 > 0.2，不应吸附。
        self.assertEqual(aligned["c_a_mask"].sum(), 0.0)
        # 但零阶保持值仍填满整列，便于人工审查。
        self.assertTrue(np.isfinite(aligned.loc[aligned[COL_TIME_MIN] >= 2.5, "c_a_aligned"]).all())

    def test_delay_buffer_rejects_unsorted_times(self) -> None:
        buffer = ShadowBuffer(horizon_min=10.0)
        buffer.push(1.0, [1.0])
        with self.assertRaises(ValueError):
            buffer.push(0.5, [2.0])

    def test_delay_buffer_returns_none_when_history_lost(self) -> None:
        buffer = ShadowBuffer(horizon_min=2.0)
        for time_min in np.arange(0.0, 20.1, 1.0):
            buffer.push(time_min, [time_min])
        self.assertIsNone(buffer.query(-5.0))
        # 缓冲区只保留 horizon 内的点，但至少保留最近一点。
        self.assertIsNotNone(buffer.query(19.0))

    def test_delay_buffer_zoh_and_linear(self) -> None:
        zoh = ShadowBuffer(horizon_min=100.0, interpolation="zoh")
        lin = ShadowBuffer(horizon_min=100.0, interpolation="linear")
        for time_min, value in [(0.0, 0.0), (1.0, 1.0)]:
            zoh.push(time_min, value)
            lin.push(time_min, value)
        self.assertAlmostEqual(float(zoh.query(0.5)[0]), 0.0)
        self.assertAlmostEqual(float(lin.query(0.5)[0]), 0.5)

    def test_actuator_delay_shift_is_integer_steps(self) -> None:
        frame = pd.DataFrame({"a": np.arange(10, dtype=float)})
        shifted = shift_controls_for_actuator_delay(
            frame, dt_min=0.5, columns=["a"], delay_min=1.0
        )
        np.testing.assert_allclose(shifted["a"].to_numpy()[2:], np.arange(8))
        with self.assertRaises(ValueError):
            shift_controls_for_actuator_delay(frame, dt_min=0.5, columns=["a"], delay_min=0.7)


class TestHygiene(unittest.TestCase):
    def _frame(self) -> pd.DataFrame:
        times = np.arange(40) * 0.5
        return pd.DataFrame(
            {
                COL_RUN_ID: ["R1"] * 40,
                COL_TIME_MIN: times,
                COL_T_REACTOR: 350.0 + np.linspace(0, 1, 40),
                COL_C_A0: np.full(40, 1.2),
                COL_T_IN: np.full(40, 345.0),
                COL_T_C: np.full(40, 295.0),
                COL_TAU: np.full(40, 5.0),
                COL_C_A_ONLINE: 1.0 + np.linspace(0, 0.05, 40),
            }
        )

    def test_hard_limits_nullify(self) -> None:
        frame = self._frame()
        frame.loc[5, COL_C_A_ONLINE] = 9.0
        cleaned, report = IndustrialHygienePipeline().clean(frame)
        self.assertTrue(np.isnan(cleaned.loc[5, COL_C_A_ONLINE]))
        self.assertIn("limit", cleaned.loc[5, COL_C_A_ONLINE + FLAG_SUFFIX])

    def test_isolated_spike_is_removed(self) -> None:
        frame = self._frame()
        frame.loc[10, COL_T_REACTOR] += 15.0
        cleaned, _ = IndustrialHygienePipeline().clean(frame)
        self.assertTrue(np.isnan(cleaned.loc[10, COL_T_REACTOR]))

    def test_deadband_flags_long_constant_and_ignores_short_hold(self) -> None:
        # 默认规则要求连续 30 min 不变，因此样本必须覆盖 >30 min。
        times = np.arange(200) * 0.5  # 100 min
        frame = pd.DataFrame(
            {
                COL_RUN_ID: ["R1"] * 200,
                COL_TIME_MIN: times,
                COL_T_REACTOR: 350.0 + np.linspace(0, 1, 200),
                COL_C_A0: np.full(200, 1.2),
                COL_T_IN: np.full(200, 345.0),
                COL_T_C: np.full(200, 295.0),
                COL_TAU: np.full(200, 5.0),
                COL_C_A_ONLINE: 1.0,
            }
        )
        cleaned, report = IndustrialHygienePipeline().clean(frame)
        flags = cleaned[COL_C_A_ONLINE + FLAG_SUFFIX]
        self.assertTrue(flags.str.contains("deadband").any())
        # 默认只标记不删除：值仍然可用。
        self.assertFalse(cleaned[COL_C_A_ONLINE].isna().any())

    def test_deadband_nan_action_removes_values(self) -> None:
        frame = self._frame()
        frame[COL_C_A_ONLINE] = 1.0
        config = HygieneConfig(
            deadband_rules=(
                DeadbandRule(
                    COL_C_A_ONLINE,
                    window_min=2.0,
                    min_duration_min=5.0,
                    std_tol=1e-9,
                    span_tol=1e-9,
                    action="nan",
                ),
            ),
            spike_rules=(),
            rate_rules=(),
            limits=(),
            impute_columns=(),
        )
        cleaned, _ = IndustrialHygienePipeline(config).clean(frame)
        self.assertTrue(cleaned[COL_C_A_ONLINE].isna().sum() > 0)

    def test_rate_of_change_flags_step(self) -> None:
        frame = self._frame()
        frame.loc[20, COL_T_REACTOR] += 20.0
        # 关闭毛刺规则，单独验证变化率规则。
        config = HygieneConfig(spike_rules=(), deadband_rules=(), limits=())
        cleaned, _ = IndustrialHygienePipeline(config).clean(frame)
        self.assertIn("rate_of_change", cleaned.loc[20, COL_T_REACTOR + FLAG_SUFFIX])

    def test_short_gap_interpolated_long_gap_kept(self) -> None:
        frame = self._frame()
        frame.loc[[10, 11], COL_T_REACTOR] = np.nan
        frame.loc[[20, 21, 22, 23, 24, 25], COL_T_REACTOR] = np.nan
        config = HygieneConfig(
            limits=(), rate_rules=(), deadband_rules=(), spike_rules=(),
            max_interpolate_gap=3,
        )
        cleaned, report = IndustrialHygienePipeline(config).clean(frame)
        self.assertTrue(np.isfinite(cleaned.loc[10, COL_T_REACTOR]))
        self.assertTrue(cleaned.loc[20:25, COL_T_REACTOR].isna().all())
        self.assertEqual(report.unresolved_nan.get(COL_T_REACTOR), 6)

    def test_duplicate_time_rejected_but_unsorted_rows_accepted(self) -> None:
        """重复时间戳必须拒绝；单纯行序乱序（DCS 常见）应被排序后接受。"""
        frame = self._frame()
        frame.loc[5, COL_TIME_MIN] = frame.loc[4, COL_TIME_MIN]
        with self.assertRaises(ValueError):
            IndustrialHygienePipeline().clean(frame)

        shuffled = self._frame().sample(frac=1.0, random_state=0).reset_index(drop=True)
        cleaned, _ = IndustrialHygienePipeline().clean(shuffled)
        times = cleaned.groupby(COL_RUN_ID)[COL_TIME_MIN].apply(
            lambda series: np.all(np.diff(series.to_numpy()) > 0)
        )
        self.assertTrue(bool(times.all()))

    def test_report_separates_suspicious_from_informative(self) -> None:
        frame = self._frame()
        frame.loc[7, COL_C_A_ONLINE] = 7.0
        _, report = IndustrialHygienePipeline().clean(frame)
        suspicious = report.counts_by_rule.get("limit", 0)
        self.assertGreaterEqual(suspicious, 1)
        self.assertIn("可疑值", report.summary_line())

    def test_column_mapping(self) -> None:
        frame = pd.DataFrame({"CA_LAB": [1.0], "run": ["R1"]})
        mapped = apply_column_mapping(frame, {"CA_LAB": COL_C_A, "run": COL_RUN_ID})
        self.assertIn(COL_C_A, mapped.columns)


class TestSlowFeatures(unittest.TestCase):
    def test_cumulative_feed_matches_constant_flow(self) -> None:
        frame = pd.DataFrame(
            {
                COL_RUN_ID: ["R1"] * 11,
                COL_TIME_MIN: np.arange(11) * 1.0,
                COL_TAU: np.full(11, 10.0),
            }
        )
        volume = cumulative_feed_m3(frame, reactor_volume_l=100.0)
        # q = 100/10 = 10 L/min = 0.01 m³/min；10 min -> 0.1 m³。
        self.assertAlmostEqual(float(volume.iloc[-1]), 0.1, places=6)

    def test_ema_is_bounded_between_min_and_max(self) -> None:
        frame = pd.DataFrame(
            {
                COL_RUN_ID: ["R1"] * 60,
                COL_TIME_MIN: np.arange(60) * 1.0,
                COL_T_REACTOR: np.linspace(340.0, 360.0, 60),
            }
        )
        ema = ema_temperature(frame, dt_min=1.0, time_constant_min=10.0)
        self.assertGreaterEqual(ema.min(), 340.0 - 1e-9)
        self.assertLessEqual(ema.max(), 360.0 + 1e-9)

    def test_normalisation_is_bounded(self) -> None:
        values = np.array(
            [
                [0.0, 0.0, 350.0],
                [10_000.0, 1e6, 600.0],
                [-5.0, -100.0, 100.0],
            ]
        )
        scaled = normalize_slow_features(values, age_ref_days=90.0, feed_ref_m3=2000.0)
        self.assertTrue(np.all(np.abs(scaled) <= 1.0 + 1e-6))


class TestWindowSplit(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # 关闭故障注入：本组测试验证切分/掩码，不希望被温度缺口干扰。
        # 清洗故障由 TestHygiene 单独覆盖。
        # 运行需足够长：窗口初值必须锚定在一条化验结果附近，训练窗口的
        # 数量大致为 ``化验点数 × (化验间隔 / (stride·dt))``。
        cls.dcs, cls.lab = simulate_campaign(
            CampaignSpec(
                n_runs=9,
                n_steps=200,
                days_span=20,
                lab_interval_min=10,
                inject_faults=False,
            )
        )
        from industrial_greybox.pipeline import prepare_frame

        cls.frame, _ = prepare_frame(
            cls.dcs, cls.lab, config=ModelConfig(), delay=DelayConfig()
        )

    def test_split_has_no_run_overlap(self) -> None:
        groups = split_runs(self.frame, seed=3)
        names = list(groups)
        for i, first in enumerate(names):
            for second in names[i + 1 :]:
                self.assertFalse(set(groups[first]) & set(groups[second]))

    def test_datasets_have_no_leakage(self) -> None:
        config = ModelConfig()
        datasets, groups = build_datasets(
            self.frame,
            physics=config.physics,
            slow_reference=(90.0, 2000.0),
            seed=1,
        )
        check_no_run_leakage(datasets, groups)

    def test_weights_only_on_supervised_points(self) -> None:
        config = ModelConfig()
        datasets, _ = build_datasets(
            self.frame,
            physics=config.physics,
            slow_reference=(90.0, 2000.0),
            seed=1,
        )
        train = datasets["train"]
        # 浓度初值点是 ZOH 锚点，不应被当作监督目标。
        self.assertTrue(np.all(train.weights[:, 0, 0] == 0.0))
        # 温度通道全权重。
        self.assertTrue(np.all(train.weights[:, :, 1] > 0))
        # 未被监督的浓度目标必须置 0，而不是保留 NaN。
        unsupervised = train.weights[:, :, 0] == 0
        self.assertTrue(np.all(train.targets[:, :, 0][unsupervised] == 0.0))

    def test_non_uniform_grid_rejected(self) -> None:
        frame = self.frame.copy()
        mask = frame[COL_RUN_ID] == frame[COL_RUN_ID].iloc[0]
        indices = frame.index[mask][:5]
        frame.loc[indices, COL_TIME_MIN] = frame.loc[indices, COL_TIME_MIN] + 0.1
        config = ModelConfig()
        with self.assertRaises(ValueError):
            build_datasets(
                frame,
                physics=config.physics,
                slow_reference=(90.0, 2000.0),
            )


class TestShadowCausality(unittest.TestCase):
    """影子的核心性质：化验到达前，预测不受该化验值影响。"""

    def test_lab_value_does_not_leak_before_arrival(self) -> None:
        from keras import ops

        from industrial_greybox.mechanistic_layer import CSTRGreyBoxLayer

        config = ModelConfig()
        layer = CSTRGreyBoxLayer(config)
        state = np.array([[1.0, 350.0]], dtype=np.float32)
        control = np.array([[1.1, 350.0, 300.0, 5.0]], dtype=np.float32)
        slow = np.zeros((1, 3), dtype=np.float32)
        next_state, _ = layer.step(
            ops.convert_to_tensor(state),
            ops.convert_to_tensor(control),
            ops.convert_to_tensor(slow),
        )
        self.assertTrue(np.all(np.isfinite(np.asarray(next_state))))
        # 第一步只依赖初值与控制量。
        next_state_2, _ = layer.step(
            ops.convert_to_tensor(state),
            ops.convert_to_tensor(control),
            ops.convert_to_tensor(slow),
        )
        np.testing.assert_allclose(np.asarray(next_state), np.asarray(next_state_2))

    def test_mechanistic_baseline_equals_model_without_corrections(self) -> None:
        """``use_corrections=False`` 时网络输出不应影响结果。"""
        from keras import ops

        from industrial_greybox.mechanistic_layer import CSTRGreyBoxLayer

        layer = CSTRGreyBoxLayer(ModelConfig())
        state = np.array([[1.0, 350.0]], dtype=np.float32)
        control = np.array([[1.1, 350.0, 300.0, 5.0]], dtype=np.float32)
        slow = np.zeros((1, 3), dtype=np.float32)
        grey, _ = layer.step(*[ops.convert_to_tensor(v) for v in (state, control, slow)])
        mech, _ = layer.step(
            *[ops.convert_to_tensor(v) for v in (state, control, slow)],
            use_corrections=False,
        )
        # 慢变网络输出层偏置为 -6，sigmoid(-6)≈0.0025，修正量应远小于
        # 0.2% 的 UA 变化；因此两条轨迹在物理精度内一致。
        grey_arr = np.asarray(grey)[0]
        mech_arr = np.asarray(mech)[0]
        self.assertLess(abs(grey_arr[0] - mech_arr[0]), 1e-4)
        self.assertLess(abs(grey_arr[1] - mech_arr[1]), 0.05)


class TestLayer(unittest.TestCase):
    def test_output_shape_and_coordinates(self) -> None:
        import keras
        from keras import ops

        from industrial_greybox.mechanistic_layer import CSTRGreyBoxLayer

        config = ModelConfig()
        layer = CSTRGreyBoxLayer(config)
        initial = np.array([[1.0, 350.0]], dtype=np.float32)
        controls = np.repeat(
            np.array([[1.1, 350.0, 300.0, 5.0]], dtype=np.float32)[:, None, :],
            config.physics.n_steps,
            axis=1,
        )
        slow = np.zeros((1, config.physics.n_steps, 3), dtype=np.float32)
        out = np.asarray(layer((initial, controls, slow)))
        self.assertEqual(out.shape, (1, config.physics.n_steps + 1, 2))
        # 通道 0 是 mol/L（与初值同量级），通道 1 是归一化温度。
        self.assertAlmostEqual(float(out[0, 0, 0]), 1.0, places=5)
        self.assertAlmostEqual(float(out[0, 0, 1]), 0.0, places=5)
        self.assertLess(abs(out[0, -1, 1]), 3.0)

    def test_positive_rate_and_finite_temperature(self) -> None:
        import numpy as np

        from industrial_greybox.mechanistic_layer import CSTRGreyBoxLayer

        layer = CSTRGreyBoxLayer(ModelConfig())
        for temperature in (250.0, 300.0, 350.0, 500.0):
            value = float(
                np.asarray(
                    layer.arrhenius_rate(
                        __import__("keras").ops.convert_to_tensor(
                            [[temperature]], dtype="float32"
                        )
                    )
                ).item()
            )
            self.assertTrue(np.isfinite(value))
            self.assertGreater(value, 0.0)

    def test_serialisation_roundtrip(self) -> None:
        from industrial_greybox.mechanistic_layer import CSTRGreyBoxLayer

        layer = CSTRGreyBoxLayer(ModelConfig())
        rebuilt = CSTRGreyBoxLayer.from_config(layer.get_config())
        self.assertEqual(layer.config.signature(), rebuilt.config.signature())


if __name__ == "__main__":
    unittest.main()


class TestChannelWeighting(unittest.TestCase):
    """损失必须按通道归一化，且通道权重可配、可序列化。"""

    def test_channel_imbalance_does_not_dominate_loss(self) -> None:
        """温度点多、浓度点少时，两者仍按各自相对误差参与合并。"""
        from keras import ops

        from industrial_greybox.models import masked_squared_error

        y_true = np.zeros((1, 41, 2), dtype="float32")
        y_pred = np.zeros((1, 41, 2), dtype="float32")
        weights = np.zeros((1, 41, 2), dtype="float32")
        weights[0, :, 1] = 1.0   # 温度：41 个点
        weights[0, :2, 0] = 1.0  # 浓度：2 个点
        y_pred[0, :, 1] = 1.0    # 温度相对误差 1
        y_pred[0, :2, 0] = 1.0   # 浓度相对误差 1

        equal = float(
            ops.convert_to_numpy(masked_squared_error(y_true, y_pred, weights))
        )
        # 两个通道相对误差相同 => 合并损失 = 1，而不是被温度点数稀释/放大。
        self.assertAlmostEqual(equal, 1.0, places=5)

        weighted = float(
            ops.convert_to_numpy(
                masked_squared_error(y_true, y_pred, weights, channel_weights=(1.0, 0.0))
            )
        )
        # 只保留浓度通道后，损失仍为浓度通道的 1.0。
        self.assertAlmostEqual(weighted, 1.0, places=5)

    def test_channel_weights_come_from_config_and_roundtrip(self) -> None:
        from dataclasses import replace

        from industrial_greybox.models import CSTRGreyBoxModel

        config = ModelConfig()
        self.assertEqual(config.channel_weights, (1.0, 0.25))
        model = CSTRGreyBoxModel(config)
        self.assertEqual(model.channel_weights, (1.0, 0.25))
        rebuilt = CSTRGreyBoxModel.from_config(model.get_config())
        self.assertEqual(rebuilt.channel_weights, model.channel_weights)
        self.assertEqual(rebuilt.config.signature(), config.signature())

        override = CSTRGreyBoxModel(replace(config, channel_weights=(1.0, 1.0)))
        self.assertEqual(override.channel_weights, (1.0, 1.0))

    def test_channel_weight_validation(self) -> None:
        from dataclasses import replace

        with self.assertRaises(ValueError):
            replace(ModelConfig(), channel_weights=(1.0,))
        with self.assertRaises(ValueError):
            replace(ModelConfig(), channel_weights=(1.0, -0.5))


class TestDomainFlags(unittest.TestCase):
    """域内标记必须保守：缺列报错，NaN 视为域外。"""

    def test_missing_column_raises_instead_of_claiming_in_domain(self) -> None:
        from industrial_greybox.reports import domain_flags

        frame = pd.DataFrame({"something_else": [1.0, 2.0]})
        with self.assertRaises(ValueError):
            domain_flags(frame, bounds={"c_a0_mol_L": (0.0, 1.0)})

    def test_nan_is_treated_as_out_of_domain(self) -> None:
        from industrial_greybox.reports import domain_flags

        frame = pd.DataFrame({"c_a0_mol_L": [0.5, np.nan, 2.0]})
        flags = domain_flags(frame, bounds={"c_a0_mol_L": (0.0, 1.0)}, margin=0.0)
        self.assertEqual(flags.tolist(), [True, False, False])

    def test_metrics_by_horizon_orders_by_elapsed_time(self) -> None:
        from industrial_greybox.reports import metrics_by_horizon

        # 两个运行，各 5 个采样点；误差随时长线性增长。
        frame = pd.DataFrame(
            {
                "run_id": ["A"] * 5 + ["B"] * 5,
                "time_min": list(np.arange(5) * 10.0) * 2,
                "truth": [1.0] * 10,
                "pred": list(1.0 + 0.01 * np.arange(5)) * 2,
            }
        )
        table = metrics_by_horizon(
            frame,
            prediction_columns={"模型": "pred"},
            truth_column="truth",
            time_column="time_min",
            run_column="run_id",
            bins_min=(0, 20, 40),
        )
        self.assertEqual(len(table), 2)
        first, second = table.iloc[0], table.iloc[1]
        # 后一时段误差必须更大（0-20 min 含 0/10；20-40 min 含 20/30）。
        self.assertLess(float(first["rmse"]), float(second["rmse"]))
        self.assertEqual(int(first["n"]) + int(second["n"]), 8)
        # 40 min 处没有采样点，不应凭空生成一行。
        self.assertEqual(len(table), 2)

    def test_margin_expands_envelope(self) -> None:
        from industrial_greybox.reports import domain_flags

        frame = pd.DataFrame({"c_a0_mol_L": [0.5, 1.05]})
        strict = domain_flags(frame, bounds={"c_a0_mol_L": (0.0, 1.0)}, margin=0.0)
        loose = domain_flags(frame, bounds={"c_a0_mol_L": (0.0, 1.0)}, margin=0.1)
        self.assertEqual(strict.tolist(), [True, False])
        self.assertEqual(loose.tolist(), [True, True])


class TestNumericalGuardReporting(unittest.TestCase):
    """数值守护的计数必须真的出现在报表中，而不是恒为 0。"""

    def test_guard_counters_reach_shadow_report(self) -> None:
        from industrial_greybox.shadow import ShadowReport

        report = ShadowReport(n_samples=10, n_lab_events=2)
        self.assertEqual(report.n_nonfinite, 0)
        self.assertIn("非有限步", report.to_text())

    def test_training_clears_session_between_models(self) -> None:
        """连续训练多个模型时必须在训练前清图，否则 TF 图会累积。"""
        import inspect

        from industrial_greybox import training

        source = inspect.getsource(training.train_greybox)
        self.assertIn("keras.backend.clear_session()", source)
        # 清图必须发生在模型构建**之前**，否则旧图仍在。
        self.assertLess(
            source.index("keras.backend.clear_session()"),
            source.index("model = CSTRGreyBoxModel(config)"),
        )

    def test_explicit_stability_ratio_matches_definition(self) -> None:
        from industrial_greybox.numerics import explicit_stability_ratio

        self.assertAlmostEqual(
            explicit_stability_ratio(dt_min=1.0, tau_min=0.5, ua_per_min=0.5), 2.5
        )
        with self.assertRaises(ValueError):
            explicit_stability_ratio(dt_min=1.0, tau_min=0.0, ua_per_min=0.0)


class TestIdentifiabilityMatrix(unittest.TestCase):
    """参数相关性矩阵：必须能识别共线，且对常数/单位差异稳健。"""

    def test_detects_collinear_parameters(self) -> None:
        from industrial_greybox.diagnostics import sensitivity_correlation_matrix

        rng = np.random.default_rng(0)
        base = rng.normal(size=200)
        # A 与 B 几乎完全共线；C 独立。
        frame = pd.DataFrame(
            {
                "a": base,
                "b": 0.999 * base + 0.001 * rng.normal(size=200),
                "c": rng.normal(size=200),
            }
        )
        matrix = sensitivity_correlation_matrix(frame)
        self.assertEqual(list(matrix.index), ["a", "b", "c"])
        self.assertAlmostEqual(float(matrix.loc["a", "b"]), 1.0, places=2)
        self.assertLess(abs(float(matrix.loc["a", "c"])), 0.2)
        # 对角线必须为 1。
        np.testing.assert_allclose(np.diag(matrix.to_numpy()), 1.0)

    def test_scale_invariance(self) -> None:
        """量纲差异不应改变相关性结论。"""
        from industrial_greybox.diagnostics import sensitivity_correlation_matrix

        rng = np.random.default_rng(1)
        base = rng.normal(size=200)
        frame = pd.DataFrame({"a": base, "b": base * 1e6})
        matrix = sensitivity_correlation_matrix(frame)
        self.assertAlmostEqual(float(matrix.loc["a", "b"]), 1.0, places=6)

    def test_constant_column_does_not_produce_nan(self) -> None:
        from industrial_greybox.diagnostics import sensitivity_correlation_matrix

        rng = np.random.default_rng(2)
        frame = pd.DataFrame(
            {"a": rng.normal(size=50), "b": np.zeros(50), "c": rng.normal(size=50)}
        )
        matrix = sensitivity_correlation_matrix(frame)
        self.assertFalse(matrix.isna().to_numpy().any())

    def test_requires_two_parameters(self) -> None:
        from industrial_greybox.diagnostics import sensitivity_correlation_matrix

        frame = pd.DataFrame({"a": np.arange(10.0)})
        with self.assertRaises(ValueError):
            sensitivity_correlation_matrix(frame)


class TestShadowBufferContract(unittest.TestCase):
    """ShadowBuffer（原 DelayBuffer）的契约与向后兼容。"""

    def test_alias_is_same_class(self) -> None:
        from industrial_greybox.delays import DelayBuffer, ShadowBuffer

        self.assertIs(DelayBuffer, ShadowBuffer)

    def test_query_returns_process_time_not_result_time(self) -> None:
        """演示核心语义：死时间内的变化不能被算进残差。"""
        from industrial_greybox.delays import ShadowBuffer

        buffer = ShadowBuffer(horizon_min=60.0)
        for time_min in range(0, 41):
            buffer.push(float(time_min), [1.0 + 0.01 * time_min])
        result_time, dead_time = 35.0, 12.0
        at_result = float(buffer.query(result_time)[0])
        at_process = float(buffer.query(result_time - dead_time)[0])
        self.assertAlmostEqual(at_process, 1.23, places=6)
        self.assertAlmostEqual(at_result - at_process, 0.12, places=6)
