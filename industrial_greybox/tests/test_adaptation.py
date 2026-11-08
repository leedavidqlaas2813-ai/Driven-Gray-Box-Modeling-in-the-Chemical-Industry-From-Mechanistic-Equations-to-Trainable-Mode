"""新增能力的单元测试：域外门控、滚动一致性、自适应重锚、在线微调。

与 ``test_units`` 分开，便于单独运行::

    python -m unittest industrial_greybox.tests.test_adaptation -v
"""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from dataclasses import replace

import numpy as np
import pandas as pd

from industrial_greybox.config import LongHorizonConfig, ModelConfig, ShadowConfig
from industrial_greybox.data import WindowedDataset


def _build_layer(layer) -> None:
    """触发机理层 build，使子网络（含 Dense.bias）真正创建。

    ``Dense.bias`` 只在 build 之后存在；未 build 时访问会 AttributeError。
    """
    initial = np.array([[1.0, 350.0]], dtype="float32")
    controls = np.tile(
        np.array([1.1, 350.0, 300.0, 5.0], dtype="float32"), (1, 5, 1)
    )
    slow = np.zeros((1, 5, 3), dtype="float32")
    layer((initial, controls, slow))


class TestDomainGate(unittest.TestCase):
    """域外渐近门控：域内完整生效，远域外平滑回退，慢变主趋势不回退。"""

    def _layer(self, **overrides):
        from industrial_greybox.mechanistic_layer import CSTRGreyBoxLayer

        config = replace(ModelConfig(), long_horizon=LongHorizonConfig(**overrides))
        return CSTRGreyBoxLayer(config)

    def test_gate_is_one_inside_training_box(self) -> None:
        from keras import ops

        layer = self._layer()
        features = ops.convert_to_tensor(
            [[1.0, 0.0, 0.0, 0.0, 0.0, -0.2]], dtype="float32"
        )
        self.assertAlmostEqual(
            float(np.asarray(layer.domain_gate(features))[0, 0]), 1.0
        )

    def test_gate_decays_to_zero_far_outside(self) -> None:
        from keras import ops

        layer = self._layer(domain_box=3.0, domain_taper=1.5)
        features = ops.convert_to_tensor(
            [[1.0, 0.0, 0.0, 0.0, 0.0, 20.0]], dtype="float32"
        )
        self.assertAlmostEqual(
            float(np.asarray(layer.domain_gate(features))[0, 0]), 0.0
        )

    def test_gate_is_monotone_and_smooth(self) -> None:
        from keras import ops

        layer = self._layer(domain_box=3.0, domain_taper=1.5)
        gates = []
        for distance in np.linspace(0.0, 10.0, 41):
            features = ops.convert_to_tensor(
                [[1.0, 0.0, 0.0, 0.0, 0.0, float(distance)]], dtype="float32"
            )
            gates.append(float(np.asarray(layer.domain_gate(features))[0, 0]))
        gates = np.asarray(gates)
        self.assertTrue(np.all(np.diff(gates) <= 1e-6), "门控必须单调不增")
        self.assertAlmostEqual(gates[0], 1.0)
        self.assertAlmostEqual(gates[-1], 0.0)
        self.assertTrue(np.any((gates > 0.05) & (gates < 0.95)), "需要真有过渡区")

    def test_disabled_gate_returns_one(self) -> None:
        from keras import ops

        layer = self._layer(domain_gate=False)
        features = ops.convert_to_tensor(
            [[1.0, 0.0, 0.0, 0.0, 0.0, 999.0]], dtype="float32"
        )
        self.assertAlmostEqual(
            float(np.asarray(layer.domain_gate(features))[0, 0]), 1.0
        )

    def test_drift_main_trend_does_not_decay_far_outside(self) -> None:
        """关键物理判断：老化主趋势在域外**不**回退到 0。

        若回退到 0，等于假设老装置换热面自动变干净，会系统性高估 UA、
        低估出口浓度——方向上危险。
        """
        from keras import ops

        layer = self._layer(domain_box=0.5, domain_taper=0.5)
        _build_layer(layer)
        # 把慢变网输出层偏置推向 0，使 sigmoid(0)=0.5，主趋势非零。
        layer.slow_net.layers[-1].bias.assign(np.zeros(5, dtype="float32"))
        near, _ = layer.slow_correction(
            ops.convert_to_tensor([[0.05, 0.05, 0.0]], dtype="float32"), 0
        )
        far, _ = layer.slow_correction(
            ops.convert_to_tensor([[0.95, 0.95, 0.0]], dtype="float32"), 0
        )
        near_value = float(np.asarray(near)[0, 0])
        far_value = float(np.asarray(far)[0, 0])
        self.assertLess(near_value, -0.1)
        self.assertLess(far_value, -0.1, "主趋势在域外不应衰减到 0")
        self.assertAlmostEqual(far_value, near_value, places=5)

    def test_residual_terms_do_decay_far_outside(self) -> None:
        """符号自由的残余项在域外**应**关闭。"""
        from keras import ops

        layer = self._layer(domain_box=0.5, domain_taper=0.5)
        _build_layer(layer)
        # 让残余项非零：把最后两个偏置设大。
        bias = np.array([-6.0, -6.0, 8.0, 8.0, 0.0], dtype="float32")
        layer.slow_net.layers[-1].bias.assign(bias)
        near, _ = layer.slow_correction(
            ops.convert_to_tensor([[0.05, 0.05, 0.0]], dtype="float32"), 0
        )
        far, _ = layer.slow_correction(
            ops.convert_to_tensor([[0.95, 0.95, 0.0]], dtype="float32"), 0
        )
        # 域外残余被门控关掉，只剩主趋势；域内则含残余。
        self.assertGreater(abs(float(np.asarray(near)[0, 0])), 0.0)
        self.assertLess(
            abs(float(np.asarray(far)[0, 0])), abs(float(np.asarray(near)[0, 0]))
        )


class TestRolloutConsistency(unittest.TestCase):
    """多步滚动一致性：自洽轨迹的惩罚应接近 0，篡改起点则明显非零。"""

    def _inputs(self):
        initial = np.array([[1.0, 350.0]], dtype="float32")
        controls = np.tile(
            np.array([1.1, 350.0, 300.0, 5.0], dtype="float32"), (1, 40, 1)
        )
        slow = np.zeros((1, 40, 3), dtype="float32")
        return initial, controls, slow

    def test_exact_anchor_reproduces_trajectory(self) -> None:
        """不加扰动时，确定性积分器必然精确复现（半群性质）。

        这条测试记录了一个**设计前提**：正因为如此，惩罚必须用扰动锚点，
        否则它恒为 0、什么也不约束。
        """
        from industrial_greybox.mechanistic_layer import CSTRGreyBoxLayer

        layer = CSTRGreyBoxLayer(ModelConfig())
        initial, controls, slow = self._inputs()
        trajectory = layer((initial, controls, slow), training=False)
        deviation = layer.rollout_consistency(
            initial, controls, slow, trajectory, start=10
        )
        self.assertLess(float(np.asarray(deviation)[0]), 1e-8)

    def test_perturbed_anchor_yields_nonzero_penalty(self) -> None:
        """扰动锚点必须产生非零惩罚——这是训练时唯一有梯度的情形。"""
        from keras import ops

        from industrial_greybox.mechanistic_layer import CSTRGreyBoxLayer

        layer = CSTRGreyBoxLayer(ModelConfig())
        initial, controls, slow = self._inputs()
        trajectory = layer((initial, controls, slow), training=False)
        deviation = layer.rollout_consistency(
            initial,
            controls,
            slow,
            trajectory,
            start=10,
            max_steps=10,
            anchor_perturbation=ops.convert_to_tensor([[0.01, 0.01]], dtype="float32"),
        )
        self.assertGreater(float(np.asarray(deviation)[0]), 0.0)

    def test_penalty_detects_corrupted_reference(self) -> None:
        from keras import ops

        from industrial_greybox.mechanistic_layer import CSTRGreyBoxLayer

        layer = CSTRGreyBoxLayer(ModelConfig())
        initial, controls, slow = self._inputs()
        trajectory = np.array(layer((initial, controls, slow), training=False))
        trajectory[:, 10:, :] += 0.2
        deviation = layer.rollout_consistency(
            initial, controls, slow, ops.convert_to_tensor(trajectory), start=10
        )
        self.assertGreater(float(np.asarray(deviation)[0]), 1e-3)


class TestAdaptiveReanchor(unittest.TestCase):
    """自适应重锚：坏值忽略、漂移确认、平滑、限幅、稳健尺度。"""

    def _reanchor(self, **overrides):
        from industrial_greybox.reanchor import AdaptiveReanchor

        return AdaptiveReanchor(replace(ShadowConfig(), **overrides))

    def test_isolated_spike_is_ignored(self) -> None:
        reanchor = self._reanchor()
        for _ in range(5):
            reanchor.update(0.005)
        decision = reanchor.update(0.5)
        self.assertFalse(decision.drift_confirmed)
        self.assertAlmostEqual(decision.confidence, 0.0)
        self.assertAlmostEqual(decision.applied_bias_mol_l, 0.0)

    def test_sustained_drift_is_confirmed(self) -> None:
        """持续的大残差应被确认为漂移，从而恢复置信度。

        用 0.5 mol/L（远超门控阈值）而不是 0.05：后者约 2.7σ，本来就在
        门控阈值内，修复会被正常施加，**不需要**漂移确认。漂移确认的意义
        恰恰在于"门控本会把它当坏值关掉"的那种大而持续的偏差。
        """
        reanchor = self._reanchor(drift_confirm_count=3)
        decisions = [reanchor.update(0.5) for _ in range(4)]
        # 第一次：门控关闭（当作可能的坏值）。
        self.assertFalse(decisions[0].drift_confirmed)
        self.assertAlmostEqual(decisions[0].applied_bias_mol_l, 0.0)
        # 连续出现后：确认为漂移，置信度恢复，修正被施加。
        self.assertTrue(decisions[-1].drift_confirmed)
        self.assertGreater(abs(decisions[-1].applied_bias_mol_l), 0.0)

    def test_smoothing_never_overshoots_residual(self) -> None:
        reanchor = self._reanchor()
        for _ in range(5):
            reanchor.update(0.004)
        residual = 0.02
        decision = reanchor.update(residual)
        self.assertLessEqual(abs(decision.applied_bias_mol_l), abs(residual) + 1e-12)

    def test_applied_bias_respects_limit(self) -> None:
        reanchor = self._reanchor(
            max_bias_correction_mol_l=0.01, reseed_mode="fixed"
        )
        decision = reanchor.update(5.0)
        self.assertAlmostEqual(abs(decision.applied_bias_mol_l), 0.01)
        self.assertTrue(decision.clipped)

    def test_non_finite_residual_is_ignored(self) -> None:
        reanchor = self._reanchor()
        decision = reanchor.update(float("nan"))
        self.assertEqual(decision.applied_bias_mol_l, 0.0)
        self.assertEqual(decision.confidence, 0.0)

    def test_fixed_mode_matches_legacy(self) -> None:
        reanchor = self._reanchor(reseed_mode="fixed")
        self.assertAlmostEqual(reanchor.update(0.05).applied_bias_mol_l, 0.05)

    def test_scale_is_robust_to_single_outlier(self) -> None:
        reanchor = self._reanchor(residual_scale_window=20)
        for value in (0.01, -0.01, 0.011, -0.009, 0.010):
            reanchor.update(value)
        baseline = reanchor.scale
        reanchor.update(0.5)
        self.assertLess(reanchor.scale, baseline * 5.0)

    def test_reset_clears_run_state(self) -> None:
        reanchor = self._reanchor(drift_confirm_count=2)
        reanchor.update(0.5)
        reanchor.update(0.5)
        reanchor.reset()
        self.assertFalse(reanchor.update(0.5).drift_confirmed)


class TestOnlineAdapter(unittest.TestCase):
    """在线微调：冻结机理核心、熔断回滚、审计留痕。"""

    @staticmethod
    def _dataset(value: float = 0.9, n: int = 6, seed: int = 0, steps: int = 20):
        rng = np.random.default_rng(seed)
        initial = rng.uniform(0.8, 1.2, (n, 2)).astype("float32")
        initial[:, 1] = 350.0
        controls = np.tile(
            np.array([1.1, 350.0, 300.0, 5.0], dtype="float32"), (n, steps, 1)
        )
        slow = np.zeros((n, steps, 3), dtype="float32")
        targets = np.concatenate(
            [
                np.full((n, steps + 1, 1), value, dtype="float32"),
                np.zeros((n, steps + 1, 1), dtype="float32"),
            ],
            axis=-1,
        )
        weights = np.concatenate(
            [
                np.ones((n, steps + 1, 1), dtype="float32"),
                np.full((n, steps + 1, 1), 0.25, dtype="float32"),
            ],
            axis=-1,
        )
        return WindowedDataset(
            initial_states=initial,
            controls=controls,
            slow=slow,
            targets=targets,
            weights=weights,
            run_ids=np.array(["R"] * n),
            window_start_times=np.zeros(n, dtype="float32"),
        )

    def _model(self):
        from industrial_greybox.models import CSTRGreyBoxModel, compile_greybox

        config = replace(ModelConfig(), physics=replace(ModelConfig().physics, n_steps=20))
        return compile_greybox(CSTRGreyBoxModel(config), learning_rate=1e-3)

    def test_physics_weights_frozen_during_adaptation(self) -> None:
        from industrial_greybox.adaptation import (
            AdaptationPolicy,
            OnlineAdapter,
            physics_weights,
        )

        model = self._model()
        adapter = OnlineAdapter(model, AdaptationPolicy(trigger_patience=1))
        before = [float(np.asarray(weight)) for weight in physics_weights(model)]
        adapter.observe(1.0)
        adapter.adapt(self._dataset(0.95, seed=1), self._dataset(0.9, seed=2))
        after = [float(np.asarray(weight)) for weight in physics_weights(model)]
        np.testing.assert_allclose(before, after)
        self.assertTrue(
            all(weight.trainable for weight in physics_weights(model)),
            "微调后必须恢复可训练标记，否则污染后续离线训练",
        )

    def test_rollback_restores_correction_weights(self) -> None:
        from industrial_greybox.adaptation import (
            AdaptationPolicy,
            OnlineAdapter,
            correction_weights,
        )

        model = self._model()
        # 用极小的绝对 RMSE 上限强制熔断：任何非零误差都会触发回滚。
        # （不能用负的 max_rmse_regression——配置校验会正确拒绝它。）
        policy = AdaptationPolicy(
            trigger_patience=1,
            max_rmse_absolute_mol_l=1e-9,
        )
        adapter = OnlineAdapter(model, policy)
        snapshot = [np.array(weight) for weight in correction_weights(model)]
        adapter.observe(1.0)
        event = adapter.adapt(self._dataset(0.95, seed=3), self._dataset(0.9, seed=4))
        self.assertTrue(event.rolled_back)
        for variable, expected in zip(correction_weights(model), snapshot):
            np.testing.assert_allclose(np.asarray(variable), expected, atol=1e-7)

    def test_no_adaptation_without_trigger(self) -> None:
        from industrial_greybox.adaptation import AdaptationPolicy, OnlineAdapter

        model = self._model()
        adapter = OnlineAdapter(model, AdaptationPolicy(trigger_patience=2))
        self.assertFalse(adapter.observe(0.001))
        self.assertFalse(adapter.observe(0.001))
        event = adapter.adapt(self._dataset(seed=5), self._dataset(seed=6))
        self.assertFalse(event.triggered)

    def test_history_frame_has_audit_columns(self) -> None:
        from industrial_greybox.adaptation import AdaptationPolicy, OnlineAdapter

        model = self._model()
        adapter = OnlineAdapter(model, AdaptationPolicy(trigger_patience=1))
        adapter.observe(1.0)
        adapter.adapt(self._dataset(0.92, seed=7), self._dataset(0.9, seed=8))
        frame = adapter.state.history_frame()
        self.assertFalse(frame.empty)
        for column in ("guard_rmse_before", "guard_rmse_after", "rolled_back"):
            self.assertIn(column, frame.columns)


if __name__ == "__main__":
    unittest.main()


class TestDomainBoxCalibration(unittest.TestCase):
    """domain_box 必须从训练数据校准，且对毛刺稳健。"""

    def test_calibration_is_robust_to_spikes(self) -> None:
        """分位数校准不应被仪表毛刺（max）主导。

        这是实际踩过的坑：用 max 校准时，在线仪表的孤立毛刺会把 box
        抬到 17，门控永远不触发。
        """
        from industrial_greybox.config import COL_C_A0, COL_T_C, COL_T_IN, COL_TAU
        from industrial_greybox.config import COL_T_REACTOR
        from industrial_greybox.diagnostics import calibrate_domain_box

        rng = np.random.default_rng(0)
        n = 300
        frame = pd.DataFrame(
            {
                COL_C_A0: rng.uniform(0.9, 1.3, n),
                COL_T_IN: rng.uniform(340, 350, n),
                COL_T_C: rng.uniform(290, 300, n),
                COL_TAU: rng.uniform(4, 6, n),
                COL_T_REACTOR: rng.uniform(345, 355, n),
                "c_a_out_mol_L": rng.uniform(0.8, 1.2, n),
            }
        )
        clean = calibrate_domain_box(frame, scaling=ModelConfig().scaling)

        # 注入 3 个极端毛刺。
        spiked = frame.copy()
        spiked.loc[:2, COL_T_REACTOR] = 900.0
        robust = calibrate_domain_box(spiked, scaling=ModelConfig().scaling)
        self.assertLess(
            robust,
            clean * 1.5,
            "毛刺不应显著改变 domain_box（必须用分位数而非 max）",
        )

    def test_calibration_matches_feature_normalisation(self) -> None:
        """校准值应与 _rate_features 的归一化口径一致。"""
        from industrial_greybox.diagnostics import calibrate_domain_box
        from industrial_greybox.models import CSTRGreyBoxModel
        from keras import ops

        frame = pd.DataFrame(
            {
                "c_a0_mol_L": [1.1] * 50,
                "feed_temperature_K": [350.0] * 50,
                "coolant_temperature_K": [300.0] * 50,
                "residence_time_min": [5.0] * 50,
                "reactor_temperature_K": [350.0] * 50,
                "c_a_out_mol_L": [1.0] * 50,
            }
        )
        box = calibrate_domain_box(frame, scaling=ModelConfig().scaling)
        # 该工况下浓度/温度/进料/冷却都恰在参考点（|z|=0），但停留时间
        # 5.0 min 相对参考 5.5 min 有 -0.2 的偏离；乘安全系数 1.2 得 0.24。
        self.assertAlmostEqual(box, 0.24, places=6)
        # 且这种工况的门控应完全打开。
        layer = CSTRGreyBoxModel(ModelConfig()).physics_layer
        state = ops.convert_to_tensor([[1.0, 350.0]], dtype="float32")
        control = ops.convert_to_tensor([[1.1, 350.0, 300.0, 5.0]], dtype="float32")
        self.assertAlmostEqual(layer.domain_status(state, control, np.zeros((1, 3)))["rate_gate"], 1.0)


class TestPenaltyNormalisation(unittest.TestCase):
    """滚动惩罚的归一化必须让权重表达"占主损失的比例"，可跨装置迁移。"""

    def test_default_weight_is_meaningful_fraction_of_loss(self) -> None:
        from industrial_greybox.models import CSTRGreyBoxModel
        from keras import ops

        model = CSTRGreyBoxModel(ModelConfig())
        initial = np.tile(np.array([[1.0, 350.0]], dtype="float32"), (2, 1))
        controls = np.tile(
            np.array([1.1, 350.0, 300.0, 5.0], dtype="float32"), (2, 40, 1)
        )
        slow = np.zeros((2, 40, 3), dtype="float32")
        output = model((initial, controls, slow), training=False)
        penalty = float(
            np.asarray(
                model._rollout_penalty(
                    (
                        ops.convert_to_tensor(initial),
                        ops.convert_to_tensor(controls),
                        ops.convert_to_tensor(slow),
                    ),
                    output,
                )
            )
        )
        # 归一化后应在 O(1e-2) 量级，而不是 1e-6。
        self.assertGreater(penalty, 1e-3)
        self.assertLess(penalty, 1.0)

    def test_normalisation_can_be_disabled(self) -> None:
        from dataclasses import replace

        from industrial_greybox.models import CSTRGreyBoxModel
        from keras import ops

        config = replace(
            ModelConfig(),
            long_horizon=replace(
                ModelConfig().long_horizon, rollout_penalty_normalize=False
            ),
        )
        model = CSTRGreyBoxModel(config)
        initial = np.tile(np.array([[1.0, 350.0]], dtype="float32"), (2, 1))
        controls = np.tile(
            np.array([1.1, 350.0, 300.0, 5.0], dtype="float32"), (2, 40, 1)
        )
        slow = np.zeros((2, 40, 3), dtype="float32")
        output = model((initial, controls, slow), training=False)
        penalty = float(
            np.asarray(
                model._rollout_penalty(
                    (
                        ops.convert_to_tensor(initial),
                        ops.convert_to_tensor(controls),
                        ops.convert_to_tensor(slow),
                    ),
                    output,
                )
            )
        )
        # 关闭归一化后退回原始小量级。
        self.assertLess(penalty, 1e-4)
