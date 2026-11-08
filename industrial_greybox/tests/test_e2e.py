"""端到端回归测试：训练后检查可比性、掩码语义与影子运行因果性。

这些测试比 ``test_units`` 慢（需要训练几个 epoch），因此用很小的
campaign 和 epoch 数，目标是**发现结构性问题**，不是复现论文指标：

- 灰箱不比“同配置但无修正网络”的基线差一个数量级以上；
- 训练后的慢变修正没有把 UA 推到名义值之外很远；
- 影子日志的残差在死时间对齐后没有系统性滞后偏差；
- 掩码损失在“未监督点有巨大误差”时仍接近 0。
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
    COL_RUN_ID,
    COL_TIME_MIN,
    DelayConfig,
    ModelConfig,
    ShadowConfig,
)
from industrial_greybox.models import CSTRGreyBoxModel, masked_squared_error
from industrial_greybox.pipeline import run_pipeline
from industrial_greybox.synthetic_plant import CampaignSpec, simulate_campaign


class TestMaskedLossSemantics(unittest.TestCase):
    """掩码必须真正屏蔽未监督点（Keras 3 的 sample_weight 陷阱）。"""

    def test_unmasked_points_do_not_contribute(self) -> None:
        from keras import ops

        y_true = np.array([[[0.0, 1.0], [0.0, 1.0], [0.5, 2.0]]], dtype="float32")
        y_pred = np.array([[[99.0, 1.0], [99.0, 1.0], [0.5, 2.0]]], dtype="float32")
        weights = np.array([[[0.0, 1.0], [0.0, 1.0], [1.0, 1.0]]], dtype="float32")
        value = float(ops.convert_to_numpy(masked_squared_error(y_true, y_pred, weights)))
        self.assertAlmostEqual(value, 0.0, places=6)

    def test_weighted_magnitude_is_correct(self) -> None:
        from keras import ops

        y_true = np.zeros((1, 2, 1), dtype="float32")
        y_pred = np.ones((1, 2, 1), dtype="float32")
        weights = np.ones((1, 2, 1), dtype="float32")
        value = float(ops.convert_to_numpy(masked_squared_error(y_true, y_pred, weights)))
        self.assertAlmostEqual(value, 1.0, places=6)

        weights_half = weights * 0.5
        value_half = float(
            ops.convert_to_numpy(masked_squared_error(y_true, y_pred, weights_half))
        )
        # 权重同时出现在分子与分母，等比例缩放不改变归一化损失。
        self.assertAlmostEqual(value_half, 1.0, places=6)

    def test_compute_loss_rejects_mismatched_weight_shape(self) -> None:
        from keras import ops

        model = CSTRGreyBoxModel(ModelConfig())
        y = np.zeros((2, 5, 2), dtype="float32")
        y_pred = np.zeros((2, 5, 2), dtype="float32")
        weights = np.ones((2, 5), dtype="float32")
        with self.assertRaises(ValueError):
            model.compute_loss(
                y=y,
                y_pred=ops.convert_to_tensor(y_pred),
                sample_weight=ops.convert_to_tensor(weights),
            )


class TestPipelineRegression(unittest.TestCase):
    """小规模端到端训练：只检查“没有结构性问题”，不比较绝对精度。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.dcs, cls.lab = simulate_campaign(
            CampaignSpec(
                n_runs=9,
                n_steps=160,
                days_span=20,
                lab_interval_min=10,
                inject_faults=True,
            )
        )
        # 影子运行使用默认 ShadowConfig（重锚开启），模型配置单独保存以便
        # 测试读取偏差限值。
        cls.result_cfg = ModelConfig()
        cls.result = run_pipeline(
            cls.dcs,
            cls.lab,
            config=ModelConfig(),
            delay=DelayConfig(),
            epochs=4,
            batch_size=32,
            stride=8,
            run_baseline=True,
            run_shadow=True,
            verbose=0,
            shadow=ShadowConfig(use_temperature_feedback=True),
        )

    def test_no_numerical_guard_events(self) -> None:
        health = self.result.training.health_records
        self.assertFalse(health.empty)
        self.assertTrue((health["nonfinite_steps"] == 0).all())
        self.assertTrue((health["clipped_steps"] == 0).all())

    def test_parameters_stay_physical(self) -> None:
        params = self.result.training.model.fitted_parameters()
        self.assertGreater(params["k_ref_per_min"], 0.0)
        self.assertGreater(params["ua_per_min"], 0.0)
        # 学习后不应偏离名义值多个数量级（4 个 epoch 的实际预期范围）。
        self.assertLess(params["k_ref_per_min"], 1.0)
        self.assertLess(params["ua_per_min"], 5.0)

    def test_greybox_not_catastrophically_worse_than_baseline(self) -> None:
        """灰箱允许略差（小样本/少 epoch），但不能差一个数量级。"""
        rows = self.result.metrics.set_index("模型")
        grey = float(rows.loc["灰箱", "C_A RMSE (mol/L)"])
        mech = float(rows.loc["机理基线", "C_A RMSE (mol/L)"])
        self.assertTrue(np.isfinite(grey) and np.isfinite(mech))
        self.assertLess(grey, max(10.0 * mech, 0.5))

    def test_shadow_log_has_expected_columns_and_no_stale_history(self) -> None:
        log = self.result.shadow_log
        self.assertFalse(log.empty)
        for column in (
            "mechanistic_c_a",
            "greybox_c_a",
            "correction_c_a",
            "lab_c_a",
            "residual_vs_lab",
            "in_domain",
            "config_signature",
        ):
            self.assertIn(column, log.columns)
        report = self.result.shadow_report
        self.assertEqual(report.n_stale_history, 0)
        # 合成数据带独立真值列，因此 truth_c_a / residual_vs_truth 必须有效。
        self.assertGreater(int(log["truth_c_a"].notna().sum()), 0)
        # 独立真值是逐点记录的（不同于稀疏化验），因此 residual_vs_truth
        # 应覆盖全部采样点。
        self.assertEqual(
            int(log["residual_vs_truth"].notna().sum()), len(log)
        )
        # 数值守护计数必须反映真实情况（此处不应有发散/裁剪）。
        self.assertEqual(report.n_nonfinite, 0)
        self.assertEqual(report.n_clipped, 0)

    def test_shadow_residual_is_not_systematically_lagged(self) -> None:
        """死时间对齐正确时，残差不应随过程时刻单调漂移。

        若对齐方向反了（用当前预测减迟到化验），残差会与死时间内的过程
        变化强相关，表现为大的滞后偏差。这里用“前半段均值与后半段均值
        之差”做一个宽松的结构检查。
        """
        events = self.result.shadow_lab_events
        self.assertFalse(events.empty)
        residual = events["residual_grey"].dropna().to_numpy()
        self.assertGreater(len(residual), 4)
        half = len(residual) // 2
        drift = abs(float(np.mean(residual[:half]) - np.mean(residual[half:])))
        # 绝对阈值随数据规模变化；这里只要求不存在数量级差异。
        self.assertLess(drift, 0.2)

    def test_reanchoring_is_bounded_and_smoothed(self) -> None:
        """重锚必须：有界、且单次施加量不超过残差本身（指数平滑）。

        默认 ``reseed_mode="adaptive"``：施加量 = α·confidence·residual。
        因此除"限幅"外还有两条必须成立的性质：

        - 施加量绝对值 <= 残差绝对值（平滑不会放大修正）；
        - 施加量绝对值 <= ``max_bias_correction_mol_l``（硬边界）。
        """
        events = self.result.shadow_lab_events
        self.assertIn("applied_bias_mol_l", events.columns)
        from industrial_greybox.config import ShadowConfig

        default = ShadowConfig()
        limit = default.max_bias_correction_mol_l
        valid = events[events["applied_bias_mol_l"].notna()]
        self.assertGreater(len(valid), 0)

        applied = valid["applied_bias_mol_l"].to_numpy()
        residual = valid["residual_grey"].to_numpy()
        self.assertTrue(np.all(np.abs(applied) <= limit + 1e-6))
        self.assertTrue(np.all(np.abs(applied) <= np.abs(residual) + 1e-6))
        # 自适应模式应记录置信度；它必须是 [0, 1] 内的数。
        if "reanchor_confidence" in valid.columns:
            confidence = valid["reanchor_confidence"].dropna().to_numpy()
            self.assertTrue(np.all((confidence >= 0) & (confidence <= 1)))

    def test_shadow_lab_events_use_process_time(self) -> None:
        events = self.result.shadow_lab_events
        np.testing.assert_allclose(
            events["result_time_min"] - events["process_time_min"],
            events["dead_time_min"],
        )

    def test_split_has_no_run_leakage(self) -> None:
        groups = self.result.groups
        names = list(groups)
        for index, first in enumerate(names):
            for second in names[index + 1 :]:
                self.assertFalse(set(groups[first]) & set(groups[second]))
        for split, dataset in self.result.datasets.items():
            self.assertTrue(set(dataset.run_ids) <= set(groups[split]))


if __name__ == "__main__":
    unittest.main()
