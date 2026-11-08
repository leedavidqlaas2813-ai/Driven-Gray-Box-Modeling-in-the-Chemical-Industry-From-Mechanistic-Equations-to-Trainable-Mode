"""验证新增机制在**它该起作用的条件**下是否真的有效。

两个问题需要分开回答：

1. 滚动一致性惩罚有效吗？
   默认权重 0.1 产生的惩罚量级约 5e-6，远小于数据拟合损失 ~0.07。
   要判断机制本身是否有效，必须扫描权重，看影子指标是否随之改善。

2. 域外门控有效吗？
   域内测试数据上门控**从不触发**（rate_gate 恒为 1），所以必须用
   **外推数据**（hot_shift）验证：外推时门控应关闭修正网络，使预测
   退回机理基线，从而不劣于甚至优于「无门控」版本。

运行::

    python tools/verify_mechanisms.py --out verify.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--out", default="verify_mechanisms.json")
    args = parser.parse_args()

    import numpy as np

    from industrial_greybox.config import (
        DelayConfig,
        LongHorizonConfig,
        ModelConfig,
        ShadowConfig,
    )
    from industrial_greybox.data import frame_to_windows
    from industrial_greybox.pipeline import prepare_frame, run_pipeline
    from industrial_greybox.synthetic_plant import CampaignSpec, simulate_campaign

    train_data = simulate_campaign(
        CampaignSpec(n_runs=10, n_steps=200, days_span=30, seed=1)
    )
    config = ModelConfig()
    results: dict[str, object] = {}

    # ------------------------------------------------------------------
    # 问题 1：滚动惩罚权重扫描（影子指标）
    # ------------------------------------------------------------------
    weights = [0.0, 0.1, 1.0, 3.0]
    sweep = []
    for weight in weights:
        cfg = replace(
            config,
            long_horizon=replace(
                config.long_horizon,
                rollout_penalty=weight > 0,
                rollout_penalty_weight=weight,
                domain_gate=False,
            ),
        )
        result = run_pipeline(
            *train_data,
            config=cfg,
            delay=DelayConfig(),
            epochs=args.epochs,
            stride=args.stride,
            verbose=0,
            run_baseline=False,
            run_shadow=True,
            shadow=ShadowConfig(reseed_mode="adaptive"),
        )
        table = result.shadow_report.generalisation.set_index("模型")
        sweep.append(
            {
                "weight": weight,
                "shadow_greybox": float(table.loc["灰箱", "rmse"]),
                "shadow_mechanistic": float(table.loc["机理基线", "rmse"]),
            }
        )
        print(
            f"weight={weight}: 影子灰箱={sweep[-1]['shadow_greybox']:.5f} "
            f"机理={sweep[-1]['shadow_mechanistic']:.5f}",
            flush=True,
        )
    results["rollout_weight_sweep"] = sweep

    # ------------------------------------------------------------------
    # 问题 2：域外门控（外推数据）
    # ------------------------------------------------------------------
    hot_data = simulate_campaign(
        CampaignSpec(n_runs=4, n_steps=200, days_span=10, seed=2, hot_shift=True)
    )
    gate_rows = []
    for gate_enabled in (False, True):
        cfg = replace(
            config,
            long_horizon=replace(
                config.long_horizon,
                rollout_penalty=False,
                domain_gate=gate_enabled,
            ),
        )
        result = run_pipeline(
            *train_data,
            config=cfg,
            delay=DelayConfig(),
            epochs=args.epochs,
            stride=args.stride,
            verbose=0,
            run_baseline=False,
            run_shadow=False,
        )
        # 用外推数据评估开环误差。
        frame, _ = prepare_frame(*hot_data, config=cfg, delay=DelayConfig())
        windows = frame_to_windows(
            frame,
            physics=cfg.physics,
            slow_reference=(
                cfg.slow_drift.age_ref_days,
                cfg.slow_drift.feed_ref_m3,
            ),
            stride=20,
        )
        model = result.training.model
        prediction = np.asarray(
            model(
                (
                    windows.initial_states,
                    windows.controls,
                    windows.slow,
                ),
                training=False,
            )
        )
        from industrial_greybox.reports import regression_metrics

        metrics = regression_metrics(
            prediction[:, :, 0],
            windows.targets[:, :, 0],
            mask=windows.weights[:, :, 0],
        )
        # 门控强度（取前若干窗口）。
        layer = model.physics_layer
        gates = [
            layer.domain_status(
                windows.initial_states[i : i + 1],
                windows.controls[i, 0:1],
                windows.slow[i, 0:1],
            )["rate_gate"]
            for i in range(min(10, len(windows)))
        ]
        gate_rows.append(
            {
                "domain_gate": gate_enabled,
                "extrapolation_rmse": float(metrics["rmse"]),
                "mean_rate_gate": float(np.mean(gates)),
            }
        )
        print(
            f"domain_gate={gate_enabled}: 外推 RMSE={gate_rows[-1]['extrapolation_rmse']:.5f} "
            f"平均 rate_gate={gate_rows[-1]['mean_rate_gate']:.3f}",
            flush=True,
        )
    results["domain_gate_extrapolation"] = gate_rows

    Path(args.out).write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n结果已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
