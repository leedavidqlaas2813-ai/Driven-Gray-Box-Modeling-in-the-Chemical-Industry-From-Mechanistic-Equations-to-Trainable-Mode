"""消融实验：验证长期外推正则化的实际效果。

用途
----
回答一个具体问题：新增的滚动一致性惩罚与域外门控，是否真的改善了
**影子运行**（长周期滚动）的指标？只报开环指标不足以说明问题——本仓库
早先的记录显示灰箱可以在 20 min 开环上更好、在 100 min 影子上更差。

运行::

    python tools/ablation_regularization.py --epochs 10 --stride 6

结果逐项写入 ``ablation_results.csv``，避免长任务输出被截断后丢失。

边界
----
全部数据来自 ``synthetic_plant``；结论只适用于本合成工况，不构成任何
真实装置验证。
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# 允许从仓库根目录**或** tools/ 目录直接运行：
# `python tools/ablation_regularization.py` 时 sys.path[0] 是 tools/，
# 不包含仓库根，无法 import industrial_greybox。
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("A 无正则(旧行为)", "B 仅滚动一致性", "C 仅域外门控", "D 两者全开"), default="A 无正则(旧行为)")
    parser.add_argument("--reseed", choices=("adaptive", "fixed", "both"), default="adaptive")
    parser.add_argument("--runs", type=int, default=12)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--days", type=float, default=30.0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", default="ablation_results.csv")
    args = parser.parse_args()

    from industrial_greybox.config import (
        DelayConfig,
        LongHorizonConfig,
        ModelConfig,
        ShadowConfig,
    )
    from industrial_greybox.pipeline import run_pipeline
    from industrial_greybox.synthetic_plant import CampaignSpec, simulate_campaign

    data = simulate_campaign(
        CampaignSpec(
            n_runs=args.runs,
            n_steps=args.steps,
            days_span=args.days,
            seed=args.seed,
        )
    )
    base = ModelConfig()
    variants = {
        "A 无正则(旧行为)": replace(
            base,
            long_horizon=LongHorizonConfig(rollout_penalty=False, domain_gate=False),
        ),
        "B 仅滚动一致性": replace(
            base,
            long_horizon=LongHorizonConfig(rollout_penalty=True, domain_gate=False),
        ),
        "C 仅域外门控": replace(
            base,
            long_horizon=LongHorizonConfig(rollout_penalty=False, domain_gate=True),
        ),
        "D 两者全开": base,
    }
    shadow_adaptive = ShadowConfig(reseed_mode="adaptive")
    shadow_fixed = ShadowConfig(reseed_mode="fixed")

    rows: list[dict[str, object]] = []
    output = Path(args.out)
    for name, config in variants.items():
        for reseed_label, shadow in (
            ("adaptive", shadow_adaptive),
            ("fixed", shadow_fixed),
        ):
            result = run_pipeline(
                *data,
                config=config,
                delay=DelayConfig(),
                epochs=args.epochs,
                stride=args.stride,
                verbose=0,
                run_baseline=True,
                run_shadow=True,
                shadow=shadow,
            )
            open_loop = result.metrics.set_index("模型")
            shadow_table = result.shadow_report.generalisation.set_index("模型")
            row = {
                "变体": name,
                "重锚模式": reseed_label,
                "开环_灰箱": float(open_loop.loc["灰箱", "C_A RMSE (mol/L)"]),
                "开环_机理": float(open_loop.loc["机理基线", "C_A RMSE (mol/L)"]),
                "影子_灰箱": float(shadow_table.loc["灰箱", "rmse"]),
                "影子_机理": float(shadow_table.loc["机理基线", "rmse"]),
                "非有限步": result.shadow_report.n_nonfinite,
                "裁剪步": result.shadow_report.n_clipped,
            }
            row["影子_灰箱优于机理"] = row["影子_灰箱"] < row["影子_机理"]
            rows.append(row)
            # 每完成一项就落盘，长任务中断也不丢结果。
            pd.DataFrame(rows).to_csv(output, index=False, encoding="utf-8-sig")
            print(
                f"{name} [{reseed_label}]: "
                f"开环 灰箱={row['开环_灰箱']:.4f} 机理={row['开环_机理']:.4f} | "
                f"影子 灰箱={row['影子_灰箱']:.4f} 机理={row['影子_机理']:.4f}",
                flush=True,
            )

    print(f"\n结果已写入 {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
