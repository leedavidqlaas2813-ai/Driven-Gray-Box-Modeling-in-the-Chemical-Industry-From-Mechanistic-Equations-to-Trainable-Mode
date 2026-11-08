"""命令行入口：跑通端到端流程并打印/落盘报表。

用法（仓库根目录）::

    python -m industrial_greybox.cli --quick
    python -m industrial_greybox.cli --epochs 30 --runs 16 --out shadow_runs
    python -m industrial_greybox.cli --dcs plant_runs.csv --lab plant_lab.csv \
        --line-delay 2 --analysis-delay 10 --out shadow_runs

现场数据模式要求 CSV 已按统一列名导出（见 ``config.EXAMPLE_COLUMN_MAPPING``），
且 ``lab`` 的 ``time_min`` 是**结果到达**时刻。脚本不会替你猜测单位或
死时间；那些必须来自数据字典。

本脚本只做评估与诊断，不会向任何控制系统写值。影子运行是**只读**的。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def _load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _simulate(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    from .synthetic_plant import CampaignSpec, simulate_campaign

    spec = CampaignSpec(
        n_runs=args.runs,
        n_steps=args.steps,
        days_span=args.days,
        lab_interval_min=args.lab_interval,
        inject_faults=not args.no_faults,
    )
    return simulate_campaign(spec)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="工业灰箱 CSTR 软测量：训练、评估与影子运行（只读）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--quick", action="store_true", help="小规模冒烟运行")
    source.add_argument("--dcs", help="现场 DCS 长表 CSV")
    parser.add_argument("--lab", help="化验结果 CSV（与 --dcs 配合）")
    parser.add_argument("--runs", type=int, default=12, help="合成 campaign 的运行数")
    parser.add_argument("--steps", type=int, default=200, help="每条运行的采样点数")
    parser.add_argument("--days", type=float, default=30.0, help="campaign 跨越天数")
    parser.add_argument(
        "--lab-interval", type=float, default=10.0, help="化验间隔 [min]"
    )
    parser.add_argument(
        "--line-delay", type=float, default=2.0, help="取样管线传输滞后 [min]"
    )
    parser.add_argument(
        "--analysis-delay", type=float, default=10.0, help="分析/化验周转 [min]"
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=4, help="窗口起点步长")
    parser.add_argument(
        "--integrator",
        choices=("rk4", "imex", "euler"),
        default="rk4",
        help="可微积分格式",
    )
    parser.add_argument("--no-faults", action="store_true", help="合成数据不注入仪表故障")
    parser.add_argument("--no-baseline", action="store_true", help="跳过机理基线")
    parser.add_argument("--no-shadow", action="store_true", help="跳过影子运行")
    parser.add_argument("--out", default="shadow_runs", help="输出目录")
    parser.add_argument("--json", action="store_true", help="额外写出 JSON 摘要")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    import os

    os.environ.setdefault("KERAS_BACKEND", "tensorflow")

    from dataclasses import replace

    from .config import DelayConfig, ModelConfig, ShadowConfig
    from .pipeline import format_pipeline_summary, run_pipeline

    config = ModelConfig()
    config = replace(
        config,
        integrator=replace(config.integrator, method=args.integrator),
    )

    if args.dcs:
        if not args.lab:
            print("错误：--dcs 必须与 --lab 一起使用。", file=sys.stderr)
            return 2
        dcs = _load_csv(args.dcs)
        lab = _load_csv(args.lab)
        # 现场模式：化验时间戳必须是“结果到达”时刻；若 LIMS 导出的是
        # 取样时刻，应由数据字典确认后改这里的语义，而不是猜。
        timestamp_semantics = "result"
    else:
        if args.quick:
            dcs, lab = _simulate(
                argparse.Namespace(
                    runs=8,
                    steps=160,
                    days=30.0,
                    lab_interval=10.0,
                    no_faults=False,
                )
            )
            args.epochs = min(args.epochs, 8)
            args.stride = max(args.stride, 8)
        else:
            dcs, lab = _simulate(args)
        timestamp_semantics = "result"

    delay = DelayConfig(
        line_delay_min=args.line_delay,
        analysis_delay_min=args.analysis_delay,
        lab_interval_min=args.lab_interval,
    )
    shadow = ShadowConfig(output_dir=Path(args.out))

    result = run_pipeline(
        dcs,
        lab,
        config=config,
        delay=delay,
        shadow=shadow,
        epochs=args.epochs,
        batch_size=args.batch_size,
        stride=args.stride,
        run_baseline=not args.no_baseline,
        run_shadow=not args.no_shadow,
        verbose=1,
        seed=7,
    )
    print()
    print(format_pipeline_summary(result))

    # 落盘：日志、报表与 JSON 摘要。
    if not args.no_shadow and result.shadow_report is not None:
        from .shadow import ShadowRunner, build_training_bounds
        from .config import COL_RUN_ID

        test_runs = result.groups["test"]
        frame = result.cleaned[result.cleaned[COL_RUN_ID].astype(str).isin(test_runs)]
        runner = ShadowRunner(
            result.training.model,
            config,
            delay,
            shadow,
            slow_reference=(
                config.slow_drift.age_ref_days,
                config.slow_drift.feed_ref_m3,
            ),
            training_bounds=build_training_bounds(
                result.cleaned[result.cleaned[COL_RUN_ID].astype(str).isin(result.groups["train"])]
            ),
        )
        paths = runner.write_outputs(
            result.shadow_log, result.shadow_lab_events, result.shadow_report
        )
        for label, path in paths.items():
            print(f"写出 {label}: {path}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    result.metrics.to_csv(out_dir / "open_loop_metrics.csv", index=False, encoding="utf-8-sig")
    result.hygiene_report.to_frame().to_csv(
        out_dir / "hygiene_report.csv", index=False, encoding="utf-8-sig"
    )
    if args.json:
        payload = {
            "config_signature": config.signature(),
            "integrator": args.integrator,
            "dead_time_min": delay.total_dead_time_min,
            "parameters": result.training.model.fitted_parameters(),
            "metrics": json.loads(result.metrics.to_json(orient="records")),
            "groups": {name: len(ids) for name, ids in result.groups.items()},
        }
        (out_dir / "run_summary.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"写出摘要: {out_dir / 'run_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
