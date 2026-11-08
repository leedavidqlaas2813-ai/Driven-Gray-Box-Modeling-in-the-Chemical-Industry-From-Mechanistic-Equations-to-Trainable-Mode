"""端到端流程：合成/现场长表 → 清洗 → 对齐 → 窗口 → 训练 → 对照 → 影子运行。

这个模块是“准生产级”的最小可用装配，把各模块按**正确顺序**接起来，
并明确每一步的输入/输出契约。它不替你做工艺决策；所有阈值、单位、
时滞和验证策略都必须由工艺/仪表/安全人员审核。

顺序不能交换
------------
1. **清洗**（只删坏值，不伪造标签）；
2. **化验对齐**（把结果按死时间拉回过程时刻，生成监督掩码）；
3. **慢特征**（累计进料、运行年龄、温度 EMA，必须用清洗后的数据）；
4. **按运行切分 + 窗口化**（先切分后窗口，防止相邻点泄漏）；
5. **训练**（灰箱 vs 机理基线，同一损失/同一积分）；
6. **离线评估**（域内/域外、按运行）；
7. **影子运行**（**重新**以因果方式流式推理，不使用离线对齐结果）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    COL_RUN_ID,
    COL_TIME_MIN,
    DelayConfig,
    HygieneConfig,
    ModelConfig,
    ShadowConfig,
)
from .data import WindowedDataset, build_datasets, check_no_run_leakage
from .delays import align_lab_results
from .hygiene import IndustrialHygienePipeline, HygieneReport
from .reports import domain_flags, format_table
from .shadow import ShadowRunner, build_training_bounds
from .slow_features import build_slow_features
from .training import TrainingResult, open_loop_error_table, train_greybox


@dataclass
class PipelineResult:
    """一次端到端运行的产物。"""

    cleaned: pd.DataFrame
    aligned: pd.DataFrame
    hygiene_report: HygieneReport
    datasets: dict[str, WindowedDataset]
    groups: dict[str, list[str]]
    training: TrainingResult
    baseline: TrainingResult | None
    metrics: pd.DataFrame
    parameters: pd.DataFrame
    shadow_log: pd.DataFrame = field(default_factory=pd.DataFrame)
    shadow_lab_events: pd.DataFrame = field(default_factory=pd.DataFrame)
    shadow_report: Any = None
    split_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def prepare_frame(
    dcs: pd.DataFrame,
    lab: pd.DataFrame,
    *,
    config: ModelConfig,
    delay: DelayConfig,
    hygiene: HygieneConfig | None = None,
    timestamp_semantics: str = "result",
    campaign_start_min: float | None = None,
) -> tuple[pd.DataFrame, HygieneReport]:
    """清洗 + 化验对齐 + 慢特征，返回可直接窗口化的长表。

    关键约定：``lab[time_min]`` 默认是**结果到达**时刻，因此必须用
    ``timestamp_semantics='result'``；若 LIMS 导出的是取样时刻，改传
    ``'process'`` 并把 ``dead_time_min`` 设为 0。
    """
    pipeline = IndustrialHygienePipeline(hygiene)
    cleaned, report = pipeline.clean(dcs)
    aligned = align_lab_results(
        cleaned,
        lab,
        dead_time_min=delay.total_dead_time_min,
        timestamp_semantics=timestamp_semantics,  # type: ignore[arg-type]
    )
    slow = build_slow_features(
        aligned,
        dt_min=config.physics.dt_min,
        reactor_volume_l=config.physics.reactor_volume_l,
        campaign_start_min=campaign_start_min,
    )
    merged = aligned.reset_index(drop=True).copy()
    # 慢特征一律以**重新计算值**为准：现场 DCS 若自带类似列，其定义可能
    # 与 ``slow_features`` 不同（例如用日历天数而非运行年龄），直接保留
    # 会让训练与影子运行使用两套定义。
    for column in slow.columns:
        merged[column] = slow[column].to_numpy()
    return merged, report


def run_pipeline(
    dcs: pd.DataFrame,
    lab: pd.DataFrame,
    *,
    config: ModelConfig,
    delay: DelayConfig,
    shadow: ShadowConfig | None = None,
    hygiene: HygieneConfig | None = None,
    epochs: int = 40,
    batch_size: int = 32,
    learning_rate: float = 5e-3,
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 7,
    stride: int = 2,
    run_baseline: bool = True,
    run_shadow: bool = True,
    verbose: int = 1,
    campaign_start_min: float | None = None,
) -> PipelineResult:
    """完整流程（见模块文档）。返回的 ``notes`` 记录所有“不可比/不可用”的情况。"""
    notes: list[str] = []
    frame, report = prepare_frame(
        dcs,
        lab,
        config=config,
        delay=delay,
        hygiene=hygiene,
        timestamp_semantics="result",
        campaign_start_min=campaign_start_min,
    )
    slow_reference = (
        config.slow_drift.age_ref_days,
        config.slow_drift.feed_ref_m3,
    )
    datasets, groups = build_datasets(
        frame,
        physics=config.physics,
        slow_reference=slow_reference,
        fractions=fractions,
        seed=seed,
        stride=stride,
    )
    check_no_run_leakage(datasets, groups)

    training = train_greybox(
        datasets,
        config,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed + 1,
        verbose=verbose,
    )
    baseline = (
        train_greybox(
            datasets,
            config,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed + 1,
            verbose=max(verbose - 1, 0),
            enable_corrections=False,
        )
        if run_baseline
        else None
    )

    test = datasets["test"]
    predictions = {"灰箱": training.predictions["test"]}
    if baseline is not None:
        predictions["机理基线"] = baseline.predictions["test"]
    metrics = open_loop_error_table(
        predictions,
        test.targets,
        test.weights,
        temperature_scale_k=config.physics.temperature_scale_k,
    )
    parameters = training.parameters
    if baseline is not None and not baseline.parameters.empty:
        parameters = pd.concat(
            [
                parameters.assign(模型="灰箱"),
                baseline.parameters.assign(模型="机理基线"),
            ],
            ignore_index=True,
        )

    # 域内/域外：用训练运行的控制包络判断测试窗口是否外推。
    train_frame = frame[frame[COL_RUN_ID].astype(str).isin(groups["train"])]
    bounds = build_training_bounds(train_frame)
    test_frame = frame[frame[COL_RUN_ID].astype(str).isin(groups["test"])]
    test_flags = domain_flags(test_frame, bounds=bounds, margin=0.0)
    # 窗口级: 任一点外推即视为外推窗口。
    in_domain_windows = _window_domain_flags(test, test_frame, test_flags, config)

    shadow_log = pd.DataFrame()
    shadow_events = pd.DataFrame()
    shadow_report = None
    if run_shadow:
        shadow_frame = frame[
            frame[COL_RUN_ID].astype(str).isin(groups["test"])
        ].copy()
        runner = ShadowRunner(
            training.model,
            config,
            delay,
            shadow or ShadowConfig(),
            slow_reference=slow_reference,
            training_bounds=bounds,
        )
        shadow_log, shadow_events, shadow_report = runner.run(
            shadow_frame, lab[lab[COL_RUN_ID].astype(str).isin(groups["test"])]
        )
        if shadow_report is not None and shadow_report.n_stale_history:
            notes.append(
                f"{shadow_report.n_stale_history} 条化验到达时历史预测已被缓冲区丢弃，"
                "对应残差不可用；请增大缓冲时长或缩短死时间。"
            )

    notes.append(
        "窗口级域内/域外标记："
        f"域内 {int(in_domain_windows.sum())} / {len(in_domain_windows)} 个测试窗口。"
    )
    notes.append(
        "灰箱与机理基线使用相同切分、损失、积分格式；差异只来自网络修正。"
    )
    return PipelineResult(
        cleaned=frame,
        aligned=frame,
        hygiene_report=report,
        datasets=datasets,
        groups=groups,
        training=training,
        baseline=baseline,
        metrics=metrics,
        parameters=parameters,
        shadow_log=shadow_log,
        shadow_lab_events=shadow_events,
        shadow_report=shadow_report,
        notes=notes,
    )


def _window_domain_flags(
    dataset: WindowedDataset,
    test_frame: pd.DataFrame,
    point_flags: pd.Series,
    config: ModelConfig,
) -> np.ndarray:
    """把点级域内标记聚合成窗口级标记。

    检查窗口覆盖的时间段内是否存在任一外推点。窗口起点时间与运行 ID
    联合定位，避免同一时刻不同运行混淆。
    """
    flags = np.ones(len(dataset), dtype=bool)
    lookup = pd.DataFrame(
        {
            "run_id": test_frame[COL_RUN_ID].astype(str).to_numpy(),
            "time_min": test_frame[COL_TIME_MIN].to_numpy(),
            "in_domain": point_flags.to_numpy(),
        }
    ).set_index(["run_id", "time_min"])["in_domain"]
    n_steps = config.physics.n_steps
    dt = config.physics.dt_min
    for index in range(len(dataset)):
        run_id = str(dataset.run_ids[index])
        start = float(dataset.window_start_times[index])
        times = start + np.arange(n_steps + 1) * dt
        for time_min in times:
            key = (run_id, float(round(time_min, 6)))
            try:
                value = lookup.loc[key]
            except KeyError:
                continue
            if isinstance(value, pd.Series):
                value = value.iloc[0]
            if not bool(value):
                flags[index] = False
                break
    return flags


def evaluation_tables(result: PipelineResult, *, config: ModelConfig) -> dict[str, pd.DataFrame]:
    """汇总可直接贴进报告的表。"""
    return {
        "清洗报告": result.hygiene_report.to_frame(),
        "测试指标": result.metrics,
        "参数": result.parameters,
        "健康记录": result.training.health_records,
    }


def format_pipeline_summary(result: PipelineResult) -> str:
    """打印用摘要（中文标签，单位显式）。"""
    lines = [result.hygiene_report.summary_line(), ""]
    lines.append("[开环测试指标]")
    lines.append(format_table(result.metrics))
    if result.baseline is not None:
        lines.append("")
        lines.append("[学习到的机理参数]")
        lines.append(format_table(result.parameters))
    if result.shadow_report is not None:
        lines.append("")
        lines.append("[影子运行]")
        lines.append(result.shadow_report.to_text())
    if result.notes:
        lines.append("")
        lines.append("[说明]")
        lines.extend(f"- {note}" for note in result.notes)
    return "\n".join(lines)


__all__ = [
    "PipelineResult",
    "evaluation_tables",
    "format_pipeline_summary",
    "prepare_frame",
    "run_pipeline",
]
