"""模型评估指标与报表（训练期/影子运行共用）。

工业验收看的不是单一 RMSE，而是**分层误差**：

- 按运行/批次/日期分组，识别“某次开车特别差”；
- 训练域内 vs 域外（外推）分开报告；
- 质量（浓度）与温度分别报告，并给出有效监督点数；
- 偏差（bias）与 P90 绝对误差，比 RMSE 更能反映尾部风险；
- 参数漂移用物理单位报告（k_ref、UA），不是对数权重。

所有函数只做计算和格式化，不读取数据文件、不打印到日志系统。
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .config import COL_C_A, COL_RUN_ID


def regression_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> dict[str, float]:
    """计算一组预测的误差指标。

    Parameters
    ----------
    prediction, truth
        同形状数组，任意维度会被展平。
    mask
        可选布尔/0-1 掩码；只统计掩码为真的点。若掩码内没有点，返回
        ``n=0`` 和 NaN 指标，而不是 0——0 会被误认为“完美”。

    Returns
    -------
    dict
        ``n``、``rmse``、``mae``、``bias``（预测-真值）、``p90_abs``、
        ``max_abs``。
    """
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    truth = np.asarray(truth, dtype=np.float64).reshape(-1)
    if prediction.shape != truth.shape:
        raise ValueError("prediction 与 truth 形状必须一致。")
    if mask is None:
        valid = np.isfinite(prediction) & np.isfinite(truth)
    else:
        mask = np.asarray(mask).reshape(-1)
        if mask.shape != truth.shape:
            raise ValueError("mask 形状与数据不一致。")
        valid = (mask > 0) & np.isfinite(prediction) & np.isfinite(truth)
    n = int(valid.sum())
    if n == 0:
        return {
            "n": 0,
            "rmse": float("nan"),
            "mae": float("nan"),
            "bias": float("nan"),
            "p90_abs": float("nan"),
            "max_abs": float("nan"),
        }
    residual = prediction[valid] - truth[valid]
    absolute = np.abs(residual)
    return {
        "n": n,
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "mae": float(np.mean(absolute)),
        "bias": float(np.mean(residual)),
        "p90_abs": float(np.percentile(absolute, 90)),
        "max_abs": float(np.max(absolute)),
    }


def compare_models(
    predictions: dict[str, np.ndarray],
    truth: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    unit: str = "mol/L",
) -> pd.DataFrame:
    """把多个模型的指标拼成一张表，按 RMSE 升序。"""
    rows = []
    for name, prediction in predictions.items():
        metrics = regression_metrics(prediction, truth, mask=mask)
        rows.append({"模型": name, **metrics, "单位": unit})
    table = pd.DataFrame(rows)
    return table.sort_values("rmse", ignore_index=True)


def metrics_by_group(
    frame: pd.DataFrame,
    *,
    prediction_column: str,
    truth_column: str,
    mask_column: str | None = None,
    group_columns: Sequence[str] = (COL_RUN_ID,),
    unit: str = "mol/L",
) -> pd.DataFrame:
    """按分组列计算滚动指标（每条运行/每天/每个窗口行）。"""
    if prediction_column not in frame.columns or truth_column not in frame.columns:
        raise ValueError("报表缺少预测列或真值列。")
    rows = []
    for keys, group in frame.groupby(list(group_columns), dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        mask = (
            group[mask_column].to_numpy()
            if mask_column is not None and mask_column in group.columns
            else None
        )
        metrics = regression_metrics(
            group[prediction_column].to_numpy(),
            group[truth_column].to_numpy(),
            mask=mask,
        )
        row = dict(zip(group_columns, keys))
        rows.append({**row, **metrics, "单位": unit})
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return table.sort_values(list(group_columns), ignore_index=True)


def metrics_by_horizon(
    frame: pd.DataFrame,
    *,
    prediction_columns: dict[str, str],
    truth_column: str,
    time_column: str,
    run_column: str,
    bins_min: Sequence[float] = (0.0, 20.0, 40.0, 60.0, 120.0, 240.0),
    mask_column: str | None = None,
) -> pd.DataFrame:
    """按“距轨迹起点的预测时长”分箱报告误差。

    工业软测量里最容易被忽略的问题是**误差随时长增长**：一个在 20 min
    开环窗口上更好的模型，可能在 100 min 流式滚动时更差，因为小的速率偏差
    会持续累积。只报一个总 RMSE 会把这两种情况混在一起，导致选型结论相反。

    本函数把每个采样点的“时长”定义为 ``time - 该运行首次采样时间``，
    这与影子运行中状态锚定方式一致（状态只在起点/重锚点重置）。分箱后用
    ``regression_metrics`` 分别报告，并附样本数。
    """
    if truth_column not in frame.columns:
        raise ValueError(f"缺少真值列 {truth_column}。")
    required = {time_column, run_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"缺少列：{sorted(missing)}")
    work = frame.copy()
    first_time = work.groupby(run_column)[time_column].transform("min")
    work["_horizon_min"] = work[time_column] - first_time

    edges = np.asarray(sorted(set(bins_min)), dtype=float)
    if edges.size < 2:
        raise ValueError("bins_min 至少需要两个边界。")
    work["_bin"] = pd.cut(
        work["_horizon_min"],
        bins=edges,
        right=False,
        include_lowest=True,
    )

    rows = []
    for bin_label, group in work.groupby("_bin", observed=True, sort=True):
        mask = (
            group[mask_column].to_numpy()
            if mask_column is not None and mask_column in group.columns
            else None
        )
        for model_name, prediction_column in prediction_columns.items():
            metrics = regression_metrics(
                group[prediction_column].to_numpy(),
                group[truth_column].to_numpy(),
                mask=mask,
            )
            rows.append(
                {
                    "模型": model_name,
                    "预测时长区间 (min)": str(bin_label),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def summarise_parameters(parameters: dict[str, float]) -> pd.DataFrame:
    """把物理参数整理成报表行，附带单位与“名义值”提示。

    单位必须显式写出：``k_ref`` 是 min⁻¹，``ua`` 是 min⁻¹，
    ``heat_release`` 是 K·L/mol。缺单位的参数在跨装置对比时会导致
    数量级错误。
    """
    units = {
        "k_ref_per_min": ("min^-1", "参考温度下的速率常数"),
        "ua_per_min": ("min^-1", "UA/(rho Cp V)"),
        "heat_release_k_l_mol": ("K L/mol", "(-Delta H_r)/(rho Cp)"),
        "log_k_ref": ("-", "未反变换"),
        "log_ua": ("-", "未反变换"),
    }
    rows = []
    for name, value in parameters.items():
        unit, description = units.get(name, ("?", "未登记单位"))
        rows.append({"参数": name, "数值": float(value), "单位": unit, "含义": description})
    return pd.DataFrame(rows)


def domain_flags(
    frame: pd.DataFrame,
    *,
    bounds: dict[str, tuple[float, float]],
    margin: float = 0.05,
) -> pd.Series:
    """标记每条记录是否落在训练工况包络内（带相对裕度）。

    ``margin`` 是相对区间宽度的外扩比例；0 表示严格用训练 min/max。
    外推判断应使用**训练集**的物理范围（温度、进料浓度、停留时间），
    不用测试集统计量，否则会把“外推”误标成“内插”。

    Raises
    ------
    ValueError
        当 ``bounds`` 中的列在 ``frame`` 里不存在时**直接报错**。曾经这里的
        行为是“跳过缺失列”，结果是一个**所有列都缺失**的 frame 会得到
        “全部在域内”的结论——这会把外推风险静默掩盖掉。工业报表宁可
        报错，也不能给出没有依据的“安全”标记。
    """
    missing = [column for column in bounds if column not in frame.columns]
    if missing:
        raise ValueError(
            "计算域内标记所需的列不在数据中："
            f"{sorted(missing)}。请把控制量列一并传入（影子日志已在 "
            "LOG_COLUMNS 中写出），否则无法判断外推。"
        )
    flags = np.ones(len(frame), dtype=bool)
    for column, (low, high) in bounds.items():
        span = high - low
        lower = low - margin * span
        upper = high + margin * span
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy()
        # NaN 视为不可判定 -> 保守地标为域外。
        flags &= np.isfinite(values) & (values >= lower) & (values <= upper)
    return pd.Series(flags, index=frame.index, name="in_domain")


def control_bounds(frame: pd.DataFrame, columns: Iterable[str]) -> dict[str, tuple[float, float]]:
    """由训练数据计算各列的最小/最大值，作为外推包络。"""
    bounds: dict[str, tuple[float, float]] = {}
    for column in columns:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if values.empty:
            continue
        bounds[column] = (float(values.min()), float(values.max()))
    return bounds


def generalisation_report(
    *,
    in_domain: pd.DataFrame,
    out_of_domain: pd.DataFrame,
    prediction_column: str = "prediction_c_a",
    truth_column: str = COL_C_A,
    mask_column: str | None = None,
) -> pd.DataFrame:
    """域内与域外误差并列报告。

    这是工业验收的核心表：只报域内误差会系统性掩盖外推风险；只报域外
    误差又会因样本少而方差大。两者必须同时出现，并附样本数。
    """
    rows = []
    for label, frame in (("域内", in_domain), ("域外/外推", out_of_domain)):
        mask = (
            frame[mask_column].to_numpy()
            if mask_column is not None and mask_column in frame.columns
            else None
        )
        metrics = regression_metrics(
            frame[prediction_column].to_numpy(),
            frame[truth_column].to_numpy(),
            mask=mask,
        )
        rows.append({"工况": label, **metrics})
    return pd.DataFrame(rows)


def format_table(frame: pd.DataFrame, *, float_format: str = "{:.4f}") -> str:
    """用于日志的简洁文本表（不依赖 rich/tabulate）。"""
    if frame.empty:
        return "(空表)"
    columns = list(frame.columns)
    rendered = []
    for _, row in frame.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                cells.append("nan" if not np.isfinite(value) else float_format.format(value))
            else:
                cells.append(str(value))
        rendered.append(cells)
    widths = [
        max(len(str(column)), *(len(cell) for cell in column_cells))
        for column, column_cells in zip(columns, zip(*rendered))
    ]
    header = " | ".join(str(column).ljust(width) for column, width in zip(columns, widths))
    separator = "-+-".join("-" * width for width in widths)
    lines = [header, separator]
    lines += [
        " | ".join(cell.ljust(width) for cell, width in zip(cells, widths))
        for cells in rendered
    ]
    return "\n".join(lines)


__all__ = [
    "COL_C_A",
    "COL_T_REACTOR",
    "compare_models",
    "control_bounds",
    "domain_flags",
    "format_table",
    "generalisation_report",
    "metrics_by_group",
    "metrics_by_horizon",
    "regression_metrics",
    "summarise_parameters",
]
