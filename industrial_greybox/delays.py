"""工业纯滞后（Dead Time）与时间对齐工具。

两类延迟必须分开建模，混在一起会制造“虚假残差”：

1. **传输/分析滞后**（``DelayConfig``）：化验结果在 ``result_time`` 到达，
   但它代表 ``process_time = result_time - dead_time`` 时刻的反应器出口。
   对齐时**把化验目标拉回过程时刻**，而不是把当前预测往前推。
2. **执行机构/控制回路滞后**：阀门、夹套和进料回路对设定值的响应延迟。
   这种延迟影响过程本身，应从**操作变量**侧对齐（或作为额外状态建模）。

本模块同时服务两种使用方式：

- 离线训练：``align_lab_results`` 把化验结果映射到 DCS 时间网格，并返回
  监督掩码（只有真正的化验点参与损失，不把零阶保持的阶梯当靶值）。
- 在线影子运行：``ShadowBuffer`` 保存带时间戳的历史预测/状态，化验到达时
  按**过程时刻**查询当时的历史预测，计算真正的模型残差。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd

from .config import COL_C_A, COL_RUN_ID, COL_TIME_MIN

InterpolationMode = Literal["zoh", "linear"]
TimestampSemantics = Literal["process", "result"]


# ---------------------------------------------------------------------------
# 在线：时间戳查询缓冲
# ---------------------------------------------------------------------------
@dataclass
class ShadowBuffer:
    """固定时长的流式环形缓冲区（按时间戳查询）。

    适用场景：影子运行中保存过去 N 分钟的预测状态、操作变量或温度，
    以便**滞后到达**的化验结果按历史时间戳回查“当时”的预测值。

    为什么必须按时间戳查询而不是取最新值
    ------------------------------------
    化验结果在 ``result_time`` 到达，但它代表 ``process_time =
    result_time - dead_time`` 时刻的反应器状态。若用当前预测去减迟到的
    化验值，残差里会混入死时间内的真实过程变化，表现为规律的滞后偏差；
    模型会被训练去补偿这个假偏差，上线后死时间稍有变化就失效。

    Notes
    -----
    ``horizon_min`` 必须大于最大死时间加一个化验间隔，否则结果到达时所需
    的历史点已经被丢弃；``query`` 会明确返回 ``None`` 表示历史不可用，
    调用方必须把它记录为 ``stale_history``，不能悄悄用当前值顶替。
    """

    horizon_min: float
    interpolation: InterpolationMode = "zoh"
    _times: list[float] = field(default_factory=list, repr=False)
    _values: list[np.ndarray] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if self.horizon_min <= 0:
            raise ValueError("horizon_min 必须为正。")
        if self.interpolation not in ("zoh", "linear"):
            raise ValueError("interpolation 必须是 'zoh' 或 'linear'。")

    def push(self, time_min: float, values: Any) -> None:
        """推入一个带时间戳的观测；时间戳必须非递减。"""
        time_min = float(time_min)
        if self._times and time_min < self._times[-1]:
            raise ValueError(
                f"ShadowBuffer 只接受非递减时间戳：{time_min} < {self._times[-1]}。"
            )
        array = np.atleast_1d(np.asarray(values, dtype=np.float64))
        if self._values and array.shape != self._values[-1].shape:
            raise ValueError(
                f"值形状变化：{array.shape} != {self._values[-1].shape}。"
            )
        self._times.append(time_min)
        self._values.append(array)
        self._prune(time_min - self.horizon_min)

    def _prune(self, cutoff: float) -> None:
        # 保留 cutoff 之前最近的一点，用于零阶保持/线性插值。
        keep_from = 0
        while (
            keep_from + 1 < len(self._times)
            and self._times[keep_from + 1] <= cutoff
        ):
            keep_from += 1
        if keep_from:
            del self._times[:keep_from]
            del self._values[:keep_from]

    def __len__(self) -> int:
        return len(self._times)

    @property
    def span_min(self) -> tuple[float, float] | None:
        if not self._times:
            return None
        return self._times[0], self._times[-1]

    def query(self, time_min: float) -> np.ndarray | None:
        """查询 ``time_min`` 时刻的值；历史不足时返回 ``None``。"""
        if not self._times:
            return None
        time_min = float(time_min)
        if time_min < self._times[0]:
            return None
        times = np.asarray(self._times)
        index = int(np.searchsorted(times, time_min, side="right") - 1)
        index = min(max(index, 0), len(times) - 1)
        if self.interpolation == "zoh" or index == len(times) - 1:
            return self._values[index].copy()
        t0, t1 = times[index], times[index + 1]
        if t1 == t0:
            return self._values[index].copy()
        weight = (time_min - t0) / (t1 - t0)
        return (1.0 - weight) * self._values[index] + weight * self._values[index + 1]


# ---------------------------------------------------------------------------
# 离线：化验结果对齐到 DCS 网格
# ---------------------------------------------------------------------------
def align_lab_results(
    dcs: pd.DataFrame,
    lab: pd.DataFrame,
    *,
    dead_time_min: float,
    time_column: str = COL_TIME_MIN,
    value_column: str = COL_C_A,
    run_column: str = COL_RUN_ID,
    timestamp_semantics: TimestampSemantics = "process",
    alignment_tolerance_min: float | None = None,
) -> pd.DataFrame:
    """把化验结果对齐到 DCS 时间网格，并返回监督掩码。

    Parameters
    ----------
    dcs
        按 ``run_id``、``time_min`` 排序的 DCS 长表。
    lab
        化验结果长表，至少包含 ``run_column``、``time_column``、``value``。
    dead_time_min
        总死时间 [min]（传输 + 分析周转）。
    timestamp_semantics
        - ``'process'``：时间戳已经是取样/过程时刻，直接对齐；
        - ``'result'``：时间戳是结果到达时刻，过程时刻 = 时间戳 - 死时间。
    alignment_tolerance_min
        把化验过程时刻吸附到最近 DCS 行的最大距离。``None`` 时自动取
        ``min(网格步长, 化验间隔)`` 的一半并**不小于**半个网格步长：
        化验时刻往往与网格不对齐，若容差只取网格步长的一半，误差恰好
        落在两个网格点中间时会全部丢弃，导致该运行没有监督点。

        **不要把这个容差当成“时间戳精度”问题来收紧。** 吸附只是把化验
        值挂到最近的过程时刻，本身不改变预测时刻；真正的偏差来自死时间
        估计错误，应由 ``dead_time_min`` 处理。

    Returns
    -------
    DataFrame
        在 ``dcs`` 基础上增加三列：

        - ``c_a_aligned``：零阶保持后的最近化验值（用于画图/审查，
          不作为中间时刻的训练真值）；
        - ``c_a_mask``：1.0 表示该行被吸附到一条真实化验结果，参与损失；
        - ``lab_process_time_min``：该行对应的化验过程时刻（无监督时为 NaN）。

    Notes
    -----
    本函数是**离线**对齐，允许使用稍后到达的化验值；在线影子运行必须用
    ``ShadowBuffer`` 只查看已经到达的结果，不能用本函数。
    """
    if timestamp_semantics not in ("process", "result"):
        raise ValueError("timestamp_semantics 必须是 'process' 或 'result'。")
    if dead_time_min < 0:
        raise ValueError("dead_time_min 不能为负。")
    if run_column not in dcs.columns:
        raise ValueError(f"DCS 表缺少运行标识列 {run_column}。")

    required_lab = {run_column, time_column, value_column}
    missing = required_lab - set(lab.columns)
    if missing:
        raise ValueError(f"化验表缺少列：{sorted(missing)}")
    if not {time_column}.issubset(dcs.columns):
        raise ValueError(f"DCS 表缺少列 {time_column}。")

    pieces: list[pd.DataFrame] = []
    for run_id, run_frame in dcs.groupby(run_column, sort=False):
        run_frame = run_frame.sort_values(time_column).copy()
        run_lab = lab.loc[lab[run_column] == run_id].copy()
        grid_times = run_frame[time_column].to_numpy(dtype=np.float64)
        if run_lab.empty or len(grid_times) == 0:
            run_frame["c_a_aligned"] = np.nan
            run_frame["c_a_mask"] = 0.0
            run_frame["lab_process_time_min"] = np.nan
            pieces.append(run_frame)
            continue
        run_lab[time_column] = pd.to_numeric(run_lab[time_column], errors="coerce")
        run_lab = run_lab.dropna(subset=[time_column, value_column])
        run_lab = run_lab.sort_values(time_column)
        shift = dead_time_min if timestamp_semantics == "result" else 0.0
        lab_process = run_lab[time_column].to_numpy(dtype=np.float64) - shift
        lab_values = run_lab[value_column].to_numpy(dtype=np.float64)
        pieces.append(
            _attach_lab_values(
                run_frame,
                grid_times,
                lab_process,
                lab_values,
                alignment_tolerance_min,
            )
        )

    aligned = pd.concat(pieces, ignore_index=True)
    return aligned.sort_values([run_column, time_column]).reset_index(drop=True)


def _attach_lab_values(
    grid_frame: pd.DataFrame,
    grid_times: np.ndarray,
    lab_process: np.ndarray,
    lab_values: np.ndarray,
    alignment_tolerance_min: float | None,
) -> pd.DataFrame:
    """在单条运行内把化验值零阶保持到 DCS 网格，并生成吸附掩码。"""
    result = grid_frame.copy()
    aligned = np.full(len(grid_times), np.nan)
    mask = np.zeros(len(grid_times))
    lab_time = np.full(len(grid_times), np.nan)

    if len(lab_process) == 0:
        result["c_a_aligned"] = aligned
        result["c_a_mask"] = mask
        result["lab_process_time_min"] = lab_time
        return result

    # 零阶保持：每个网格点取过程时刻不晚于它的最后一个化验值。
    zoh_index = np.searchsorted(lab_process, grid_times, side="right") - 1
    usable = zoh_index >= 0
    aligned[usable] = lab_values[zoh_index[usable]]

    if alignment_tolerance_min is None:
        grid_step = (
            float(np.median(np.diff(grid_times))) if len(grid_times) > 1 else np.inf
        )
        lab_step = (
            float(np.median(np.diff(lab_process)))
            if len(lab_process) > 1
            else np.inf
        )
        half_min = min(grid_step, lab_step) / 2.0
        # 至少允许半个网格步长，否则“落在两格正中间”的化验会被全部丢弃。
        tolerance = max(half_min, grid_step / 2.0) if np.isfinite(half_min) else 0.5 * grid_step
        # 但不应超过一个完整网格步长：超过就意味着可能吸附到错误的采样点。
        if np.isfinite(grid_step):
            tolerance = min(tolerance, grid_step)
    else:
        if alignment_tolerance_min < 0:
            raise ValueError("alignment_tolerance_min 不能为负。")
        tolerance = float(alignment_tolerance_min)

    # 每条化验结果吸附到最近的网格行；只有吸附成功的那一行参与监督。
    for process_time, value in zip(lab_process, lab_values):
        if not np.isfinite(process_time):
            continue
        position = int(np.searchsorted(grid_times, process_time))
        candidates = [
            index
            for index in (position - 1, position)
            if 0 <= index < len(grid_times)
        ]
        if not candidates:
            continue
        nearest = min(candidates, key=lambda index: abs(grid_times[index] - process_time))
        if abs(grid_times[nearest] - process_time) <= tolerance + 1e-9:
            mask[nearest] = 1.0
            aligned[nearest] = value
            lab_time[nearest] = process_time

    result["c_a_aligned"] = aligned
    result["c_a_mask"] = mask
    result["lab_process_time_min"] = lab_time
    return result


def shift_controls_for_actuator_delay(
    controls: pd.DataFrame,
    *,
    dt_min: float,
    columns: Sequence[str],
    delay_min: float,
) -> pd.DataFrame:
    """按执行机构死时间平移操作变量（零阶保持重采样）。

    物理含义：DCS 记录的是“指令/阀位”，而进入反应器的实际进料在
    ``delay_min`` 之后才响应。若用指令时间序列直接当模型输入，会把过程
    响应提前，训练会学到虚假的更快动力学。

    约定：``delay_min`` 必须是非负的纯滞后，且是 ``dt_min`` 的整数倍。
    一阶惯性（滞后 + 过渡）不是纯滞后，应作为额外状态或一阶滤波建模。
    """
    if delay_min < 0:
        raise ValueError("delay_min 不能为负；负延迟意味着使用了未来数据。")
    if dt_min <= 0:
        raise ValueError("dt_min 必须为正。")
    missing = set(columns) - set(controls.columns)
    if missing:
        raise ValueError(f"缺少待平移列：{sorted(missing)}")
    shifted = controls.copy()
    if delay_min == 0:
        return shifted
    steps_float = delay_min / dt_min
    if abs(steps_float - round(steps_float)) > 1e-6:
        raise ValueError(
            f"执行机构延迟 {delay_min} min 不是采样步长 {dt_min} min 的整数倍；"
            "应先重采样到统一网格，避免插值引入额外误差。"
        )
    steps = int(round(steps_float))
    for column in columns:
        shifted[column] = shifted[column].shift(steps).bfill()
    return shifted


#: 向后兼容别名。早期版本使用 ``DelayBuffer``；新代码请用 ``ShadowBuffer``，
#: 它更准确地表达“影子运行的历史预测缓存”这一用途。
DelayBuffer = ShadowBuffer

__all__ = [
    "DelayBuffer",
    "ShadowBuffer",
    "TimestampSemantics",
    "align_lab_results",
    "shift_controls_for_actuator_delay",
]
