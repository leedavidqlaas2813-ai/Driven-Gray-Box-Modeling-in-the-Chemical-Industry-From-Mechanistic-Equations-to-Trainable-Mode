"""慢变状态（结垢、催化剂老化）的特征与水力学累计量计算。

为什么“时间”本身不是好特征
--------------------------
直接喂 ``t`` 给网络会学到“运行越久速率越低”，一旦装置清洗、换催化剂或
改变产量，这个单调关系立即失效。工业上更稳的做法是把**可解释的运行强度**
作为慢变量：

- ``run_age_days``：自上次清洗/换剂以来的天数；
- ``cumulative_feed_m3``：累计进料体积，近似催化剂/换热面的累计接触量；
- ``ema_temperature_K``：温度的低通均值，反映热负荷历史。

即便如此，仍建议把清洗、换剂、切换牌号等**事件标记**作为显式特征
（本教程未提供事件数据，留给现场扩展）。

本模块只做确定性的、可审计的变换，不引入任何训练得到的参数。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    COL_CUM_FEED,
    COL_EMA_T,
    COL_RUN_AGE,
    COL_RUN_ID,
    COL_T_REACTOR,
    COL_TAU,
    COL_TIME_MIN,
    SLOW_FEATURE_COLUMNS,
)


def cumulative_feed_m3(
    frame: pd.DataFrame,
    *,
    reactor_volume_l: float,
    tau_column: str = COL_TAU,
    time_column: str = COL_TIME_MIN,
    run_column: str = COL_RUN_ID,
) -> pd.Series:
    """按运行累计进料体积 [m³]。

    ``q = V / tau``（体积流量回推），再用梯形积分：
    ``ΔV ≈ 0.5 (q_i + q_{i-1}) Δt``。与“简单求和 q*dt”相比，梯形法在
    停留时间变化时误差更小（二阶 vs 一阶）。

    这是**代理量**：现场若有流量计，应采用实测 ``cumulative_feed_m3``
    并在此处替换，保留列名不变。
    """
    if reactor_volume_l <= 0:
        raise ValueError("reactor_volume_l 必须为正。")
    tau = pd.to_numeric(frame[tau_column], errors="coerce")
    if (tau <= 0).any():
        raise ValueError("停留时间必须为正，无法换算进料流量。")
    flow_l_per_min = reactor_volume_l / tau
    # 流量单位 L/min -> m³/min。
    flow_m3_per_min = flow_l_per_min / 1000.0

    out = pd.Series(np.nan, index=frame.index, dtype=float)
    for _, group in frame.groupby(run_column, sort=False):
        order = group.sort_values(time_column)
        times = order[time_column].to_numpy(dtype=np.float64)
        flow = flow_m3_per_min.loc[order.index].to_numpy(dtype=np.float64)
        if len(times) < 2:
            volume = np.zeros(len(times))
        else:
            dt = np.diff(times)
            increments = 0.5 * (flow[1:] + flow[:-1]) * dt
            volume = np.concatenate([[0.0], np.cumsum(increments)])
        out.loc[order.index] = volume
    return out


def ema_temperature(
    frame: pd.DataFrame,
    *,
    dt_min: float,
    time_constant_min: float = 60.0,
    temperature_column: str = COL_T_REACTOR,
    time_column: str = COL_TIME_MIN,
    run_column: str = COL_RUN_ID,
) -> pd.Series:
    """温度的一阶低通均值（EMA），表示热负荷历史。

    用解析衰减因子 ``alpha = 1 - exp(-dt / tau_ema)`` 而不是固定 ``alpha``，
    这样采样间隔变化时滤波时间常数保持物理一致。
    """
    if time_constant_min <= 0:
        raise ValueError("time_constant_min 必须为正。")
    if dt_min <= 0:
        raise ValueError("dt_min 必须为正。")
    alpha = 1.0 - np.exp(-dt_min / time_constant_min)
    out = pd.Series(np.nan, index=frame.index, dtype=float)
    for _, group in frame.groupby(run_column, sort=False):
        order = group.sort_values(time_column)
        values = pd.to_numeric(order[temperature_column], errors="coerce")
        smoothed = values.ewm(alpha=alpha, adjust=False, ignore_na=True).mean()
        out.loc[order.index] = smoothed
    return out


def run_age_days(
    frame: pd.DataFrame,
    *,
    dt_min: float,
    campaign_start_min: float | None = None,
    time_column: str = COL_TIME_MIN,
    run_column: str = COL_RUN_ID,
) -> pd.Series:
    """自 campaign 起点以来的运行天数。

    ``campaign_start_min`` 应是上一次清洗/换剂/开车的时刻；``None`` 时取
    数据中最早的时间戳。**这不是**数据集的相对时间，而是装置的运行年龄；
    上线时必须由维护记录提供，否则慢变修正会把“数据窗口长度”误当成
    “装置老化”。
    """
    times = frame[time_column].astype(float)
    start = float(times.min()) if campaign_start_min is None else float(campaign_start_min)
    return (times - start) / (24.0 * 60.0)


def build_slow_features(
    frame: pd.DataFrame,
    *,
    dt_min: float,
    reactor_volume_l: float,
    campaign_start_min: float | None = None,
    time_constant_min: float = 60.0,
) -> pd.DataFrame:
    """生成 ``SLOW_FEATURE_COLUMNS`` 三列，不修改输入。"""
    features = frame[[COL_RUN_ID, COL_TIME_MIN]].copy()
    features[COL_RUN_AGE] = run_age_days(
        frame, dt_min=dt_min, campaign_start_min=campaign_start_min
    )
    features[COL_CUM_FEED] = cumulative_feed_m3(
        frame, reactor_volume_l=reactor_volume_l
    )
    features[COL_EMA_T] = ema_temperature(
        frame, dt_min=dt_min, time_constant_min=time_constant_min
    )
    return features[list(SLOW_FEATURE_COLUMNS)]


def campaign_time_min(frame: pd.DataFrame, *, dt_min: float) -> pd.Series:
    """整段 campaign 的绝对时间 [min]，用于按“天/窗口”分组报表。

    与 ``run_age_days`` 的区别：后者用于模型特征（装置年龄），前者只用于
    报表分组，不能作为模型输入，否则会把绝对日历时间泄漏给模型。
    """
    del dt_min  # 保留签名一致性；当前实现只依赖时间戳本身。
    return frame[COL_TIME_MIN].astype(float) - float(frame[COL_TIME_MIN].min())


def normalize_slow_features(
    values: np.ndarray, *, age_ref_days: float, feed_ref_m3: float
) -> np.ndarray:
    """把慢特征缩放到 O(1)，供小网络使用。

    采用**有界**缩放 ``x / (x + ref)`` 而不是 z-score：慢特征随装置寿命
    单调增长，z-score 的均值/方差会随数据窗口漂移；有界缩放对超出训练
    范围的时间外推更稳健（饱和而非爆炸）。
    """
    if age_ref_days <= 0 or feed_ref_m3 <= 0:
        raise ValueError("参考尺度必须为正。")
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("慢特征数组形状必须是 (N, 3)。")
    age = values[:, 0] / (np.maximum(values[:, 0], 0.0) + age_ref_days)
    feed = values[:, 1] / (np.maximum(values[:, 1], 0.0) + feed_ref_m3)
    # EMA 温度以 350 K 为中心，20 K 为尺度，再压缩到 [-1, 1]。
    ema = np.clip((values[:, 2] - 350.0) / 20.0, -3.0, 3.0) / 3.0
    return np.column_stack([age, feed, ema]).astype(np.float32)


__all__ = [
    "COL_CAMPAIGN_TIME",
    "build_slow_features",
    "campaign_time_min",
    "cumulative_feed_m3",
    "ema_temperature",
    "normalize_slow_features",
    "run_age_days",
]
