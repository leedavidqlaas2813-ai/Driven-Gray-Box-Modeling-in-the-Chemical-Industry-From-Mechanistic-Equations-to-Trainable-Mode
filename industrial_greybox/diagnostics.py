"""参数可辨识性与模型健康诊断。

工业灰箱模型最容易出错的不是代码，而是**“拟合得好，但参数没有物理意义”**。
稳态数据上经典的不可辨识组合：

- ``UA`` 增大 ↔ 反应热 ``-ΔH_r`` 减小，两者给出相同的温度；
- ``k_ref`` 增大 ↔ ``Ea`` 减小，在窄温区内给出近似相同的速率；
- 网络速率修正 ↔ ``k_ref``：两者都在乘同一个速率，梯度可以互相抵消。

本模块提供三类诊断：

1. **灵敏度矩阵**：预测对参数的有限差分灵敏度，以及条件数；
2. **参数相关性**：由灵敏度矩阵的列相关得到“谁和谁在互相补偿”；
3. **残差结构**：残差对过程时刻/温度的线性趋势，用于发现时滞或
   结构误差被错误吸收进参数。

所有函数都用模型自身的积分器做有限差分，因此与训练时的物理实现一致。
它们**不替代**正式的辨识性分析（需要 Fisher 信息矩阵、剖面似然或
正交实验设计），只是让明显的病态在训练后立刻可见。
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from .data import WindowedDataset


def _predict(model: Any, dataset: WindowedDataset, indices: np.ndarray) -> np.ndarray:
    inputs = (
        dataset.initial_states[indices],
        dataset.controls[indices],
        dataset.slow[indices],
    )
    return np.asarray(model(inputs, training=False))


def sensitivity_matrix(
    model: Any,
    dataset: WindowedDataset,
    *,
    parameters: Sequence[str] = ("log_k_ref", "log_ua"),
    relative_step: float = 1e-3,
    max_windows: int = 32,
    channel: int = 0,
) -> pd.DataFrame:
    """预测对**对数参数**的有限差分灵敏度。

    Parameters
    ----------
    parameters
        ``CSTRGreyBoxLayer`` 上的权重名。使用对数参数（``log_k_ref``、
        ``log_ua``）而不是物理参数，是因为它们的相对变化才与“可辨识性”
        对应：``k_ref`` 的 +1% 与 ``log_k_ref`` 的 +0.01 等价。
    channel
        0 = 浓度，1 = 归一化温度。浓度与温度的灵敏度**必须分开看**：
        温度对 UA 敏感、浓度对 k 敏感，这是把两者分开辨识的基础。

    Returns
    -------
    DataFrame
        行 = 窗口×时间点，列 = 参数；可直接算条件数/相关矩阵。

    Notes
    -----
    使用**未加权**的开环轨迹：加权会让“没有化验的时刻”不产生灵敏度，
    掩盖参数在这些时段的影响。
    """
    if max_windows < 1:
        raise ValueError("max_windows 必须 >= 1。")
    n = min(len(dataset), max_windows)
    indices = np.linspace(0, len(dataset) - 1, n).astype(int)
    layer = model.physics_layer
    variables = {name: getattr(layer, name) for name in parameters}
    original = {name: np.array(np.asarray(var)) for name, var in variables.items()}

    base = _predict(model, dataset, indices)[:, :, channel]

    columns: dict[str, np.ndarray] = {}
    try:
        for name, var in variables.items():
            # 对数参数本身是无量纲的，直接加一个小的绝对扰动。
            step = relative_step
            var.assign(original[name] + step)
            perturbed = _predict(model, dataset, indices)[:, :, channel]
            columns[f"+{name}"] = (perturbed - base) / step
            var.assign(original[name] - step)
            perturbed = _predict(model, dataset, indices)[:, :, channel]
            columns[f"-{name}"] = (perturbed - base) / step
    finally:
        for name, var in variables.items():
            var.assign(original[name])

    # 中心差分。
    sensitivity = {
        name: 0.5 * (columns[f"+{name}"] + columns[f"-{name}"])
        for name in parameters
    }
    frame = pd.DataFrame({name: values.reshape(-1) for name, values in sensitivity.items()})
    frame["window"] = np.repeat(indices, sensitivity[parameters[0]].shape[1])
    frame["step"] = np.tile(np.arange(sensitivity[parameters[0]].shape[1]), n)
    return frame


def sensitivity_correlation_matrix(
    sensitivities: pd.DataFrame,
    *,
    parameters: Sequence[str] | None = None,
) -> pd.DataFrame:
    """参数灵敏度列的相关矩阵（“谁和谁在互相补偿”）。

    为什么单独输出矩阵而不只给标量
    ------------------------------
    两个参数的可辨识性问题本质上是一个**子空间**问题：参数 A 与 B 共线，
    不代表 A 与 C 也共线。只看 ``parameter_identifiability`` 里的两两标量
    容易漏掉“三个参数两两不共线、但合起来只有两个独立方向”的情形。

    矩阵按列做尺度归一化后再求 Pearson 相关：不同参数的单位与量级不同，
    不归一化时相关性会被量纲主导。

    Returns
    -------
    DataFrame
        索引与列均为参数名，对角线为 1。|r| > 0.95 表示两者几乎无法从
        当前数据中分别辨识，应补充能打破共线的工况（例如只改变冷却、
        不改变进料）。
    """
    if parameters is None:
        parameters = [
            column
            for column in sensitivities.columns
            if column not in ("window", "step")
        ]
    if len(parameters) < 2:
        raise ValueError("参数相关性矩阵至少需要两个参数。")
    matrix = sensitivities[list(parameters)].to_numpy(dtype=np.float64)
    matrix = matrix[np.all(np.isfinite(matrix), axis=1)]
    if matrix.shape[0] < 2:
        raise ValueError("有效观测不足，无法计算相关性。")
    norms = np.linalg.norm(matrix, axis=0)
    norms[norms == 0] = 1.0
    scaled = matrix / norms
    # 某列全为常数时 corrcoef 会产生 NaN；这里显式置 0（无信息 ≠ 完全相关）。
    correlation = np.corrcoef(scaled, rowvar=False)
    correlation = np.nan_to_num(correlation, nan=0.0)
    np.fill_diagonal(correlation, 1.0)
    return pd.DataFrame(correlation, index=list(parameters), columns=list(parameters))


def parameter_identifiability(
    sensitivities: pd.DataFrame,
    *,
    parameters: Sequence[str] | None = None,
) -> dict[str, float]:
    """由灵敏度矩阵计算条件数与列相关性。

    - ``condition_number``：最大/最小奇异值之比。> 1e3 表示至少一个参数
      几乎不可辨识（其变化可被另一个参数的方向抵消）。
    - ``correlation_<a>_<b>``：两列灵敏度的 Pearson 相关。接近 ±1 表示
      两者在数据上高度共线，不能同时从这批数据里辨识。

    若需要**完整**的相关结构（含三个以上参数的情形），用
    :func:`sensitivity_correlation_matrix`。
    """
    if parameters is None:
        parameters = [column for column in sensitivities.columns if column not in ("window", "step")]
    matrix = sensitivities[list(parameters)].to_numpy(dtype=np.float64)
    matrix = matrix[np.all(np.isfinite(matrix), axis=1)]
    if matrix.shape[0] < matrix.shape[1] or matrix.shape[1] == 0:
        raise ValueError("灵敏度矩阵的观测数不足，无法评估可辨识性。")
    # 按列做尺度归一化：不同通道/参数的量纲差异不应决定条件数。
    norms = np.linalg.norm(matrix, axis=0)
    norms[norms == 0] = 1.0
    scaled = matrix / norms
    singular = np.linalg.svd(scaled, compute_uv=False)
    smallest = singular[-1] if singular[-1] > 0 else np.inf
    result: dict[str, float] = {
        "condition_number": float(singular[0] / smallest),
        "n_observations": float(matrix.shape[0]),
    }
    correlation = np.corrcoef(scaled, rowvar=False)
    for i, first in enumerate(parameters):
        for j in range(i + 1, len(parameters)):
            result[f"correlation_{first}_{parameters[j]}"] = float(correlation[i, j])
    return result


def residual_structure(
    residual: np.ndarray,
    *,
    time_min: np.ndarray | None = None,
    temperature_k: np.ndarray | None = None,
) -> pd.DataFrame:
    """用简单线性回归检查残差是否随过程量趋势漂移。

    工业软测量里，残差与时间呈趋势通常意味着：

- **时滞估计错误**：残差与时间正相关且分段恒定；
- **慢变参数未建模**：残差随时间缓慢漂移（例如结垢未修正）；
- **结构误差**：残差与温度或进料浓度相关。

返回每个解释变量的斜率、R² 与样本数。R² 不高是**好事**；关键看斜率
的符号和量级是否与工艺预期一致。
    """
    residual = np.asarray(residual, dtype=np.float64).reshape(-1)
    rows = []
    candidates: list[tuple[str, np.ndarray]] = []
    if time_min is not None:
        candidates.append(("time_min", np.asarray(time_min, dtype=np.float64).reshape(-1)))
    if temperature_k is not None:
        candidates.append(
            ("temperature_K", np.asarray(temperature_k, dtype=np.float64).reshape(-1))
        )
    for name, values in candidates:
        if values.shape != residual.shape:
            raise ValueError(f"{name} 与残差形状不一致。")
        valid = np.isfinite(residual) & np.isfinite(values)
        if valid.sum() < 3:
            rows.append({"变量": name, "斜率": np.nan, "R2": np.nan, "n": int(valid.sum())})
            continue
        x = values[valid]
        y = residual[valid]
        x_centered = x - x.mean()
        denominator = float(np.sum(x_centered**2))
        slope = float(np.sum(x_centered * (y - y.mean())) / denominator) if denominator > 0 else 0.0
        intercept = float(y.mean() - slope * x.mean())
        predicted = intercept + slope * x
        total = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - float(np.sum((y - predicted) ** 2)) / total if total > 0 else 0.0
        rows.append({"变量": name, "斜率": slope, "R2": r2, "n": int(valid.sum())})
    return pd.DataFrame(rows)


def template_residual_report(
    events: pd.DataFrame,
    *,
    residual_column: str = "residual_grey",
    process_time_column: str = "process_time_min",
) -> pd.DataFrame:
    """影子运行化验事件的残差结构检查。

    这是判断“死时间是否对齐正确”的主要手段：若死时间偏小，残差会与
    过程时刻呈系统性斜率；若偏大，斜率反号。用 ``residual_structure``
    给出定量结果，而不是靠看图。
    """
    if events.empty:
        return pd.DataFrame()
    return residual_structure(
        events[residual_column].to_numpy(),
        time_min=events[process_time_column].to_numpy(),
    )


def calibrate_domain_box(
    frame: pd.DataFrame,
    *,
    scaling: "FeatureScaling",
    quantile: float = 0.95,
    safety_factor: float = 1.2,
    concentration_column: str = "c_a_out_mol_L",
) -> float:
    """从训练数据推导 ``LongHorizonConfig.domain_box`` 的推荐值。

    为什么必须校准
    --------------
    ``domain_box`` 是"多远算外推"的判据，单位是归一化特征尺度。它必须
    反映**该装置实际的工况变化幅度**：

    - 设太大 → 门控永远不触发，机制形同虚设（本仓库初版就是这个错误：
      ``box=3.0`` 相当于温度偏离 60 K，远大于实际工况波动）；
    - 设太小 → 正常波动也被判为外推，修正网络被无谓关闭。

    本函数把训练集的归一化特征按分位数统计，取 ``quantile`` 分位（默认
    95%）再乘安全系数。这样 box 自动贴合该装置的操作范围，而不是沿用
    示例里的数字。

    Parameters
    ----------
    frame
        训练数据长表，需含 ``CONTROL_COLUMNS``、``T_REACTOR`` 与浓度列。
    concentration_column
        浓度列名。合成数据用 ``c_a_online_mol_L``（在线分析仪），现场可传
        化验对齐后的列。该列仅用于估计**工况范围**，不参与训练。
    scaling
        与模型一致的 ``FeatureScaling``（尺度必须匹配，否则归一化错了）。
    quantile
        取多少分位作为"正常范围边界"。95% 表示允许 5% 的训练点落在盒外。
    safety_factor
        在分位值上再放宽的比例，避免边界工况被一刀切判为外推。

    Returns
    -------
    float
        建议的 ``domain_box``。至少为 0.1，避免退化。
    """
    from .config import COL_C_A0, COL_T_C, COL_T_IN, COL_T_REACTOR, COL_TAU

    required = {
        COL_C_A0,
        COL_T_IN,
        COL_T_C,
        COL_TAU,
        COL_T_REACTOR,
        concentration_column,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"校准 domain_box 缺少列：{sorted(missing)}")

    import numpy as _np

    # 与 mechanistic_layer._rate_features 完全一致的归一化。
    concentration = (
        frame[concentration_column].to_numpy(dtype=float) - scaling.c_a_ref_mol_l
    ) / scaling.c_a_ref_mol_l
    reactor = (
        frame[COL_T_REACTOR].to_numpy(dtype=float) - scaling.temperature_ref_k
    ) / scaling.temperature_scale_k
    feed = (
        frame[COL_C_A0].to_numpy(dtype=float) - scaling.c_a0_ref_mol_l
    ) / scaling.c_a0_scale_mol_l
    feed_t = (
        frame[COL_T_IN].to_numpy(dtype=float) - scaling.temperature_ref_k
    ) / scaling.temperature_scale_k
    coolant = (
        frame[COL_T_C].to_numpy(dtype=float) - scaling.coolant_ref_k
    ) / scaling.coolant_scale_k
    tau = (
        frame[COL_TAU].to_numpy(dtype=float) - scaling.tau_ref_min
    ) / scaling.tau_scale_min

    columns = _np.column_stack(
        [concentration, reactor, feed, feed_t, coolant, tau]
    )
    # 逐维取 |z| 的分位数，再取各维最大值：这样"任何一维超出正常范围"
    # 都会触发门控，而不是被其他维的平均值稀释。
    per_dimension = _np.nanquantile(_np.abs(columns), quantile, axis=0)
    value = float(_np.nanmax(per_dimension)) * safety_factor
    return max(value, 0.1)


__all__ = [
    "calibrate_domain_box",
    "parameter_identifiability",
    "residual_structure",
    "sensitivity_correlation_matrix",
    "sensitivity_matrix",
    "template_residual_report",
]
