"""可微数值积分与数值安全层（Keras 3 ops，后端无关）。

为什么需要这一层
----------------
教程中的显式 Euler 固定步长在两类情况下会失真：

1. **刚性**：停留时间很短（``dt/tau`` 不小）或夹套冷却很强
   （``dt*UA`` 不小）时，显式格式的稳定域要求步长极小；一旦越过稳定
   边界，温度会出现非物理的振荡甚至发散。
2. **长周期滚动**：一次预测要积分很多步，浮点误差与模型误差逐步累积，
   一个 NaN 会让整条轨迹、整批梯度都失效。

本模块提供：

- ``rk4``：经典四阶显式 Runge–Kutta（局部误差 O(h^5)），支持子步。
- ``imex``：对**线性衰减项**（``-C_A/tau``、``-(1/tau+UA)T``）做后向
  Euler，对**反应/放热非线性项**显式。它对标量线性项无条件稳定，适合
  强冷却和短停留时间的准刚性工况；反应项仍显式，因此非常强的反应速率
  需要配合子步或裁剪，见 ``docs`` 说明。
- ``euler``：显式 Euler，仅用于回归对照。
- ``guard_state``：NaN/Inf 回退到上一有效状态，并按守护区间裁剪。

所有函数只使用 ``keras.ops``，因此可在 TensorFlow / JAX / PyTorch
后端下自动微分。物理守恒本身不由本层保证，守护区间只防止数值爆炸。
"""

from __future__ import annotations

from typing import Any, Callable, Literal

from keras import ops

#: 状态向量按 [C_A, T] 排列。
StateTensor = object  # keras.ops 张量；运行时类型由后端决定。
VectorField = Callable[[StateTensor], StateTensor]
IntegratorName = Literal["euler", "rk4", "imex"]


# ---------------------------------------------------------------------------
# 数值安全
# ---------------------------------------------------------------------------
def guard_state(
    state: StateTensor,
    fallback: StateTensor,
    *,
    c_a_bounds: tuple[float, float],
    t_bounds: tuple[float, float],
) -> tuple[StateTensor, StateTensor, StateTensor]:
    """把状态拉回物理/数值守护区间，并隔离 NaN/Inf。

    Parameters
    ----------
    state
        ``(batch, 2)`` 状态，列顺序 ``[C_A, T]``。
    fallback
        上一步的有效状态；当本步出现非有限值时整行回退到它。
    c_a_bounds, t_bounds
        数值守护区间，**不是**工艺安全限值。上界只用于阻止积分爆炸。

    Returns
    -------
    safe_state
        清洗后的状态。
    nonfinite_mask
        ``(batch,)`` bool，True 表示该样本本步出现 NaN/Inf 并已回退。
    clipped_mask
        ``(batch,)`` bool，True 表示该样本本步被区间裁剪（数值提示）。

    Notes
    -----
    回退而不是把 NaN 替换成 0：0 是有物理意义的浓度，伪造它会静默污染
    目标值和后续梯度。保留“上一步”等价于零阶保持，并在报表中显式计数。
    """
    finite = ops.all(ops.isfinite(state), axis=-1)
    restored = ops.where(ops.expand_dims(finite, -1), state, fallback)

    c_a_low, c_a_high = c_a_bounds
    t_low, t_high = t_bounds
    bounds = ops.convert_to_tensor(
        [[c_a_low, t_low], [c_a_high, t_high]], dtype=restored.dtype
    )
    clipped = ops.clip(restored, bounds[0], bounds[1])
    clipped_mask = ops.any(
        ops.abs(clipped - restored) > ops.cast(1e-7, restored.dtype), axis=-1
    )
    return clipped, ops.logical_not(finite), clipped_mask


def check_finite_gradients(value: StateTensor, name: str = "state") -> StateTensor:
    """把非有限**中间量**替换为 0，避免它们污染后续计算。

    .. note::
       本函数用于前向张量（如某个中间激活），不是梯度钩子。Keras 3 没有
       跨后端的“梯度清洗”入口；实际的梯度保护由
       ``compile_greybox(clipnorm=...)`` 完成。若训练中反复出现非有限值，
       应回查积分步长、学习率和物理区间，而不是依赖任何“清洗”。
    """
    nan_mask = ops.logical_not(ops.all(ops.isfinite(value), axis=-1, keepdims=True))
    # stop_gradient 防止该掩码本身参与求导。
    return ops.where(
        nan_mask,
        ops.stop_gradient(ops.zeros_like(value)),
        value,
    )


# ---------------------------------------------------------------------------
# 积分格式
# ---------------------------------------------------------------------------
def _clip_derivative(
    derivative: StateTensor, clip: tuple[float, float] | None
) -> StateTensor:
    if clip is None:
        return derivative
    low, high = clip
    return ops.clip(
        derivative,
        ops.cast(low, derivative.dtype),
        ops.cast(high, derivative.dtype),
    )


def _as_dtype(value: Any, like: Any) -> Any:
    """把 forcing/decay 等常量张量对齐到状态张量的 dtype。

    调用方可能传入 float64 的 numpy 数组（例如从配置直接构造），若状态是
    float32，后端会在乘法处抛 dtype 不匹配。这里统一转换，避免每个调用点
    各写一遍 ``ops.cast``。
    """
    tensor = ops.convert_to_tensor(value)
    if ops.dtype(tensor) != ops.dtype(like):
        tensor = ops.cast(tensor, ops.dtype(like))
    return tensor


def full_rhs(
    state: StateTensor,
    *,
    decay_per_min: StateTensor,
    forcing: StateTensor,
    nonlinear: VectorField,
) -> StateTensor:
    """组合 RHS：``d state/dt = forcing - decay * state + nonlinear(state)``。

    ``decay`` 只包含线性衰减项（单位 min⁻¹），``forcing`` 只包含与状态
    无关的进料/夹套项；这样 IMEX 可以精确地对线性部分做隐式处理。
    """
    return (
        _as_dtype(forcing, state)
        - _as_dtype(decay_per_min, state) * state
        + nonlinear(state)
    )


def euler_step(
    state: StateTensor,
    dt_min: float,
    *,
    decay_per_min: StateTensor,
    forcing: StateTensor,
    nonlinear: VectorField,
    derivative_clip: tuple[float, float] | None = None,
) -> StateTensor:
    """显式 Euler（一阶）。仅用于对照教程基线。"""
    derivative = full_rhs(
        state, decay_per_min=decay_per_min, forcing=forcing, nonlinear=nonlinear
    )
    return state + ops.cast(dt_min, state.dtype) * _clip_derivative(
        derivative, derivative_clip
    )


def rk4_step(
    state: StateTensor,
    dt_min: float,
    *,
    decay_per_min: StateTensor,
    forcing: StateTensor,
    nonlinear: VectorField,
    derivative_clip: tuple[float, float] | None = None,
) -> StateTensor:
    """经典四阶 Runge–Kutta。

    非线性项在每个 stage 重新计算（Arrhenius 速率依赖 stage 温度），
    因此对温升较快的放热工况比“在步首冻结速率”更准确。
    """

    def f(x: StateTensor) -> StateTensor:
        derivative = full_rhs(
            x, decay_per_min=decay_per_min, forcing=forcing, nonlinear=nonlinear
        )
        return _clip_derivative(derivative, derivative_clip)

    h = ops.cast(dt_min, state.dtype)
    k1 = f(state)
    k2 = f(state + 0.5 * h * k1)
    k3 = f(state + 0.5 * h * k2)
    k4 = f(state + h * k3)
    return state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def imex_step(
    state: StateTensor,
    dt_min: float,
    *,
    decay_per_min: StateTensor,
    forcing: StateTensor,
    nonlinear: VectorField,
    derivative_clip: tuple[float, float] | None = None,
) -> StateTensor:
    """线性隐式 / 非线性显式 Euler（一阶，对线性衰减无条件稳定）。

    对 ``d x/dt = b - a x + n(x)`` 做后向 Euler（仅 ``a x`` 隐式）::

        x_{n+1} = (x_n + h (b + n(x_n))) / (1 + h a)

    对 CSTR 而言 ``a = [1/tau, 1/tau + UA]``，因此短停留时间、强冷却
    都不再受显式稳定域 ``h a < 2`` 的限制。局限：反应项 ``n(x)`` 仍是
    显式的，若 ``h k_eff`` 很大（极快反应），仍需减小 ``dt`` 或增加子步。
    """
    h = ops.cast(dt_min, state.dtype)
    explicit = _as_dtype(forcing, state) + _clip_derivative(
        nonlinear(state), derivative_clip
    )
    numerator = state + h * explicit
    denominator = 1.0 + h * _as_dtype(decay_per_min, state)
    return numerator / denominator


def integrate_step(
    state: StateTensor,
    dt_min: float,
    *,
    method: IntegratorName,
    decay_per_min: StateTensor,
    forcing: StateTensor,
    nonlinear: VectorField,
    derivative_clip: tuple[float, float] | None = None,
) -> StateTensor:
    """按名称分派一步积分。"""
    kwargs = dict(
        decay_per_min=decay_per_min,
        forcing=forcing,
        nonlinear=nonlinear,
        derivative_clip=derivative_clip,
    )
    if method == "euler":
        return euler_step(state, dt_min, **kwargs)
    if method == "rk4":
        return rk4_step(state, dt_min, **kwargs)
    if method == "imex":
        return imex_step(state, dt_min, **kwargs)
    raise ValueError(f"未知积分方法: {method}")


def integrate_interval(
    state: StateTensor,
    dt_min: float,
    *,
    method: IntegratorName,
    substeps: int,
    decay_per_min: StateTensor,
    forcing: StateTensor,
    nonlinear: VectorField,
    derivative_clip: tuple[float, float] | None = None,
) -> StateTensor:
    """在一个采样间隔内做 ``substeps`` 次子步积分。

    控制量在采样间隔内按零阶保持（ZOH）处理，这是 DCS 数据的常见假设；
    若实际是斜坡变化，应减小采样间隔或在 RHS 中加入插值。
    """
    if substeps < 1:
        raise ValueError("substeps 必须 >= 1。")
    sub_dt = dt_min / substeps
    for _ in range(substeps):
        state = integrate_step(
            state,
            sub_dt,
            method=method,
            decay_per_min=decay_per_min,
            forcing=forcing,
            nonlinear=nonlinear,
            derivative_clip=derivative_clip,
        )
    return state


# ---------------------------------------------------------------------------
# 稳定性诊断（CPU，供测试/报表使用；不参与训练图）
# ---------------------------------------------------------------------------
def explicit_stability_ratio(
    dt_min: float, tau_min: float, ua_per_min: float
) -> float:
    """显式 Euler 在线性部分的最大稳定比 ``h * a_max``。

    经验判据：``< 1`` 安全，``1..2`` 边缘，``> 2`` 可能振荡/发散。用于在
    报表中解释为什么某条工况需要 IMEX 或更小步长。反应项的附加刚性
    （``h*k_eff``）不包含在这个比值内。
    """
    if tau_min <= 0:
        raise ValueError("tau_min 必须为正。")
    a_max = 1.0 / tau_min + max(ua_per_min, 0.0)
    return dt_min * a_max
