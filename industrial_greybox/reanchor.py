"""自适应状态重锚（soft-sensor update）：置信度门控 + 指数平滑。

要解决的问题
------------
旧实现是``bias = clip(residual, ±max_bias)`` 后一次性加到状态上。这在
真实工厂有两个具体毛病：

1. **无法区分"仪表坏值"与"真实工况漂移"。** 两者的表现都是"化验与模型
   差很多"。盲目相信坏值会把内部状态拽偏；盲目忽略大残差又会漏掉真实的
   模型失配。
2. **瞬时加性修正会制造状态阶跃。** 在准刚性工况下，一个 0.1 mol/L 的
   瞬间跳跃可能让积分器产生非物理的振荡，甚至触发数值守护回退。

本模块的两层处理
----------------
**第一层：稳健置信度门控**

- 用滑动窗口中位绝对偏差（MAD）估计"正常残差尺度"，其中
  ``scale = 1.4826 * MAD`` 在正态噪声下等于标准差；
- 残差超过 ``residual_gate_sigma`` 倍尺度时，置信度按 smoothstep 衰减，
  到 ``+residual_gate_taper_sigma`` 倍时降为 0；
- **连续确认**：孤立的一次大残差大概率是坏值；连续 ``drift_confirm_count``
  次都大，才判定为真实漂移，此时放宽门控（恢复较高置信度）。

  这个区分很重要：坏值通常在下一个采样点就恢复，而真实漂移会持续。

**第二层：指数平滑过渡**

```
applied = α · confidence · bias
```

而不是一次到位。α 小则过渡平滑但收敛慢，需要更频繁的化验；默认 0.5。

为什么用 MAD 而不是标准差
-------------------------
标准差对坏值敏感（一个 10σ 的坏值能把标准差抬高数倍，从而让后续所有
残差都"看起来正常"）。MAD 的崩溃点是 50%，更适合工业数据的坏值工况。

边界
----
本模块只决定"施加多少修正"，**不**决定状态应该是什么。物理量纲、单位
和状态定义仍由调用方（影子运行）负责。它也不修改化验值本身。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import ShadowConfig


@dataclass
class ReanchorDecision:
    """一次重锚的决策记录，可逐条写进诊断日志。"""

    residual_mol_l: float
    """化验值与过程时刻预测之差（lab - ŷ）。"""

    scale_mol_l: float
    """当前稳健残差尺度（MAD 换算的标准差）。窗口不足时用下限。"""

    confidence: float
    """置信度 ∈ [0, 1]。1 表示残差完全可信。"""

    applied_bias_mol_l: float
    """实际施加到状态的修正量（已含 α、置信度与限幅）。"""

    clipped: bool
    """是否被 ``max_bias_correction_mol_l`` 限幅。"""

    drift_confirmed: bool
    """是否判定为真实漂移（连续大残差）而非孤立坏值。"""

    mode: str
    """``"fixed"`` 或 ``"adaptive"``，便于对比排查。"""


@dataclass
class AdaptiveReanchor:
    """有状态的自适应重锚器（每条运行一个实例）。

    用法::

        reanchor = AdaptiveReanchor(shadow_config)
        ...
        decision = reanchor.update(residual, current_bias_applied=0.0)
        state[0, 0] += decision.applied_bias_mol_l

    Notes
    -----
    该类持有**运行内**的状态（残差历史、连续计数）。跨运行复用会污染
    稳健尺度估计，因此影子运行应每条 ``run_id`` 新建一个实例。
    """

    config: ShadowConfig
    _history: list[float] = field(default_factory=list, repr=False)
    _consecutive_large: int = field(default=0, repr=False)
    _fallback_scale: float = field(default=0.0, repr=False)

    def __post_init__(self) -> None:
        if self._fallback_scale <= 0:
            # 窗口不足时的尺度下限：取限幅值的一小部分。它必须显著小于
            # max_bias，否则第一条化验就永远落在"正常"范围内。
            self._fallback_scale = max(self.config.max_bias_correction_mol_l / 4.0, 1e-4)

    @property
    def scale(self) -> float:
        """当前稳健残差尺度（mol/L）——"正常波动有多大"的估计。

        这是整个自适应重锚的**基准尺子**：后面判断"这个残差算大还是小"，
        都是拿残差除以它。所以它必须既稳健又不会坍缩。

        逐步拆解（以历史残差 [0.008, -0.006, 0.010, 0.007] 为例）::

            1. center = median(历史) = 0.0075        ← 用中位数而非均值
            2. 绝对偏差 = |0.008-0.0075|=0.0005, |−0.006−0.0075|=0.0135,
                          |0.010−0.0075|=0.0025, |0.007−0.0075|=0.0005
            3. mad = median([0.0005, 0.0135, 0.0025, 0.0005]) = 0.0015
            4. scale = 1.4826 × mad = 0.00222

        为什么乘 1.4826
        ---------------
        ``1.4826 × MAD`` 在**正态分布**下恰好等于标准差。这是一个统计学
        常数：正态分布的中位绝对偏差 = 0.6745σ，取倒数即 1.4826。所以
        ``scale`` 与熟悉的标准差同量纲，但它对坏值稳健得多。

        为了让改动的量级看得见，MAD/标准差的抗坏值能力对比：若历史是
        20 个 ~0.006 的残差再混入 1 个 0.5 的坏值：

        - 标准差：被这个 0.5 抬高到约 0.108（20 倍）；
        - MAD：中位数完全不受影响，仍是约 0.004。

        标准差被抬高的后果很严重：之后所有正常残差看起来都"很小"，
        门控永远不关闭，坏值会被当成真实漂移吃进去。

        为什么必须对 MAD 设下限
        -----------------------
        若窗口内残差几乎恒定（例如模型存在一个稳定的系统偏差，历史全是
        0.0091, 0.0090, 0.0091, 0.0090），``MAD ≈ 0.00005``，尺度会坍缩到
        接近 0。此时一个 0.01 的残差会被算成 0.01/0.00007 ≈ 140 倍尺度，
        判定为"极端异常"，门控关闭、修正停止——漂移**永远不会**被确认。

        正确解读是：**"残差高度一致"说明这是一个可信的稳定偏差**，而不是
        "正常波动为零"。因此尺度不得低于兜底下限的一半。
        """
        if len(self._history) < 3:
            # 样本太少时用兜底尺度。它来自 max_bias/4（默认 0.15/4 ≈
            # 0.0375），是"量级上合理"的初值，避免第一条化验就误判。
            return self._fallback_scale
        values = np.asarray(self._history, dtype=float)
        center = float(np.median(values))
        mad = float(np.median(np.abs(values - center)))
        scale = 1.4826 * mad
        floor = self._fallback_scale * 0.5
        return max(scale, floor)

    def update(self, residual_mol_l: float) -> ReanchorDecision:
        """根据一条化验残差给出本次应施加的修正量。

        Parameters
        ----------
        residual_mol_l
            ``lab - ŷ(process_time)``，单位 mol/L。正值表示模型低估了浓度。

        Returns
        -------
        ReanchorDecision
            包含施加量、置信度、是否限幅、是否判定为漂移。
        """
        residual = float(residual_mol_l)
        if not np.isfinite(residual):
            # 非有限残差不能参与统计，也不能施加修正。
            return ReanchorDecision(
                residual_mol_l=residual,
                scale_mol_l=self.scale,
                confidence=0.0,
                applied_bias_mol_l=0.0,
                clipped=False,
                drift_confirmed=False,
                mode=self.config.reseed_mode,
            )

        scale_before = self.scale
        if self.config.reseed_mode == "fixed":
            decision = self._fixed(residual, scale_before)
        else:
            decision = self._adaptive(residual, scale_before)

        self._remember(residual)
        return decision

    # ------------------------------------------------------------------
    # 两种模式
    # ------------------------------------------------------------------
    def _fixed(self, residual: float, scale: float) -> ReanchorDecision:
        """旧行为：截断后全量施加。保留用于对照与回归测试。"""
        limit = self.config.max_bias_correction_mol_l
        applied = float(np.clip(residual, -limit, limit))
        return ReanchorDecision(
            residual_mol_l=residual,
            scale_mol_l=scale,
            confidence=1.0,
            applied_bias_mol_l=applied,
            clipped=bool(abs(residual) > limit),
            drift_confirmed=False,
            mode="fixed",
        )

    def _adaptive(self, residual: float, scale: float) -> ReanchorDecision:
        """自适应模式：把"该信多少"和"修多少"分成三步算。

        先用一个完整例子把三步串起来，再看代码。设：

        - 当前稳健尺度 ``scale = 0.00222 mol/L``（见 :attr:`scale`）
        - 门控参数 ``gate_sigma = 4``、``taper = 4``（即 4σ→8σ 之间衰减）
        - 平滑系数 ``α = 0.5``、限幅 ``max_bias = 0.15 mol/L``

        **例 A：正常残差 0.004**（约 1.8σ）::

            幅度 = 0.004 / 0.00222 = 1.8
            1.8 < 4  →  x = 0  →  confidence = 1.0
            施加 = 0.5 × 1.0 × 0.004 = 0.002

        结论：正常修正，且只施加一半（指数平滑），避免状态跳变。

        **例 B：孤立坏值 0.5**（约 225σ）::

            幅度 = 225
            x = clip((225-4)/4, 0, 1) = 1  →  confidence = 0
            连续大残差计数 = 1（< drift_confirm_count=3）
            施加 = 0.5 × 0 × 0.5 = 0    ← 完全忽略

        结论：坏值被丢弃。若照旧算法，0.5 会被截断成 0.15 直接打进去，
        把内部状态拽偏，而"真值"其实没变。

        **例 C：持续漂移，连续第 3 次 0.5**::

            幅度 = 225 → 基础 confidence = 0（门控关闭）
            连续计数 = 3 ≥ 3  →  drift_confirmed = True
            confidence = max(0, 0.8) = 0.8   ← 恢复信任
            施加 = 0.5 × 0.8 × 0.5 = 0.2 → 限幅到 0.15

        结论：坏值不会连续出现三次，所以连读三次大残差必然是真实漂移，
        此时应当相信它并修正。
        """
        cfg = self.config
        # 幅度以"几个稳健尺度"计。除以 scale 让判据与浓度绝对量级无关：
        # 换成 ppm 单位时，residual 和 scale 同比例放大，幅度不变。
        magnitude = abs(residual) / max(scale, 1e-12)

        # ------------------------------------------------------------------
        # 第 1 步：基础置信度 —— 这个残差有多可信？
        # ------------------------------------------------------------------
        # smoothstep 曲线：x=0 时 1，x=1 时 0，中间平滑过渡。
        #   公式 1 - x²(3-2x) 的取值：x=0→1, x=0.5→0.5, x=1→0
        # 为什么不用阶梯（残差>阈值就信/不信）：
        #   阶梯会在阈值处突变，导致同一个模型在不同工况下行为跳变，
        #   而且对训练不可导。smoothstep 在两端导数为 0，过渡自然。
        low = cfg.residual_gate_sigma          # 4σ：此以内完全信任
        high = low + cfg.residual_gate_taper_sigma   # 8σ：此以外完全不信
        x = float(np.clip((magnitude - low) / max(high - low, 1e-12), 0.0, 1.0))
        confidence = 1.0 - x * x * (3.0 - 2.0 * x)

        # ------------------------------------------------------------------
        # 第 2 步：连续确认 —— 这是坏值还是真实漂移？
        # ------------------------------------------------------------------
        # 核心假设：**坏值是孤立的，真实漂移是持续的**。
        # 仪表跳变通常下一个采样点就恢复；模型失配/工况变化会一直存在。
        #
        # 注意计数条件与门控**解耦**（用 drift_magnitude_sigma 而不是
        # "门控是否已关闭"）。若用后者会形成自锁：
        #   门控关闭 → 修正被压到 0 → 残差不会因此变小
        #            → 计数条件仍不满足 → 永远无法确认漂移
        # 这是本项目实际踩过的坑，单测 test_sustained_drift_is_confirmed
        # 专门锁住这个行为。
        sustained = magnitude > cfg.drift_magnitude_sigma
        self._consecutive_large = self._consecutive_large + 1 if sustained else 0
        drift_confirmed = self._consecutive_large >= cfg.drift_confirm_count
        if drift_confirmed:
            # 真实漂移：残差方向一致且持续，应较高程度地相信它。
            # 恢复 0.8 而不是 1.0，仍保留平滑过渡，避免状态阶跃。
            confidence = max(confidence, 0.8)

        # ------------------------------------------------------------------
        # 第 3 步：指数平滑 + 限幅 —— 这次实际修多少？
        # ------------------------------------------------------------------
        # α 的作用：不是一次把偏差全补上，而是每来一条化验补 α 比例。
        #   α=0.5 时，连续 2 条化验后补到 75%，3 条后补到 87.5%。
        # 为什么需要平滑：瞬时跳变（例如 0.1 mol/L）在准刚性工况下可能
        # 触发积分器振荡，甚至被数值守护回退，反而更糟。
        #
        # 限幅是最后的安全网：即使前面判断全错，单次修正也不会超过
        # max_bias。超出说明有更深层问题（外推/仪表故障/死时间错误）。
        limit = cfg.max_bias_correction_mol_l
        smoothed = cfg.reseed_smoothing * confidence * residual
        applied = float(np.clip(smoothed, -limit, limit))
        return ReanchorDecision(
            residual_mol_l=residual,
            scale_mol_l=scale,
            confidence=float(confidence),
            applied_bias_mol_l=applied,
            # clipped 记录"原本要施加的"是否被砍，而不是"最终值"被砍，
            # 这样报表能看出"安全网被触发过"，即使最终值为 0。
            clipped=bool(abs(smoothed) > limit),
            drift_confirmed=drift_confirmed,
            mode="adaptive",
        )

    # ------------------------------------------------------------------
    # 统计
    # ------------------------------------------------------------------
    def _remember(self, residual: float) -> None:
        self._history.append(residual)
        window = self.config.residual_scale_window
        if len(self._history) > window:
            del self._history[: len(self._history) - window]

    def reset(self) -> None:
        """清空运行内状态（切换 ``run_id`` 时调用）。"""
        self._history.clear()
        self._consecutive_large = 0


__all__ = ["AdaptiveReanchor", "ReanchorDecision"]
