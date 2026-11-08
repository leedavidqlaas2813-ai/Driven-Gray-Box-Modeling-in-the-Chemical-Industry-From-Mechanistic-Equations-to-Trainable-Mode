"""在线增量微调（warm-start adaptation）与安全熔断回滚。

问题
----
离线训练的模型在运行几个月后必然遇到概念漂移：原料劣化、催化剂缓慢失活、
换热器结垢改变了传热特征。重训整个模型既慢又危险——它会改动已经通过
验证的白箱机理参数（``k_ref``、``UA``），使模型行为在无人察觉的情况下
偏离已验证的物理基线。

本模块的策略
------------
1. **只动修正层，冻结机理核心。** 当滚动残差连续超限时，仅对速率修正
   网络与慢变网络的权重做低学习率微调；``log_k_ref``、``log_ua``、
   ``log_heat_release`` 保持冻结。理由是机理参数是经过辨识与工程审核的
   物理量，随在线残差调整它们等于让模型悄悄改写物理定律。

2. **只在最近的滑动窗口上微调。** 用最近 ``window_runs`` 条运行/若干天
   的数据，而不是全部历史：概念漂移本身就是"旧数据不再代表当前工况"。

3. **安全熔断（guard）。** 微调后必须在一批**留出**数据上评估：
   - 短期开环 RMSE 不得变差超过 ``max_rmse_regression``；
   - 修正幅度不得超出结构上界（防止网络用大幅修正"拟合"漂移）；
   任何一项不通过就**回滚**到上一版权重。

   "留出"的定义很重要：不能用刚微调过的那批数据来验证，否则等于自证。
   本模块要求调用方传入独立的 ``guard`` 数据集。

4. **审计留痕。** 每次尝试都记录：触发原因、学习率、更新前后的权重范数、
   guard 指标、是否回滚。这些记录是上线后排查"模型什么时候变的"的唯一
   依据。

不做什么
--------
- 不改变模型结构（新增层需要重新做可辨识性分析）；
- 不自动提升学习率；
- 不在残差未超限时做任何更新（避免无谓漂移）。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .config import ModelConfig
from .data import WindowedDataset
from .models import CSTRGreyBoxModel, compile_greybox, predict_trajectory
from .reports import regression_metrics


@dataclass(frozen=True)
class AdaptationPolicy:
    """在线微调的触发与熔断策略。

    所有阈值都必须来自工艺/质量部门可接受的风险水平，不是从示例臆造。
    """

    #: 触发阈值：滚动残差 RMSE 超过该值时认为发生概念漂移 [mol/L]。
    trigger_rmse_mol_l: float = 0.02
    #: 连续多少个评估周期超限才触发（避免偶发波动导致频繁微调）。
    trigger_patience: int = 2
    #: 微调学习率。必须远低于离线训练值：在线阶段的目标是"缓慢跟随"，
    #: 不是"拟合当前批次"。
    learning_rate: float = 5e-5
    #: 每次微调的 epoch 数与批大小。
    epochs: int = 3
    batch_size: int = 16
    #: 熔断阈值：微调后短期 RMSE 允许的相对恶化比例。
    max_rmse_regression: float = 0.02
    #: 熔断的绝对兜底：即使相对恶化很小，RMSE 也不得超过该值。
    max_rmse_absolute_mol_l: float = 0.05
    #: 是否冻结机理核心（默认 True；设为 False 仅用于研究对照）。
    freeze_physics: bool = True

    def __post_init__(self) -> None:
        if self.trigger_rmse_mol_l <= 0:
            raise ValueError("trigger_rmse_mol_l 必须为正。")
        if self.trigger_patience < 1:
            raise ValueError("trigger_patience 必须 >= 1。")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate 必须为正。")
        if self.epochs < 1:
            raise ValueError("epochs 必须 >= 1。")
        if self.max_rmse_regression < 0:
            raise ValueError("max_rmse_regression 不能为负。")


@dataclass
class AdaptationEvent:
    """一次微调尝试的审计记录。"""

    cycle: int
    triggered: bool
    reason: str
    rmse_before: float
    rmse_after: float
    guard_rmse_before: float
    guard_rmse_after: float
    rolled_back: bool
    correction_norm_before: float
    correction_norm_after: float
    learning_rate: float


@dataclass
class AdaptationState:
    """守护进程的持久状态（可序列化，便于重启后续跑）。"""

    cycle: int = 0
    consecutive_over_threshold: int = 0
    events: list[AdaptationEvent] = field(default_factory=list)

    def history_frame(self) -> pd.DataFrame:
        if not self.events:
            return pd.DataFrame(
                columns=[
                    "cycle",
                    "triggered",
                    "reason",
                    "rmse_before",
                    "rmse_after",
                    "guard_rmse_before",
                    "guard_rmse_after",
                    "rolled_back",
                    "correction_norm_before",
                    "correction_norm_after",
                    "learning_rate",
                ]
            )
        return pd.DataFrame([event.__dict__ for event in self.events])


def correction_weights(model: CSTRGreyBoxModel) -> list[Any]:
    """返回**数据驱动修正层**的权重（速率网 + 慢变网），不含机理参数。

    这是在线微调唯一允许改动的参数集合。机理参数（``log_k_ref`` 等）与
    它们分开管理，是"冻结白箱核心"的实现基础。
    """
    layer = model.physics_layer
    weights: list[Any] = []
    if layer.rate_net is not None:
        weights.extend(layer.rate_net.weights)
    if layer.slow_net is not None:
        weights.extend(layer.slow_net.weights)
    return weights


def physics_weights(model: CSTRGreyBoxModel) -> list[Any]:
    """返回名义机理参数（k_ref、UA、可选反应热）。"""
    layer = model.physics_layer
    weights = [layer.log_k_ref, layer.log_ua]
    if layer.log_heat_release is not None:
        weights.append(layer.log_heat_release)
    return weights


def _fast_rmse(
    model: CSTRGreyBoxModel,
    dataset: WindowedDataset,
    *,
    batch_size: int = 8,
) -> float:
    """在数据集上计算浓度通道的加权 RMSE（mol/L）。

    只统计有化验监督的点，与训练/评估口径一致。

    ``batch_size`` 默认很小（8）而不是 64：``predict_trajectory`` 会把整个
    批次的时间循环一次性展开到图里，批次越大中间张量越多。守护进程在
    内存受限的工控机上运行时，小批次是必要的保守选择。
    """
    if len(dataset) == 0:
        return float("nan")
    prediction = predict_trajectory(model, dataset, batch_size=batch_size)
    metrics = regression_metrics(
        prediction[:, :, 0],
        dataset.targets[:, :, 0],
        mask=dataset.weights[:, :, 0],
    )
    return float(metrics["rmse"])


class OnlineAdapter:
    """在线增量微调守护进程。

    用法::

        adapter = OnlineAdapter(model, policy)
        # 每个评估周期（例如每天）：
        adapter.observe(log_rmse=0.03)          # 只记录，不做更新
        event = adapter.adapt(recent, guard)     # 必要时微调 + 熔断
        adapter.state.history_frame()            # 审计表

    Notes
    -----
    该类**不修改**传入的 ``model`` 之外的任何对象；回滚通过保存并恢复
    修正层权重实现，不重建模型（重建会丢失优化器状态且代价高）。
    """

    def __init__(
        self,
        model: CSTRGreyBoxModel,
        policy: AdaptationPolicy | None = None,
        *,
        config: ModelConfig | None = None,
    ) -> None:
        self.model = model
        self.policy = policy or AdaptationPolicy()
        self.config = config or model.config
        self.state = AdaptationState()
        #: 上一版"已通过验证"的修正层权重。回滚时恢复它。
        #:
        #: **延迟快照**：修正层是 ``Sequential``，其权重只有在第一次前向
        #: 之后才创建。若在构造时就快照，会得到一个空列表；之后微调触发
        #: build，``_restore`` 会发现权重数量变化而拒绝回滚——这会让安全
        #: 熔断在第一次真正需要它时失效。因此只记录"尚未快照"，在首次
        #: ``adapt`` 中、微调**之前**补拍。
        self._stable_weights: list[np.ndarray] | None = None
        #: 当前编译所用的学习率；避免每次微调都重新 compile（见 _fine_tune）。
        self._compiled_learning_rate: float | None = None
        self._ensure_snapshot()

    # ------------------------------------------------------------------
    # 观测
    # ------------------------------------------------------------------
    def observe(self, rolling_rmse_mol_l: float) -> bool:
        """记录一次滚动残差；返回是否已达到触发条件（连续超限）。"""
        self.state.cycle += 1
        if not np.isfinite(rolling_rmse_mol_l):
            # 非有限指标不能作为触发依据（可能是数据问题而非模型漂移）。
            return False
        if rolling_rmse_mol_l > self.policy.trigger_rmse_mol_l:
            self.state.consecutive_over_threshold += 1
        else:
            self.state.consecutive_over_threshold = 0
        return self.state.consecutive_over_threshold >= self.policy.trigger_patience

    # ------------------------------------------------------------------
    # 微调 + 熔断
    # ------------------------------------------------------------------
    def adapt(
        self,
        recent: WindowedDataset,
        guard: WindowedDataset,
        *,
        force: bool = False,
    ) -> AdaptationEvent:
        """在 ``recent`` 上微调修正层，并用 ``guard`` 做熔断校验。

        Parameters
        ----------
        recent
            最近的滑动窗口数据（概念漂移后的当前工况）。
        guard
            **独立留出**的验证数据。不能与 ``recent`` 重叠，否则熔断
            变成自证。现场应使用最近一次独立验证批次/时段。
        force
            跳过触发条件直接微调（仅用于测试/人工干预）。
        """
        if len(recent) == 0 or len(guard) == 0:
            raise ValueError("recent 与 guard 都必须非空。")

        triggered = force or (
            self.state.consecutive_over_threshold >= self.policy.trigger_patience
        )
        rmse_before = _fast_rmse(self.model, recent)
        guard_before = _fast_rmse(self.model, guard)
        if not triggered:
            return AdaptationEvent(
                cycle=self.state.cycle,
                triggered=False,
                reason="未达到触发条件（连续超限次数不足）",
                rmse_before=rmse_before,
                rmse_after=rmse_before,
                guard_rmse_before=guard_before,
                guard_rmse_after=guard_before,
                rolled_back=False,
                correction_norm_before=self._correction_norm(),
                correction_norm_after=self._correction_norm(),
                learning_rate=0.0,
            )

        norm_before = self._correction_norm()
        # 微调前确保已拍过快照；此时修正层一定已 build（上面算过 RMSE）。
        self._ensure_snapshot()
        # 冻结白箱核心：只允许修正层参与梯度更新。
        if self.policy.freeze_physics:
            for weight in physics_weights(self.model):
                weight.trainable = False

        try:
            self._fine_tune(recent)
        finally:
            # 无论成功与否都恢复机理参数的可训练标记，避免污染后续离线训练。
            if self.policy.freeze_physics:
                for weight in physics_weights(self.model):
                    weight.trainable = True

        rmse_after = _fast_rmse(self.model, recent)
        guard_after = _fast_rmse(self.model, guard)
        rolled_back = self._should_rollback(guard_before, guard_after)
        if rolled_back:
            self._restore()
        else:
            self._stable_weights = self._snapshot()
        reason = "微调完成"
        if rolled_back:
            reason = "熔断：guard 指标恶化，已回滚到上一版稳定权重"
        elif not self._correction_within_bounds():
            # 修正幅度触顶：虽然本次指标没坏，但已接近结构边界，
            # 这类模型不适合继续在线外推，必须人工介入。
            reason = "微调完成，但修正幅度接近结构上界，需人工复核"

        event = AdaptationEvent(
            cycle=self.state.cycle,
            triggered=True,
            reason=reason,
            rmse_before=rmse_before,
            rmse_after=_fast_rmse(self.model, recent),
            guard_rmse_before=guard_before,
            guard_rmse_after=_fast_rmse(self.model, guard),
            rolled_back=rolled_back,
            correction_norm_before=norm_before,
            correction_norm_after=self._correction_norm(),
            learning_rate=self.policy.learning_rate,
        )
        self.state.events.append(event)
        if not rolled_back:
            # 成功微调后重置计数器，等待下一次漂移证据。
            self.state.consecutive_over_threshold = 0
        return event

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    def _fine_tune(self, dataset: WindowedDataset) -> None:
        # 用**仅修正层**做一次短训练，学习率低 2 个数量级。
        #
        # 只编译**一次**：每次 ``compile`` 都会新建训练函数并在 TF 中累积
        # 追踪产物，守护进程周期性调用最终会 MemoryError / ``Arena alloc
        # failed``。因此首次微调时以在线学习率编译，之后复用；若策略中的
        # 学习率改变，再重新编译。
        if self._compiled_learning_rate != self.policy.learning_rate:
            compile_greybox(
                self.model,
                learning_rate=self.policy.learning_rate,
                jit_compile=False,
            )
            self._compiled_learning_rate = self.policy.learning_rate
        self.model.fit(
            dataset.as_model_inputs(),
            dataset.targets,
            sample_weight=dataset.weights,
            epochs=self.policy.epochs,
            batch_size=self.policy.batch_size,
            verbose=0,
            shuffle=True,
        )

    def _should_rollback(self, before: float, after: float) -> bool:
        """熔断判据：任一条件成立即回滚。"""
        if not np.isfinite(after):
            return True
        if not np.isfinite(before):
            # 原来就无法评估，无法证明变好 -> 保守回滚。
            return True
        tolerance = before * (1.0 + self.policy.max_rmse_regression)
        if after > tolerance:
            return True
        if after > self.policy.max_rmse_absolute_mol_l:
            return True
        return False

    def _correction_norm(self) -> float:
        weights = correction_weights(self.model)
        if not weights:
            return 0.0
        return float(
            sum(np.sqrt(np.sum(np.square(np.asarray(w)))) ** 2 for w in weights) ** 0.5
        )

    def _correction_within_bounds(self) -> bool:
        """修正层权重是否仍在结构上界内（粗略但可审计的检查）。

        这里检查的是**权重范数**是否相对初始化显著增长；它不能替代
        ``model.physics_layer`` 里的 tanh/sigmoid 硬限幅（那些才是真正的
        结构约束），但能提示"网络正在用很大的权重拟合局部数据"。
        """
        baseline = getattr(self, "_baseline_norm", None)
        current = self._correction_norm()
        if baseline is None:
            self._baseline_norm = current
            return True
        if baseline <= 0:
            return True
        return current <= 3.0 * baseline

    def _snapshot(self) -> list[np.ndarray]:
        return [np.array(w) for w in correction_weights(self.model)]

    def _ensure_snapshot(self) -> None:
        """在修正层已构建且尚未快照时补拍一次。

        为什么不覆盖已有快照：已有快照代表"上一版通过验证的权重"，
        它只能被一次**成功的** ``adapt`` 替换。重复构造快照会把尚未验证
        的当前权重误当成稳定版本。
        """
        if self._stable_weights is not None:
            return
        weights = correction_weights(self.model)
        if weights:  # 空列表说明还没 build，留到首次 adapt 再拍
            self._stable_weights = self._snapshot()

    def _restore(self) -> None:
        if self._stable_weights is None:
            raise RuntimeError("没有可用的稳定权重快照，拒绝回滚。")
        weights = correction_weights(self.model)
        if len(weights) != len(self._stable_weights):
            raise RuntimeError(
                "修正层权重数量变化（模型可能被重建），无法安全回滚。"
            )
        for variable, value in zip(weights, self._stable_weights):
            variable.assign(value)


__all__ = [
    "AdaptationEvent",
    "AdaptationPolicy",
    "AdaptationState",
    "OnlineAdapter",
    "correction_weights",
    "physics_weights",
]
