"""灰箱 CSTR 模型，以及按通道掩码的物理损失（Keras 3）。

为什么不用 ``loss="mse"``
-------------------------
工业数据的两类“空值”含义不同：

- 温度缺失：传感器无效/通讯中断——该时刻不可行监督；
- 浓度缺失：**没有化验结果**——不是“浓度为 0”。

若直接用 MSE，缺失会被当成目标 0，模型会学到“出口浓度趋于 0”这种
危险偏差。本模块用 ``targets`` 旁的 ``weights`` 张量逐点控制：

- 浓度通道：仅在有化验结果（零阶保持+吸附掩码）时权重为 1；
- 温度通道：全程参与。通道内部各自归一化后，再由
  ``ModelConfig.channel_weights`` 决定两通道的相对重要性。

同时，若某窗口完全没有任何浓度监督，它的浓度损失为 0，不应假装贡献
了质量拟合。训练报表会额外输出“有效质量监督点数”，便于判断。
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import keras
from keras import ops

from .config import TARGET_CHANNELS, ModelConfig
from .mechanistic_layer import CSTRGreyBoxLayer

#: 多步滚动惩罚的起算点轮换游标。
#:
#: 为什么需要模块级游标而不是随机数：``call`` 中的时间循环需要 **Python
#: 整数**边界，``keras.random.randint`` 返回符号张量，在 graph mode 下不能
#: 用于 ``range``。轮换候选点既保证每个 batch 看到不同起点（数据增强的
#: 效果），又不破坏图追踪。
_ROLLOUT_CURSOR: list[int] = [0]


class MaskedPhysicsLoss(keras.losses.Loss):
    """加权均方误差；权重来自每个样本的 (T+1, 2) 掩码。

    约定 ``y_true``、``y_pred``、``sample_weight`` 形状均为 ``(batch, T+1, 2)``。
    权重为 0 的点完全不计入分母，避免“缺失越多、损失越小”的假象。

    .. warning::
       **Keras 3 的 ``Loss.__call__`` 不会把 ``sample_weight`` 传给
       ``call``**（它先调用 ``call(y_true, y_pred)``，再把权重交给
       ``reduce_weighted_values`` 做整体缩放）。因此不要把本类直接
       ``compile(loss=...)`` 后指望逐点掩码生效——那样掩码会被静默忽略，
       未监督的浓度点会被当成 0 mol/L 参与训练。

       正确用法见 :meth:`CSTRGreyBoxModel.compute_loss`：在
       ``compute_loss`` 中显式接收 ``sample_weight`` 并调用
       :meth:`masked_squared_error`。本类保留是为了让 ``compile`` 的损失
       配置可读、可序列化，实际计算由模型重写的 ``compute_loss`` 完成。
    """

    def __init__(self, name: str = "masked_physics_loss", **kwargs: Any) -> None:
        super().__init__(name=name, reduction="sum_over_batch_size", **kwargs)

    def call(self, y_true: Any, y_pred: Any, sample_weight: Any = None) -> Any:
        if sample_weight is None:
            sample_weight = ops.ones_like(y_true)
        return masked_squared_error(y_true, y_pred, sample_weight)


def masked_squared_error(
    y_true: Any,
    y_pred: Any,
    weights: Any,
    channel_weights: Any = None,
) -> Any:
    """**按通道归一化**的加权 MSE，并支持显式通道权重。

    工业标签密度极不均衡：浓度只在稀疏化验点有监督（例如 4 h 窗口内
    约 6 个点），温度却每个采样点都有（41 个点）。若把两个通道的点混在
    一起归一化，温度项会以约 ``41/T_scale vs 6`` 的比例压倒浓度项，
    网络会为了拟合温度而牺牲质量软测量的主目标——这正是消融实验里
    “仅速率修正”反而让 ``C_A`` 变差的原因。

    因此本函数分两步：

    1. **通道内归一化**：每个通道的损失 = 该通道的加权平方误差 /
       该通道的权重和。这样通道内部的采样频率、缺失比例都不再影响量级。
    2. **通道间加权**：按 ``channel_weights`` 合并。它表达的是工艺意图
       （浓度是主目标、温度是辅助约束），而不是数据密度的副产品。

    若某通道在本批样本中完全没有监督，它不参与合并（不会把分母拉大、
    稀释其他通道）。

    Parameters
    ----------
    y_true, y_pred, weights
        ``(batch, T+1, C)``。``weights == 0`` 表示该点不可监督；应作为
        纯掩码使用（0/1），通道重要性由 ``channel_weights`` 表达。
    channel_weights
        长度 ``C`` 的非负序列，默认全 1。设 ``[1.0, 0.25]`` 表示温度
        误差在合并时按 1/4 权重计。

    Notes
    -----
    用 ``ops.where`` 屏蔽无监督点的误差，避免 ``0 * NaN = NaN`` 污染梯度；
    占位目标值（无监督处填 0）因此不会被“学成 0 浓度”。
    """
    weights = ops.cast(weights, y_pred.dtype)
    squared = ops.square(y_pred - y_true)
    supervised = ops.cast(weights > 0, y_pred.dtype)
    squared = ops.where(supervised > 0, squared * weights, ops.zeros_like(squared))

    # 逐通道归一化：shape 保持 (batch, C)。
    squared_sum = ops.sum(squared, axis=1)
    weight_sum = ops.sum(weights, axis=1)
    has_supervision = ops.cast(weight_sum > 0, weights.dtype)
    channel_loss = ops.divide(squared_sum, ops.maximum(weight_sum, 1.0))

    if channel_weights is None:
        alpha = ops.ones_like(channel_loss)
    else:
        alpha = ops.cast(channel_weights, channel_loss.dtype)
        alpha = ops.broadcast_to(alpha, ops.shape(channel_loss))
    effective = alpha * has_supervision
    per_sample = ops.divide(
        ops.sum(channel_loss * effective, axis=1),
        ops.maximum(ops.sum(effective, axis=1), 1.0),
    )
    return ops.mean(per_sample)


def _static_shape(tensor: Any) -> list[int] | None:
    """返回完全已知的静态形状；含 ``None``（动态 batch）时返回 ``None``。

    Keras 3 的 ``Tensor.shape`` 是 ``TensorShape``，元素可能是 ``None``
    （动态维度）。只有所有维度都已知时才能做严格的逐轴比较。
    """
    shape = getattr(tensor, "shape", None)
    if shape is None:
        return None
    dimensions = list(shape)
    if any(dimension is None for dimension in dimensions):
        return None
    return [int(dimension) for dimension in dimensions]


class CSTRGreyBoxModel(keras.Model):
    """完整的灰箱 CSTR 模型。

    输入 ``(initial_states, controls, slow)``；输出为未缩放/已缩放的混合
    轨迹 ``(batch, T+1, 2)``：通道 0 为 ``C_A`` (mol/L)，通道 1 为
    ``(T - T_ref)/scale``。该设计与教程 05 的 ``scale_states`` 一致，
    因此标签与预测可直接比较。

    为什么重写 ``compute_loss``
    ---------------------------
    Keras 3 的 ``Loss.__call__`` 只把 ``y_true``/``y_pred`` 传给子类的
    ``call``，``sample_weight`` 被用于**整体**缩放而不是逐点掩码。工业
    数据里“没有化验”和“浓度为 0”必须区分，因此本模型在
    ``compute_loss`` 中直接使用逐点权重，并显式加上层的修正幅度惩罚。
    """

    def __init__(
        self,
        config: ModelConfig,
        channel_weights: tuple[float, ...] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.config = config
        self.physics_layer = CSTRGreyBoxLayer(config, name="cstr_physics")
        #: 损失中的通道权重 ``[C_A, T]``。默认 ``[1, 1]``：通道内部已各自
        #: 归一化，这里只表达“谁更重要”。质量软测量建议 ``[1.0, 0.25]``。
        self.channel_weights = (
            tuple(float(value) for value in channel_weights)
            if channel_weights is not None
            else tuple(float(value) for value in config.channel_weights)
        )
        if len(self.channel_weights) != len(TARGET_CHANNELS):
            raise ValueError(
                f"channel_weights 需要 {len(TARGET_CHANNELS)} 个元素（[C_A, T]）。"
            )
        if any(value < 0 for value in self.channel_weights):
            raise ValueError("channel_weights 不能为负。")
        if sum(self.channel_weights) <= 0:
            raise ValueError("channel_weights 之和必须为正。")

    def call(self, inputs: Any, training: bool | None = None) -> Any:
        self._last_inputs = inputs
        output = self.physics_layer(inputs, training=training)
        if training and self.config.long_horizon.rollout_penalty:
            self.add_loss(self._rollout_penalty(inputs, output))
        return output

    def _rollout_penalty(self, inputs: Any, output: Any) -> Any:
        """扰动重锚后的滚动一致性惩罚（长期外推正则化）。

        做法：在窗口中途取模型轨迹的状态，**加一个小扰动**作为"被化验
        重锚修正过"的起点，再向前滚动若干步，要求它回到原轨迹。

        为什么必须加扰动
        ----------------
        不加扰动时，确定性积分器从精确参考状态重新起算必然精确复现原轨迹
        （半群性质自动成立），惩罚恒为 0——它什么也没约束。而部署时状态
        总是带着重锚偏差，所以有意义的检验是"扰动能否被吸回"。

        扰动幅度按通道分别设定：浓度用 ``rollout_anchor_noise_c_a``，
        温度用 ``rollout_anchor_noise_temperature``（缩放坐标）。它们应
        接近现场重锚的实际量级——太大变成"任意初值都能收敛"的过强约束，
        太小则没有梯度信号。
        """
        cfg = self.config.long_horizon
        n_steps = int(output.shape[1]) - 1
        low = max(int(round(cfg.rollout_min_fraction * n_steps)), 1)
        high = max(int(round(cfg.rollout_max_fraction * n_steps)), low)
        starts = list(range(low, high + 1))
        total = ops.zeros((), dtype=output.dtype)
        samples = max(1, cfg.rollout_samples)
        # 扰动是绝对量（mol/L），需转成训练目标同尺度才能与数据项比较：
        # 浓度通道用原单位，温度通道按 temperature_scale_k 缩放。
        scale = ops.convert_to_tensor(
            [cfg.rollout_anchor_noise_c_a, cfg.rollout_anchor_noise_temperature],
            dtype=output.dtype,
        )
        # 注：``rollout_consistency`` 在缩放坐标下工作，但浓度通道本就
        # 是原单位（见 mechanistic_layer 的输出约定），温度通道的缩放
        # 由 scale 的第二项体现。
        for offset in range(samples):
            index = (_ROLLOUT_CURSOR[0] + offset) % len(starts)
            start = starts[index]
            # 交替正负扰动，避免系统性偏置（不需要随机数即可覆盖两个方向）。
            sign = 1.0 if (index + offset) % 2 == 0 else -1.0
            # 形状 (1, 2)：加到 reference[:, start, :] 上广播到 (batch, 2)。
            perturbation = ops.reshape(scale, (1, 2)) * sign
            deviation = self.physics_layer.rollout_consistency(
                inputs[0],
                inputs[1],
                inputs[2],
                output,
                start=start,
                max_steps=cfg.rollout_max_steps,
                anchor_perturbation=perturbation,
            )
            total = total + ops.mean(deviation)
        _ROLLOUT_CURSOR[0] += samples
        penalty = total / ops.cast(samples, output.dtype)
        # **相对权重归一化**：``rollout_penalty_weight`` 表达"惩罚应占数据
        # 拟合项的多大比例"，而不是一个绝对系数。
        #
        # 为什么不直接用绝对系数：惩罚量级取决于扰动幅度的平方与窗口长度，
        # 实测在默认扰动 0.01 mol/L 下仅约 5e-6，而数据损失约 0.07；
        # 若权重是绝对数，用户需要猜一个 1e3 量级的魔法数，极易配错
        # （本仓库初版就把权重设成 0.1，导致惩罚完全不起作用）。
        #
        # 这里用**当前批次的滚动偏差尺度**归一化：除以 ``penalty`` 自身
        # 会退化为常数 1（无信息），因此除以"同量纲的参照量"——
        # 用轨迹长度与扰动幅度构造的期望偏差量级：
        #     ref = anchor_noise² · steps
        # 这样权重 0.1 表示"惩罚约为主损失量级的 10%×相对偏差"，
        # 与数据尺度无关，跨装置可直接迁移。
        # 归一化：除以"同量纲的参照量"。这里用**本批轨迹自身的典型偏离
        # 尺度**（相邻步变化的均方），它天然与浓度单位同量纲、且随工况
        # 自适应。因此权重表达"惩罚相对于轨迹自身的动态幅度"，跨装置可迁移。
        if cfg.rollout_penalty_normalize:
            reference = ops.mean(
                ops.square(output[:, 1:, :] - output[:, :-1, :])
            )
            penalty = penalty / ops.maximum(reference, ops.cast(1e-12, output.dtype))
        return ops.cast(cfg.rollout_penalty_weight, output.dtype) * penalty

    def compute_loss(
        self,
        x: Any = None,
        y: Any = None,
        y_pred: Any = None,
        sample_weight: Any = None,
        training: bool = True,
    ) -> Any:
        """逐点掩码损失 + 结构正则项。

        ``sample_weight`` 的形状必须与 ``y_pred`` 一致：``(batch, T+1, 2)``。
        Keras 只有在 ``fit`` 传入的权重与输出同形时才会原样透传；若形状
        不匹配，应在上游广播好，而不是在这里猜。
        """
        del x, training
        if y_pred is None or y is None:
            raise ValueError("compute_loss 需要 y 与 y_pred。")
        if sample_weight is None:
            sample_weight = ops.ones_like(y)
        weights = ops.cast(sample_weight, y_pred.dtype)
        # 用**静态**形状做检查：``ops.shape`` 在 graph mode 下返回符号张量，
        # 直接比较会触发 OperatorNotAllowedInGraphError。只要两侧形状都完全
        # 已知，就要求**逐轴相等**（包括秩）——权重必须是逐点掩码，不能是
        # 逐样本或广播形状，否则会出现“看起来训练了，其实掩码错位”的问题。
        weight_shape = _static_shape(weights)
        pred_shape = _static_shape(y_pred)
        if (
            weight_shape is not None
            and pred_shape is not None
            and weight_shape != pred_shape
        ):
            raise ValueError(
                "sample_weight 必须与预测同形 (batch, T+1, C)，"
                f"实际 {weight_shape} vs {pred_shape}。"
            )
        total = masked_squared_error(
            y, y_pred, weights, channel_weights=self.channel_weights
        )
        # 层的 add_loss（修正幅度惩罚）需要显式汇总，因为本方法覆盖了
        # 默认实现。
        for extra in self.losses:
            total = total + self._aggregate_additional_loss(extra)
        return total

    # ------------------------------------------------------------------
    # 可读参数
    # ------------------------------------------------------------------
    def fitted_parameters(self) -> dict[str, float]:
        """返回已学习的机理参数（反变换回物理单位）。"""
        layer = self.physics_layer
        params = {
            "k_ref_per_min": float(ops.convert_to_numpy(ops.exp(layer.log_k_ref))),
            "ua_per_min": float(ops.convert_to_numpy(ops.exp(layer.log_ua))),
        }
        if layer.log_heat_release is not None:
            params["heat_release_k_l_mol"] = float(
                ops.convert_to_numpy(ops.exp(layer.log_heat_release))
            )
        return params

    def get_config(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "channel_weights": list(self.channel_weights),
            **super().get_config(),
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "CSTRGreyBoxModel":
        from .config import config_from_dict

        model_config = dict(config)
        model_config["config"] = config_from_dict(model_config["config"])
        weights = model_config.get("channel_weights")
        if weights is not None:
            model_config["channel_weights"] = tuple(weights)
        return cls(**model_config)


def compile_greybox(
    model: CSTRGreyBoxModel,
    *,
    learning_rate: float = 5e-3,
    clipnorm: float = 1.0,
    jit_compile: bool | str = "auto",
) -> keras.Model:
    """按工业训练惯例编译：掩码损失 + 梯度范数裁剪。

    ``clipnorm`` 是最后一道防线：即使某个批次出现极端残差，参数更新也
    不会被放大到破坏机理参数（例如把 ``log_k_ref`` 推到一个物理上不可能
    的值）。裁剪后仍需在报表里检查 ``gradient_norm``。

    ``jit_compile`` 默认交给后端决定。**周期性重新编译**的场景（在线
    微调守护进程）应显式传 ``False``：每次编译都会生成新的训练函数，
    长时间运行会累积编译产物，最终可能出现 ``MemoryError``。
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=clipnorm),
        loss=MaskedPhysicsLoss(),
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
        jit_compile=jit_compile,
    )
    return model


def predict_trajectory(
    model: keras.Model, dataset: Any, *, batch_size: int = 64
) -> Any:
    """对 ``WindowedDataset`` 做批量前向，返回 numpy ``(N, T+1, 2)``。"""
    import numpy as np

    predictions = []
    total = len(dataset)
    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        inputs = (
            dataset.initial_states[start:stop],
            dataset.controls[start:stop],
            dataset.slow[start:stop],
        )
        predictions.append(np.asarray(model(inputs, training=False)))
    return np.concatenate(predictions, axis=0)


__all__ = [
    "CSTRGreyBoxModel",
    "MaskedPhysicsLoss",
    "compile_greybox",
    "predict_trajectory",
]
