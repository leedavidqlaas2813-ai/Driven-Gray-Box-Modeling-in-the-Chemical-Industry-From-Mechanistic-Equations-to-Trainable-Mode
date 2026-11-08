"""灰箱 CSTR 的可微机理层（Keras 3）。

结构
----
::

    操作变量 + 慢特征
          │
          ├── 速率修正网络  →  delta_rate   (|δ| ≤ max_log_correction)
          └── 结垢/活性网络 →  delta_ua, delta_activity (主趋势 ≤ 0)
                      │
    名义机理参数 ──► k_eff  = k_ref exp(Ea 项 + δ_rate + δ_activity)
                     UA_eff = UA_0  exp(δ_ua)
                      │
                      ▼
        RK4 / IMEX / Euler 积分 ──► [C_A, T] 轨迹（输出坐标见下）

输出坐标
--------
返回 ``(batch, T+1, 2)``：

- 通道 0：``C_A``，单位 mol/L；
- 通道 1：``(T - T_ref) / temperature_scale_k``，无量纲。

这样与教程 05 的 ``scale_states`` 保持一致：浓度按原单位进入损失（mol/L），
温度按 20 K 尺度归一化，避免温度数值量级（~350）压过浓度（~1）。

物理约定（必须与数据字典一致）
------------------------------
- 状态列顺序 ``[C_A (mol/L), T (K)]``；控制列顺序见 ``config.CONTROL_COLUMNS``。
- 反应放热为正：``heat_release_k_l_mol = (-ΔH_r)/(ρ Cp) > 0`` 时温度上升；
  吸热反应该值为负，方程形式不变。
- 传热项 ``UA_eff * (T_c - T)`` 假定夹套为理想恒温边界。真实夹套有
  冷却水流量、壁热惯性和滞后，应扩展为夹套能量衡算状态，不能指望
  速率修正网络去“学”热惯性。
- 一级反应、恒容、完全混合是**结构假设**，网络不能修正反应级数或
  体积变化；这些需要改机理结构并重新做可辨识性分析。
- 浓度守护上界是数值护栏，不是工艺安全限值。

可训练参数
----------
- ``log_k_ref``：参考速率常数（对数参数化保证为正）；
- ``log_ua``：传热系数基准（对数参数化保证为正）；
- 可选 ``log_heat_release``：仅当 ``learn_heat_release=True``。默认关闭，
  因为稳态数据上放热与传热常不可分（UA 增大与反应热减小可给出相同温度）。
"""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any

import keras
from keras import layers, ops

from .config import ModelConfig
from .numerics import guard_state, integrate_interval


class CSTRGreyBoxLayer(layers.Layer):
    """可微 CSTR 机理层（含速率修正与慢变漂移修正）。

    输入（均在 ``call`` 的 tuple 中，按顺序）：

    - ``initial_state``: ``(batch, 2)``，``[C_A mol/L, T K]``；
    - ``controls``: ``(batch, T, 4)``，列顺序
      ``[C_A0, T_in, T_c, tau]``，单位见 ``config.CONTROL_COLUMNS``；
    - ``slow``: ``(batch, T, 3)`` 或 ``(batch, 3)``，已缩放到 O(1) 的
      慢特征（见 ``slow_features.normalize_slow_features``）。
    """

    def __init__(self, config: ModelConfig, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.config = config
        self.physics = config.physics
        self.integrator = config.integrator
        self.scaling = config.scaling

        self.log_k_ref = self.add_weight(
            name="log_k_ref",
            shape=(),
            initializer=keras.initializers.Constant(
                math.log(self.physics.nominal_k_ref_per_min)
            ),
            trainable=True,
        )
        self.log_ua = self.add_weight(
            name="log_ua",
            shape=(),
            initializer=keras.initializers.Constant(
                math.log(max(self.physics.nominal_ua_per_min, 1e-6))
            ),
            trainable=True,
        )
        self.log_heat_release = (
            self.add_weight(
                name="log_heat_release",
                shape=(),
                initializer=keras.initializers.Constant(
                    math.log(abs(self.physics.heat_release_k_l_mol))
                ),
                trainable=True,
            )
            if config.learn_heat_release
            else None
        )

        self.rate_net = (
            keras.Sequential(
                [
                    layers.Dense(config.rate_correction.hidden_units, activation="tanh"),
                    layers.Dense(
                        1, kernel_initializer="zeros", bias_initializer="zeros"
                    ),
                ],
                name="rate_correction",
            )
            if config.rate_correction.hidden_units > 0
            else None
        )
        self.slow_net = (
            keras.Sequential(
                [
                    layers.Dense(
                        config.slow_drift.residual_hidden_units, activation="tanh"
                    ),
                    layers.Dense(
                        5,  # δ_ua主趋势, δ_activity主趋势, δ_ua残余, δ_activity残余, 未用
                        kernel_initializer="zeros",
                        bias_initializer=keras.initializers.Constant(
                            [
                                config.slow_drift.initial_decay_bias,
                                config.slow_drift.initial_decay_bias,
                                0.0,
                                0.0,
                                0.0,
                            ]
                        ),
                    ),
                ],
                name="slow_drift",
            )
            if config.enable_slow_drift
            else None
        )

        # 数值诊断：每个 epoch/批次可观察到是否发生过回退或裁剪。
        # 注意这是“指数”而非工艺报警；非零值先回查积分与模型，再决定上线。
        self.nonfinite_steps = keras.metrics.Mean(name="nonfinite_steps")
        self.clipped_steps = keras.metrics.Mean(name="clipped_steps")

    # ------------------------------------------------------------------
    # 物理参数
    # ------------------------------------------------------------------
    @property
    def activation_energy_over_r(self) -> float:
        return self.physics.activation_energy_j_mol / self.physics.gas_constant_j_mol_k

    def arrhenius_rate(self, temperature_k: Any) -> Any:
        """名义 Arrhenius 速率常数 [min⁻¹]，始终为正。

        指数先裁剪到 ±18：对应 ``k_ref * e^{-18} .. k_ref * e^{18}``，
        既覆盖工艺温区，又避免 float32 溢出导致 NaN 污染梯度。
        """
        exponent = self.activation_energy_over_r * (
            ops.divide(1.0, temperature_k) - 1.0 / self.physics.t_ref_k
        )
        return ops.exp(self.log_k_ref - ops.clip(exponent, -18.0, 18.0))

    # ------------------------------------------------------------------
    # 数据驱动修正
    # ------------------------------------------------------------------
    def _rate_features(self, c_a: Any, temperature_k: Any, controls: Any, step: int) -> Any:
        """构造速率修正网络的输入（同时用于域外门控的距离计算）。

        这 6 个特征既喂给网络，也用于判断"当前工况离训练域有多远"。
        每个特征都写成 ``(当前值 - 参考值) / 尺度``，即"偏离了参考值几个
        尺度单位"。这样做的两个理由：

        1. **网络好训练**：输入在 0 附近、量级 O(1)，梯度不会爆炸或消失。
           若直接喂 350 K 这样的原始值，与其他 O(1) 特征混合会让训练
           极不稳定。
        2. **门控可解释**：``|z| = 2`` 直白地表示"偏离了 2 个尺度"。
           以温度为例，尺度是 20 K，所以 ``|z| = 2`` 就是偏高/偏低 40 K。

        具体换算（以温度为例）::

            350 K → (350-350)/20 = 0      （参考点）
            370 K → (370-350)/20 = +1.0   （偏高 1 个尺度 = 20 K）
            390 K → (390-350)/20 = +2.0   （偏高 2 个尺度 = 40 K）

        **浓度特征为什么必须居中**（本项目实际踩过的坑）
        -----------------------------------------------
        初版写成 ``c_a / c_a_ref``，没有减参考值。当 c_a 在 0.8–1.2 之间
        波动时，这个特征的值在 0.8–1.2 之间——**在域内就已经接近 1**。

        而门控用 ``max|z|`` 判断距离，于是浓度这一维始终贡献 ~1，把
        温度的偏移完全掩盖：温度高 30 K 只贡献 1.5，看起来"和域内的
        浓度一样"。结果门控在真正外推时不触发。

        改成 ``(c_a - c_a_ref)/c_a_ref`` 后，域内浓度特征接近 0，温度的
        外推才成为 ``max`` 的主导项。
        """
        scaling = self.scaling
        c_a0 = controls[:, step, 0:1]
        t_in = controls[:, step, 1:2]
        t_c = controls[:, step, 2:3]
        tau = controls[:, step, 3:4]
        return ops.concatenate(
            [
                # 浓度：相对参考值居中（理由见上面文档）。尺度用参考值
                # 本身，所以这是一个"相对偏离比例"——例如 -0.1 表示比
                # 参考浓度低 10%。
                (c_a - scaling.c_a_ref_mol_l) / scaling.c_a_ref_mol_l,
                # 釜温：以 350 K 为参考、20 K 为尺度。
                (temperature_k - scaling.temperature_ref_k) / scaling.temperature_scale_k,
                # 进料浓度：以 1.1 mol/L 为参考、0.3 mol/L 为尺度。
                (c_a0 - scaling.c_a0_ref_mol_l) / scaling.c_a0_scale_mol_l,
                # 进料温度：与釜温共用同一温标基准。
                (t_in - scaling.temperature_ref_k) / scaling.temperature_scale_k,
                # 冷却介质温度：以 300 K 为参考、20 K 为尺度。
                (t_c - scaling.coolant_ref_k) / scaling.coolant_scale_k,
                # 停留时间：以 5.5 min 为参考、2.5 min 为尺度。
                (tau - scaling.tau_ref_min) / scaling.tau_scale_min,
            ],
            axis=-1,
        )

    def rate_correction(self, features: Any) -> Any:
        """有界对数速率修正，``|δ_rate| ≤ max_log_correction``。

        域外门控在此处施加：``δ_rate`` 是符号自由的修正，训练域外的取值
        没有数据支持，因此随距离平滑衰减到 0（退回名义 Arrhenius 速率）。
        注意门控作用于**修正项**而不是整个速率：机理部分永远保留。
        """
        if self.rate_net is None:
            return ops.zeros((ops.shape(features)[0], 1), dtype=features.dtype)
        raw = self.rate_net(features)
        limit = self.config.rate_correction.max_log_correction
        return limit * ops.tanh(raw) * self.domain_gate(features)

    def slow_domain_gate(self, step_slow: Any) -> Any:
        """慢特征的域外门控（以参考尺度为单位的外推倍率）。

        慢特征经 ``x/(x+ref)`` 有界缩放后落在 [0, 1)，因此可反解出外推倍率
        ``x/ref = z/(1-z)``。例如 ``z = 0.9`` 表示已运行到参考寿命的 9 倍。
        超过 ``domain_box`` 倍参考尺度即认为远离训练分布。

        为什么用这个量而不是 ``z`` 本身：``z`` 是饱和的，90 天与 900 天在
        ``z`` 上只差 0.05，无法区分"刚过参考点"与"远超参考点"；而
        ``z/(1-z)`` 保持了对真实运行龄的敏感性。
        """
        cfg = self.config.long_horizon
        if not cfg.domain_gate:
            return ops.ones((ops.shape(step_slow)[0], 1), dtype=step_slow.dtype)
        z = ops.clip(step_slow[:, 0:2], 0.0, 1.0 - 1e-4)
        ratio = z / (1.0 - z)  # age/age_ref, feed/feed_ref
        distance = ops.max(ratio, axis=-1, keepdims=True)
        x = ops.clip((distance - cfg.domain_box) / cfg.domain_taper, 0.0, 1.0)
        return 1.0 - x * x * (3.0 - 2.0 * x)

    def slow_correction(self, slow: Any, step: int) -> tuple[Any, Any]:
        """返回 ``(delta_ua, delta_activity)``；主趋势单项 ≤ 0。

        物理先验：结垢使传热下降，老化使活性下降。网络输出经 sigmoid 后
        取负，再各自叠加一个很小的符号自由的残余项（清洗、换催化剂、
        切换原料等非单调事件）。把“单调不增”写进结构，比让网络从数据里
        “学到”更可靠，也能防止在训练范围外外推出“结垢增强传热”。

        两个通道的残余项**必须分开**：结垢影响传热面，老化影响催化剂，
        共用一个残余项会让清洗事件同时被解释成“活性恢复”，在物理上
        错误，也削弱了可解释性。

        初始化：输出层前两个偏置为 ``initial_decay_bias``（默认 -6），使
        ``sigmoid(-6) ≈ 0.0025``，即训练起点 ≈ 名义机理模型。这一点
        **必须**保持：若从默认 0 偏置出发，模型开头就已假设“UA 下降了
        26%”，等于悄悄改变了名义参数。

        域外行为：只对**符号自由的残余项**做渐近回退，单调主趋势**不做**
        回退。理由是两者物理含义不同：

        - 残余项表示清洗/换剂等事件，域外没有数据支持，回退到 0 是安全的
          （等价于"没有发生这类事件"）；
        - 单调趋势表示结垢/老化，域外（运行更久）它**本来就应该更大**。
          若把它在域外回退到 0，等于假设老装置换热面自动变干净，会系统性
          高估 UA、低估出口浓度——这是安全上错误的方向。

        主趋势本身已被 ``max_decay_log`` 与 sigmoid 双重限幅，不存在无界
        外推；因此不需要（也不应该）门控。
        """
        if self.slow_net is None:
            zeros = ops.zeros((ops.shape(slow)[0], 1), dtype=slow.dtype)
            return zeros, zeros
        step_slow = slow[:, step, :] if len(ops.shape(slow)) == 3 else slow
        raw = self.slow_net(step_slow)
        drift = self.config.slow_drift

        delta_ua_main = -drift.max_decay_log * ops.sigmoid(raw[:, 0:1])
        if drift.enable_activity:
            delta_activity_main = -drift.max_decay_log * ops.sigmoid(raw[:, 1:2])
        else:
            delta_activity_main = ops.zeros_like(delta_ua_main)
        # 残余项受域外门控；主趋势不受。
        #
        # 用一个具体数字理解"为什么主趋势不能门控"：
        # 假设装置已运行 120 天，从数据学到 UA 已衰减 30%（δ_ua = ln 0.7
        # ≈ -0.357）。现在做一次 150 天的外推预测：
        #
        #   若主趋势也被门控回退到 0：
        #     UA_eff = UA_0 × exp(0) = UA_0     ← 假设换热面"自动变干净"
        #     冷却能力被高估约 43%，釜温被低估，出口浓度被低估
        #
        #   正确行为（主趋势饱和保持）：
        #     UA_eff ≈ UA_0 × 0.7              ← 至少不往危险方向错
        #
        # 结垢是**不可逆、随时间单调**的物理过程。域外（运行更久）修正
        # 应该"更大或至少不减"，绝不是"归零"。且主趋势已被 sigmoid 与
        # max_decay_log 双重限幅（上界 0.6，即最多降 45%），不存在无界
        # 外推风险，因此不需要门控。
        #
        # 相反，残余项表示"清洗/换催化剂"这类离散事件。域外没有数据
        # 支持"发生过清洗"，回退到 0 是保守且合理的（等价于"未发生"）。
        gate = self.slow_domain_gate(step_slow)
        residual_ua = drift.max_residual_log * ops.tanh(raw[:, 2:3]) * gate
        residual_activity = (
            drift.max_residual_log * ops.tanh(raw[:, 3:4]) * gate
            if drift.enable_activity
            else ops.zeros_like(residual_ua)
        )
        return delta_ua_main + residual_ua, delta_activity_main + residual_activity

    def domain_gate(self, features: Any) -> Any:
        """域外渐近门控：远离训练分布时把**符号自由**的修正平滑压向 0。

        为什么用平滑门控而不是硬阈值
        ----------------------------
        硬阈值会让修正量在跨越阈值时阶跃，积分器在此处可能产生抖动，而且
        跨越点附近的模型行为不可导，训练时梯度不连续。用 ``smoothstep``
        在 ``box`` 与 ``box+taper`` 之间平滑过渡，既保证域内有数据支持时
        修正完整生效，也保证域外退回机理。

        距离的定义
        ----------
        用 ``feed-forward`` 特征（``_rate_features``）的**盒距离**：
        ``d = max_i |z_i|``，其中 ``z_i`` 是已归一化到 O(1) 的特征。取
        ``max`` 而非欧氏范数，是因为盒形训练域更贴近工艺实际的"每列都有
        上下限"；欧氏距离会让某一维的极端值被其他维稀释。

        返回
        ----
        ``(batch, 1)``，取值 1（域内，修正完全生效）→ 0（远域外，修正关闭）。
        """
        cfg = self.config.long_horizon
        if not cfg.domain_gate:
            return ops.ones((ops.shape(features)[0], 1), dtype=features.dtype)
        distance = ops.max(ops.abs(features), axis=-1, keepdims=True)
        # smoothstep: 在 box 处为 1，在 box+taper 处为 0。
        x = ops.clip((distance - cfg.domain_box) / cfg.domain_taper, 0.0, 1.0)
        return 1.0 - x * x * (3.0 - 2.0 * x)
    def build(self, input_shape: Any) -> None:
        """显式构建子网络。

        机理层的 ``call`` 里用 Python 循环逐步展开时间，Keras 的自动
        形状推断无法在构建阶段追踪到子网络，因此这里显式给出它们的输入
        维度，否则权重会在第一次 ``fit`` 时才创建，导致保存/加载与
        ``model.summary()`` 不一致。
        """
        # 输入第三个元素是慢特征；形状可能为 (batch, 3) 或 (batch, T, 3)。
        slow_shape = input_shape[2] if isinstance(input_shape, (tuple, list)) else None
        if self.rate_net is not None:
            # 速率修正网络输入 6 维：[C_A, T, C_A0, T_in, T_c, tau] 的缩放值。
            self.rate_net.build((None, 6))
        if self.slow_net is not None and slow_shape is not None:
            feature_dim = int(slow_shape[-1])
            self.slow_net.build((None, feature_dim))
        super().build(input_shape)

    # ------------------------------------------------------------------
    # 多步滚动一致性（长期外推正则化）
    # ------------------------------------------------------------------
    def rollout_consistency(
        self,
        initial_state: Any,
        controls: Any,
        slow: Any,
        reference: Any,
        *,
        start: int,
        max_steps: int | None = None,
        anchor_perturbation: Any = None,
    ) -> Any:
        """从 ``start`` 处的**被扰动**锚点重新起算，检验模型能否回到原轨迹。

        先用一个具体场景说明"什么叫不一致"
        ----------------------------------
        假设模型有两条轨迹：A 从真实初始状态出发，B 从 A 在 t=10 处的状态
        出发。对**正确的**微分方程，B 应当与 A 的后半段完全重合（这正是
        微分方程解的唯一性）。这个性质叫半群性质::

            Φ(t2, t0) = Φ(t2, t1) ∘ Φ(t1, t0)
            （"从 t0 积到 t2" = "先积到 t1，再从 t1 积到 t2"）

        **陷阱**：如果 B 的起点完全取自 A 的精确状态，那么对任何确定性
        积分器，重合是**自动成立**的——不管模型对不对。此时惩罚恒为 0，
        什么也没约束。本项目第一版就是这样，消融实验里"开/关惩罚"两组
        数字逐位相同，才发现这个错误。

        为什么真实部署中会不一致
        ------------------------
        影子运行里，状态**不是**从模型自己的轨迹出发的：每收到一条化验
        结果，状态就被**重锚**（加一个偏差修正）。也就是说，部署时的
        每一步都是"从被扰动过的状态重新起算"。

        于是真正有意义的检验是：**给定一个被扰动的小偏差，模型的动力学
        会把它吸回原轨迹，还是放大它？**

        - 吸回 → 说明修正项是"状态函数"，扰动会被物理规律拉回，滚动稳定；
        - 放大 → 说明模型在某些状态下不自洽，长滚动时小误差会不断累积。

        惩罚前者、抑制后者，就是本函数的目的。它不需要额外数据。

        ``anchor_perturbation`` 的形状
        ------------------------------
        ``(batch, 2)`` 或可广播到 ``(batch, 2)`` 的形状。传 ``None`` 时惩罚
        恒为 0（仅用于回归测试，验证半群性质本身成立）。

        ``max_steps`` 截断
        ------------------
        惩罚积分**嵌套**在正向积分里，若从中途一直滚到窗口末尾，计算图
        大小近似翻倍（每步还含 RK4 的 4 个 stage × 子步）。40+ 步的窗口
        在内存受限的机器上会 ``MemoryError``——这也是本项目实际踩过的坑。

        截断到 ``k`` 步仍然有效：从中间点滚 15 步已经足够判断"扰动是被
        吸回还是被放大"，完全滚到底的额外约束有限，而内存代价线性增长。

        Returns
        -------
        ``(batch,)`` 每个样本的平均平方偏差（缩放坐标，即浓度用 mol/L、
        温度用 (T-350)/20）。
        """
        # 取轨迹在 start 处的状态作为"真实起点"，再叠加扰动。
        anchor = reference[:, start, :]
        if anchor_perturbation is not None:
            anchor = anchor + anchor_perturbation
        state = anchor
        steps = int(controls.shape[1])
        stop = steps if max_steps is None else min(start + max_steps, steps)
        squared_total = ops.zeros((), dtype=state.dtype)
        count = 0
        for step in range(start, stop):
            # 从扰动后的状态出发，一步步积回去。
            state, _, _, _ = self._advance(state, controls, slow, step)
            # 与"没有扰动时应有的轨迹"逐点比较，累计平方偏差。
            deviation = ops.square(state - reference[:, step + 1, :])
            squared_total = squared_total + ops.sum(deviation, axis=-1)
            count += 1
        return squared_total / ops.cast(max(count, 1), state.dtype)

    # ------------------------------------------------------------------
    # 前向积分
    # ------------------------------------------------------------------
    def call(self, inputs: Any, training: bool | None = None) -> Any:
        initial_state, controls, slow = inputs
        initial_state = ops.cast(initial_state, controls.dtype)
        slow = ops.cast(slow, controls.dtype)

        scale = self.physics.temperature_scale_k
        t_ref = self.physics.t_ref_k
        state = ops.concatenate(
            [initial_state[:, 0:1], (initial_state[:, 1:2] - t_ref) / scale], axis=-1
        )
        trajectory = [state]

        steps = int(controls.shape[1])
        nan_total = ops.zeros((), dtype="float32")
        clip_total = ops.zeros((), dtype="float32")
        penalty_total = ops.zeros((), dtype="float32")
        for step in range(steps):
            state, step_nan, step_clip, step_penalty = self._advance(
                state, controls, slow, step
            )
            trajectory.append(state)
            nan_total = nan_total + ops.cast(ops.sum(step_nan), "float32")
            clip_total = clip_total + ops.cast(ops.sum(step_clip), "float32")
            penalty_total = penalty_total + step_penalty

        if training:
            # 只在训练时更新诊断指标，推理不产生额外开销。
            self.nonfinite_steps.update_state(nan_total)
            self.clipped_steps.update_state(clip_total)
        self._diagnostics = (nan_total, clip_total)

        # 修正幅度惩罚：鼓励模型优先用机理，除非数据强烈要求修正。
        # 这是一项结构先验，不是精度技巧：修正越大约束越弱、外推越危险。
        if self.config.rate_correction.l2_penalty > 0:
            self.add_loss(
                ops.cast(
                    self.config.rate_correction.l2_penalty
                    * penalty_total
                    / ops.cast(max(steps, 1), "float32"),
                    "float32",
                )
            )
        return ops.stack(trajectory, axis=1)

    def _advance(
        self,
        state_scaled: Any,
        controls: Any,
        slow: Any,
        step: int,
        *,
        use_corrections: bool = True,
    ) -> tuple[Any, Any, Any, Any]:
        """积分一个采样间隔，并应用数值守护。

        ``use_corrections=False`` 只使用已拟合的机理参数（无网络修正），
        用于影子运行中同时产生“机理基线”和“灰箱预测”。

        Returns
        -------
        state_next, nonfinite_mask, clipped_mask, correction_penalty
        """
        scale = self.physics.temperature_scale_k
        t_ref = self.physics.t_ref_k

        c_a0 = controls[:, step, 0:1]
        t_in = controls[:, step, 1:2]
        t_c = controls[:, step, 2:3]
        # 物理上 tau 必须为正；这里的下限只防止坏数据造成除零。
        tau = ops.maximum(controls[:, step, 3:4], 1e-3)

        temperature_k = state_scaled[:, 1:2] * scale + t_ref
        k_nominal = self.arrhenius_rate(temperature_k)
        if use_corrections:
            rate_features = self._rate_features(
                state_scaled[:, 0:1], temperature_k, controls, step
            )
            delta_rate = self.rate_correction(rate_features)
            delta_ua, delta_activity = self.slow_correction(slow, step)
        else:
            delta_rate = ops.zeros_like(k_nominal)
            delta_ua = ops.zeros_like(k_nominal)
            delta_activity = ops.zeros_like(k_nominal)
        penalty = (ops.mean(ops.square(delta_rate)) + ops.mean(ops.square(delta_ua)))
        k_effective = k_nominal * ops.exp(delta_rate + delta_activity)
        ua_effective = ops.exp(self.log_ua + delta_ua)

        heat_release = (
            ops.exp(self.log_heat_release)
            if self.log_heat_release is not None
            else ops.cast(self.physics.heat_release_k_l_mol, state_scaled.dtype)
        )

        # 缩放温度 x2 = (T - T_ref)/S 下的线性部分：
        #   dx1/dt = C_A0/tau - x1/tau + n1(x)
        #   dx2/dt = [(T_in-T_ref)/(S tau) + UA (T_c-T_ref)/S]
        #            - (1/tau + UA) x2 + n2(x)
        # 这个拆分让 IMEX 能对 1/tau 与 UA 做隐式处理。
        decay = ops.concatenate([ops.divide(1.0, tau), 1.0 / tau + ua_effective], axis=-1)
        forcing = ops.concatenate(
            [
                c_a0 / tau,
                (t_in - t_ref) / scale / tau + ua_effective * (t_c - t_ref) / scale,
            ],
            axis=-1,
        )

        def nonlinear(x: Any) -> Any:
            # 每个 RK4 stage 用该 stage 的温度重算 Arrhenius，避免在大温升
            # 时把步首速率冻结导致能量衡算失真。
            x_temp_k = x[:, 1:2] * scale + t_ref
            k_stage = self.arrhenius_rate(x_temp_k) * ops.exp(
                delta_rate + delta_activity
            )
            rate_stage = k_stage * x[:, 0:1]
            return ops.concatenate(
                [-rate_stage, heat_release / scale * rate_stage], axis=-1
            )

        integrated = integrate_interval(
            state_scaled,
            self.physics.dt_min,
            method=self.integrator.method,
            substeps=self.integrator.substeps,
            decay_per_min=decay,
            forcing=forcing,
            nonlinear=nonlinear,
            derivative_clip=self.integrator.derivative_clip,
        )

        if self.integrator.clip_state_to_guard:
            c_low, c_high = self.physics.c_a_guard_mol_l
            t_low = (self.physics.t_guard_k[0] - t_ref) / scale
            t_high = (self.physics.t_guard_k[1] - t_ref) / scale
            integrated, nonfinite, clipped = guard_state(
                integrated,
                state_scaled,
                c_a_bounds=(c_low, c_high),
                t_bounds=(t_low, t_high),
            )
        else:
            batch = ops.shape(integrated)[0]
            nonfinite = ops.zeros((batch,), dtype="bool")
            clipped = ops.zeros((batch,), dtype="bool")
        return integrated, nonfinite, clipped, penalty

    # ------------------------------------------------------------------
    # 单步推理（影子运行/在线软测量）
    # ------------------------------------------------------------------
    def domain_status(self, state: Any, control: Any, slow: Any) -> dict[str, float]:
        """返回给定工况的域内程度（1 = 域内，0 = 远离训练域）。

        这是一个**独立的只读诊断**，不在 ``call`` 里计算：训练图内读取张量
        的 Python 标量会破坏 graph mode。影子运行按需调用它即可。

        ``rate_gate`` 明显小于 1 说明该工况位于训练域外，修正网络已部分
        关闭；此时预测更依赖机理，应检查是否需要扩充训练工况。
        """
        import numpy as np

        state = ops.cast(state, "float32")
        control = ops.cast(control, "float32")
        slow = ops.cast(slow, "float32")
        # 注意：state 是**物理单位**（C_A mol/L，T K），不要再乘温度尺度。
        temperature_k = state[:, 1:2]
        features = self._rate_features(
            state[:, 0:1], temperature_k, ops.expand_dims(control, 1), 0
        )
        gates = {"rate_gate": float(np.asarray(self.domain_gate(features)).mean())}
        step_slow = slow[:, None, :] if len(ops.shape(slow)) == 2 else slow
        if self.slow_net is not None:
            gates["drift_gate"] = float(
                np.asarray(self.slow_domain_gate(step_slow[:, 0, :])).mean()
            )
        return gates

    def step(
        self, state: Any, control: Any, slow: Any, *, use_corrections: bool = True
    ) -> Any:
        """从当前状态推进**一个**采样间隔，返回 ``(next_state, diagnostics)``。

        ``state`` 为物理单位 ``(batch, 2)``；``control`` 为 ``(batch, 4)``；
        ``slow`` 为 ``(batch, 3)``。返回的 ``next_state`` 仍为物理单位，
        与 ``call`` 的输出坐标不同，专门供流式推理使用，避免在 Python
        层反复拼接/拆解轨迹张量。

        ``use_corrections=False`` 时跳过全部网络修正，只保留已拟合的机理
        参数——影子运行用它同时得到“机理基线”与“灰箱预测”，两者共用同一
        积分实现，因此残差修正值不会混入数值格式差异。

        本方法与 ``call`` 共用同一 ``_advance``：离线训练与影子运行不
        允许存在两套物理实现。
        """
        state = ops.cast(state, "float32")
        control = ops.cast(control, "float32")
        slow = ops.cast(slow, "float32")
        scale = self.physics.temperature_scale_k
        t_ref = self.physics.t_ref_k
        scaled = ops.concatenate(
            [state[:, 0:1], (state[:, 1:2] - t_ref) / scale], axis=-1
        )
        next_scaled, nonfinite, clipped, penalty = self._advance(
            scaled,
            ops.expand_dims(control, axis=1),
            slow,
            0,
            use_corrections=use_corrections,
        )
        next_state = ops.concatenate(
            [next_scaled[:, 0:1], next_scaled[:, 1:2] * scale + t_ref], axis=-1
        )
        return next_state, (nonfinite, clipped, penalty)

    # ------------------------------------------------------------------
    # 序列化
    # ------------------------------------------------------------------
    def get_config(self) -> dict[str, Any]:
        return {"config": asdict(self.config), **super().get_config()}

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "CSTRGreyBoxLayer":
        from .config import config_from_dict

        layer_config = dict(config)
        layer_config["config"] = config_from_dict(layer_config["config"])
        return cls(**layer_config)


__all__ = ["CSTRGreyBoxLayer"]
