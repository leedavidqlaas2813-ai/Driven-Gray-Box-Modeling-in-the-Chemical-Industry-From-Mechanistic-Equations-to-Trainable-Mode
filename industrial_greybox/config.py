"""工业灰箱软测量系统的配置对象。

设计原则
--------
1. **单位写进变量名**：代码中不出现无单位的 ``k``、``T``、``tau``；
   所有物理量在属性名末尾标注单位，避免仿真、训练和现场数据交接时混用。
2. **配置不可变**：所有配置使用 ``frozen=True`` 的 dataclass，训练、
   影子运行和报表可安全共享同一份配置，不会在运行期间被意外改写。
3. **默认值只是教学先验**：默认的活化能、传热系数等来自教程中的合成案例，
   **不是**任何真实装置的标定值。接入现场数据前必须逐项审核。

环境变量 ``KERAS_BACKEND`` 必须在导入 ``keras`` 之前设置；本包不在导入
配置时触发 Keras 初始化，因此 ``KERAS_BACKEND`` 可由调用方控制。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Mapping

# ---------------------------------------------------------------------------
# 工业数据列名约定
# ---------------------------------------------------------------------------
# 这些列名同时被数据清洗、训练集构建和影子运行使用。现场数据接入时应先
# 通过 rename/mapping 映射到统一列名，而不是在下游到处写别名。
COL_RUN_ID = "run_id"
COL_TIME_MIN = "time_min"
COL_C_A0 = "c_a0_mol_L"
COL_T_IN = "feed_temperature_K"
COL_T_C = "coolant_temperature_K"
COL_TAU = "residence_time_min"
COL_T_REACTOR = "reactor_temperature_K"
#: 稀疏化验/色谱结果（训练与评估的**参考**浓度）。
COL_C_A = "c_a_out_mol_L"
#: 在线分析仪读数（高频但有噪声/漂移）；与化验值语义不同，不能混用。
COL_C_A_ONLINE = "c_a_online_mol_L"
COL_CAMPAIGN_TIME = "campaign_time_min"
COL_RUN_AGE = "run_age_days"
COL_CUM_FEED = "cumulative_feed_m3"
COL_EMA_T = "ema_temperature_K"

#: 操作变量顺序。任何 (N, 4) 数组都遵循此顺序。
CONTROL_COLUMNS: tuple[str, str, str, str] = (
    COL_C_A0,
    COL_T_IN,
    COL_T_C,
    COL_TAU,
)

#: 训练/影子运行所需的最小 DCS 列。
DCS_COLUMNS: tuple[str, ...] = (
    COL_RUN_ID,
    COL_TIME_MIN,
    *CONTROL_COLUMNS,
    COL_T_REACTOR,
)

#: 可选的高频在线分析仪列。若存在，可用于状态初始化；**不**作为监督目标。
OPTIONAL_DCS_COLUMNS: tuple[str, ...] = (COL_C_A_ONLINE,)

#: 慢特征顺序。``SlowFeatureTransformer`` 也按此顺序输出。
SLOW_FEATURE_COLUMNS: tuple[str, ...] = (
    COL_RUN_AGE,
    COL_CUM_FEED,
    COL_EMA_T,
)


@dataclass(frozen=True)
class PhysicsConfig:
    """非等温 CSTR 的名义机理参数（先验值，可在训练中部分校正）。

    方程与单位::

        r      = k_eff(T) * C_A                    [mol/(L*min)]
        dC_A/dt = (C_A0 - C_A)/tau - r             [mol/(L*min)]
        dT/dt   = (T_in - T)/tau
                  + HR * r
                  + UA_eff * (T_c - T)             [K/min]

    - ``heat_release_k_l_mol``：``(-ΔH_r)/(ρ Cp)``，单位 K·L/mol。
    - ``nominal_ua_per_min``：``UA/(ρ Cp V)``，单位 min⁻¹，供学习初值使用。
    - 传热项 ``UA_eff*(T_c - T)`` 默认把夹套当作理想恒温边界；真实夹套
      动态、壁厚热惯性和冷却水流量变化需要扩展状态，见 ``README.md``。
    """

    gas_constant_j_mol_k: float = 8.314
    activation_energy_j_mol: float = 42_000.0
    t_ref_k: float = 350.0
    heat_release_k_l_mol: float = 20.0
    nominal_k_ref_per_min: float = 0.025
    nominal_ua_per_min: float = 0.08
    dt_min: float = 0.5
    n_steps: int = 40
    #: 釜有效体积 [L]。用于把停留时间换算成流量，从而累计进料量：
    #: ``q = V / tau``。若现场有直接流量计，应优先使用实测流量并在
    #: 数据字典中注明体积基准（含气相/持液量假设）。
    reactor_volume_l: float = 50.0

    # 数值求解的安全包围盒。它们**不是**工艺安全限值，只用于防止数值发散。
    c_a_guard_mol_l: tuple[float, float] = (0.0, 5.0)
    t_guard_k: tuple[float, float] = (280.0, 520.0)

    # 训练目标中温度的缩放尺度（K），与教程 05 保持一致。
    temperature_scale_k: float = 20.0

    def __post_init__(self) -> None:
        if self.reactor_volume_l <= 0:
            raise ValueError("reactor_volume_l 必须为正。")
        if self.activation_energy_j_mol <= 0:
            raise ValueError("activation_energy_j_mol 必须为正。")
        if self.t_ref_k <= 0:
            raise ValueError("t_ref_k 必须为绝对温度。")
        if self.dt_min <= 0 or self.n_steps < 1:
            raise ValueError("dt_min 必须为正，n_steps 至少为 1。")
        if not (
            self.c_a_guard_mol_l[0]
            <= 0
            < self.c_a_guard_mol_l[1]
        ):
            raise ValueError("c_a_guard_mol_l 必须包含 0 且上界为正。")
        if self.t_guard_k[0] >= self.t_guard_k[1]:
            raise ValueError("t_guard_k 下界必须小于上界。")
        if self.nominal_k_ref_per_min <= 0 or self.nominal_ua_per_min < 0:
            raise ValueError("速率常数必须为正，传热系数不能为负。")

    @property
    def horizon_min(self) -> float:
        """单条训练/仿真轨迹的时长 [min]。"""
        return self.n_steps * self.dt_min


@dataclass(frozen=True)
class IntegratorConfig:
    """可微机理层的数值积分设置。

    - ``rk4``：经典四阶显式 Runge–Kutta，精度高，但对极刚性系统仍受
      稳定域限制；``substeps`` 用于缩小有效步长。
    - ``imex``：线性隐式 / 非线性显式 Euler。CSTR 的 ``-C_A/tau`` 与
      ``-UA*T`` 线性衰减项被隐式处理，因此对长停留时间/强冷却的刚性问题
      比显式 Euler 稳定得多；格式一阶，建议配 ``substeps>=2``。
    - ``euler``：仅用于复现教程基线或做对比，不推荐用于生产。

    ``derivative_clip`` 可进一步限制单步状态变化幅值（单位与状态一致），
    默认关闭；它可能掩盖模型错误，启用时必须记录在模型版本中。
    """

    method: Literal["euler", "rk4", "imex"] = "rk4"
    substeps: int = 2
    derivative_clip: tuple[float, float] | None = None
    clip_state_to_guard: bool = True

    def __post_init__(self) -> None:
        if self.method not in ("euler", "rk4", "imex"):
            raise ValueError(f"未知积分方法: {self.method}")
        if self.substeps < 1:
            raise ValueError("substeps 必须 >= 1。")
        if self.derivative_clip is not None:
            low, high = self.derivative_clip
            if not low < high:
                raise ValueError("derivative_clip 必须是 (下界, 上界)。")


@dataclass(frozen=True)
class FeatureScaling:
    """网络输入的无量纲化常数。

    这些常数应来自训练工况的物理范围，而不是测试集统计量。现场部署时
    把它们与模型权重一起版本化，避免“同一模型、不同缩放”的静默漂移。
    """

    c_a_ref_mol_l: float = 1.0
    c_a0_ref_mol_l: float = 1.1
    c_a0_scale_mol_l: float = 0.3
    temperature_ref_k: float = 350.0
    temperature_scale_k: float = 20.0
    coolant_ref_k: float = 300.0
    coolant_scale_k: float = 20.0
    tau_ref_min: float = 5.5
    tau_scale_min: float = 2.5


@dataclass(frozen=True)
class RateCorrectionConfig:
    """动力学速率常数的数据驱动对数修正网络。

    ``k_eff = k_Arrhenius * exp(delta_rate)``，``delta_rate`` 被
    ``max_log_correction`` 截断，保证速率始终为正且修正幅度可审计。

    ``hidden_units=0`` 表示**关闭**速率修正网络，得到纯机理基线；它与
    ``enable_slow_drift=False`` 一起构成“无数据修正”的对照模型。
    """

    hidden_units: int = 12
    max_log_correction: float = 0.25
    l2_penalty: float = 1e-4

    def __post_init__(self) -> None:
        if self.hidden_units < 0:
            raise ValueError("hidden_units 不能为负（0 表示关闭修正网络）。")
        if self.max_log_correction <= 0:
            raise ValueError("max_log_correction 必须为正。")


@dataclass(frozen=True)
class SlowDriftConfig:
    """传热系数/催化剂活性的慢变修正（结垢、老化）。

    物理假设：

    - 结垢使 ``UA_eff`` 单调不增：``UA_eff = UA_0 * exp(delta_ua)``，
      ``delta_ua <= 0``。
    - 催化剂活性随累计进料缓慢下降：``k_eff *= exp(delta_activity)``，
      ``delta_activity <= 0``。
    - 允许一个**小幅**残余网络修正非单调效应（例如清洗、切换原料），
      上界由 ``max_residual_log`` 控制。

    若现场存在清洗后恢复、换催化剂等事件，应把事件标记作为特征输入，
    而不是让网络从时间外推。
    """

    age_ref_days: float = 90.0
    feed_ref_m3: float = 2_000.0
    max_decay_log: float = 0.6
    max_residual_log: float = 0.12
    residual_hidden_units: int = 8
    enable_activity: bool = True
    #: 慢变网络输出层的初始偏置。必须显著为负，使 ``sigmoid(bias)`` 接近
    #: 0，从而 **训练起点就是名义机理模型**（结垢修正 ≈ 0）。若用默认
    #: 的 0 偏置，``sigmoid(0)=0.5`` 会让 UA 一上来就比名义值低
    #: ``0.5*max_decay_log``（默认约 26%），等于用一个未经数据支持的
    #: “装置已经结垢一半”的假设去拟合，会系统性推高预测浓度。
    initial_decay_bias: float = -6.0

    def __post_init__(self) -> None:
        if self.age_ref_days <= 0 or self.feed_ref_m3 <= 0:
            raise ValueError("慢特征参考尺度必须为正。")
        if not 0 < self.max_decay_log <= 2.0:
            raise ValueError("max_decay_log 应在 (0, 2] 内。")
        if self.residual_hidden_units < 1:
            raise ValueError("residual_hidden_units 必须 >= 1。")
        if self.max_residual_log < 0:
            raise ValueError("max_residual_log 不能为负。")
        if self.initial_decay_bias > 0:
            raise ValueError(
                "initial_decay_bias 必须 <= 0：结垢/老化修正的起点应接近零。"
            )


@dataclass(frozen=True)
class LongHorizonConfig:
    """长期外推的正则化与渐近边界。

    为什么需要
    ----------
    训练窗口（例如 20–40 min）内学到的速率/慢变修正，在连续滚动 100 min
    以上时缺乏回拉约束：任何小的速率偏差都会累积成浓度偏差。实测中灰箱
    模型可以在 20 min 开环上优于机理基线、却在 100 min 影子运行上更差。

    两类措施（可分别开关）：

    1. **多步滚动一致性惩罚**（``rollout_penalty``）
       从模型自身在窗口中途的预测状态重新起算，再滚到窗口末尾，要求其与
       从真实锚点出发的轨迹一致。这是一个"半群/流复合"约束：正确的 ODE
       满足 ``Φ(t2,t0) = Φ(t2,t1)∘Φ(t1,t0)``，违反它说明模型在长滚动中
       不自洽。它**不需要额外数据**，因此可以直接在现有窗口上抽样计算。

    2. **域外渐近回退**（``domain_gate_*``）
       修正网络在远离训练分布时会给出没有数据支持的外推。用输入到训练域
       的归一化距离做平滑门控，使**符号自由**的修正（速率修正、慢变残余
       项）在域外平滑衰减到 0，即退回名义机理。

       注意：**单调的结垢/老化主趋势不衰减到 0，而是饱和保持**。理由见
       ``SlowDriftConfig``：若老化趋势在域外"回退到零"，等于假设老装置
       没有结垢，会低估 UA 衰减、系统性低估出口浓度。物理上更安全的域外
       行为是保持最后一个有界值（饱和），而不是清零。本包按此实现。
    """

    #: 是否启用多步滚动一致性惩罚。
    rollout_penalty: bool = True
    #: 该惩罚的权重。
    #:
    #: 默认开启 ``rollout_penalty_normalize``，此时该值表达"惩罚归一化后
    #: 应占数据拟合项的比例"（典型 0.05–0.2），与装置尺度无关。
    #:
    #: 若关闭归一化，则退化为绝对系数——**此时必须自行校准**：实测在
    #: 扰动 0.01 mol/L、40 步窗口下惩罚约 5e-6，而主损失约 0.07，需要
    #: 约 1500 的权重才能占到 10%。初版把该值设为 0.1，惩罚只占主损失的
    #: 7e-6，形同虚设。这正是引入归一化的原因。
    rollout_penalty_weight: float = 0.1
    #: 是否把惩罚归一化到"与数据损失同量纲"（推荐 True）。
    #:
    #: 参照量取 ``anchor_noise² × max_steps``：它是"扰动在 k 步内未被
    #: 吸回时"的典型偏差量级，与轨迹长度和扰动幅度同量纲，因此归一化后
    #: 的权重可跨装置直接复用，不必每个现场重新猜。
    rollout_penalty_normalize: bool = True
    #: 每个窗口抽样几次"中途重锚"（每次随机选取重锚点）。
    rollout_samples: int = 1
    #: 重锚点允许的最小/最大比例位置（相对窗口长度）。太靠末尾时轨迹太短，
    #: 惩罚没有信息量；太靠开头时与主损失几乎重复。
    rollout_min_fraction: float = 0.25
    rollout_max_fraction: float = 0.75
    #: 从重锚点向后展开的最大步数（**不是**比例）。超过它则截断。
    #:
    #: 为什么要截断而不是滚到窗口末尾：惩罚积分嵌套在正向积分里，若从
    #: 中途点一直滚到末尾，计算图大小近似翻倍（40 步窗口 → 另外 30 步的
    #: 展开，且每步含 RK4×子步）。对长时间窗口，这会显著增加内存并可能
    #: MemoryError。
    #:
    #: 截断在数学上仍然有效：从中间点滚 `k` 步（`k` 接近窗口一半）已经能
    #: 检验"从任意状态出发的自洽性"；完全滚到末尾带来的额外约束有限，
    #: 而内存代价线性增长。现场可按显存把该值调大。
    rollout_max_steps: int = 15
    #: 重锚扰动幅度：浓度通道 [mol/L]。
    #:
    #: **必须非零**，否则惩罚恒为 0：确定性积分器从精确状态重新起算会
    #: 精确复现原轨迹（半群性质），不产生任何梯度。有意义的检验是"从被
    #: 重锚扰动过的起点能否回到原轨迹"——这正是部署时的真实情形。
    #:
    #: 取值应接近现场重锚的实际量级（约 ``reseed_smoothing`` × 典型残差）。
    #: 太大变成"任意初值都能收敛"的过强约束，太小则梯度信号微弱。
    #:
    #: 惩罚值随扰动幅度**近似平方**增长，这是校准 weight 时要一并考虑的量。
    rollout_anchor_noise_c_a: float = 0.01
    #: 重锚扰动幅度：温度通道（缩放坐标，1.0 = 20 K）。
    rollout_anchor_noise_temperature: float = 0.01

    #: 是否启用域外门控（对符号自由的修正项）。
    domain_gate: bool = True
    #: 训练域盒半径，单位是**归一化特征尺度**（约等于标准差数）。
    #:
    #: **这个值必须按现场工况范围校准，不能沿用示例值。** 归一化特征见
    #: ``FeatureScaling``：温度按 20 K 缩放、进料浓度按 0.3 mol/L 缩放。
    #: 因此 ``box=0.5`` 约等于"温度偏离参考点 10 K"。
    #:
    #: 校准经历：初版用 3.0（≈60 K）→ 连 15 K 偏移都不触发；改为 1.5
    #: （≈30 K）→ 测试集外推 17 K（归一化 0.86）仍不触发。真正的问题是
    #: "归一化尺度（20 K）比实际工况变化幅度大得多"。
    #:
    #: **推荐做法**：用 ``diagnostics.calibrate_domain_box`` 从训练集
    #: 自动推导（取各维 |z| 的 95% 分位最大值 × 1.2）。示例工况下为
    #: 约 1.55；其中温度与停留时间是主导维（P95 分别约 1.29、1.02）。
    #:
    #: 注意：**不要用 max 校准**——在线仪表含毛刺，max 会到 17，导致门控
    #: 永远不触发。分位数对坏值稳健。
    domain_box: float = 1.5
    #: 盒外的过渡宽度（归一化单位）。``taper=0.5`` 表示再往外半个
    #: "标准尺度"内平滑衰减到 0。取 0.5 使过渡区落在典型外推量级内
    #: （例如温度再高 10 K）。
    domain_taper: float = 0.5

    def __post_init__(self) -> None:
        if self.rollout_penalty_weight < 0:
            raise ValueError("rollout_penalty_weight 不能为负。")
        if self.rollout_samples < 0:
            raise ValueError("rollout_samples 不能为负。")
        if not 0 <= self.rollout_min_fraction < self.rollout_max_fraction <= 1:
            raise ValueError(
                "需要 0 <= rollout_min_fraction < rollout_max_fraction <= 1。"
            )
        if self.rollout_max_steps < 1:
            raise ValueError("rollout_max_steps 必须 >= 1。")
        if self.rollout_anchor_noise_c_a < 0 or self.rollout_anchor_noise_temperature < 0:
            raise ValueError("重锚扰动幅度不能为负。")
        if self.domain_box <= 0:
            raise ValueError("domain_box 必须为正。")
        if self.domain_taper <= 0:
            raise ValueError("domain_taper 必须为正。")


@dataclass(frozen=True)
class ModelConfig:
    """一个灰箱 CSTR 模型的完整配置。"""

    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    integrator: IntegratorConfig = field(default_factory=IntegratorConfig)
    scaling: FeatureScaling = field(default_factory=FeatureScaling)
    rate_correction: RateCorrectionConfig = field(default_factory=RateCorrectionConfig)
    slow_drift: SlowDriftConfig = field(default_factory=SlowDriftConfig)
    long_horizon: LongHorizonConfig = field(default_factory=LongHorizonConfig)
    learn_heat_release: bool = False
    #: 是否启用慢变结垢/活性修正。``False`` 时模型退化为“参数 + 速率修正”。
    enable_slow_drift: bool = True
    #: 损失中的通道权重 ``[C_A, T]``。通道内部已各自归一化（见
    #: ``models.masked_squared_error``），这里只表达“谁更重要”。
    #: 质量软测量中浓度是主目标、温度是辅助约束，默认 ``[1.0, 0.25]``。
    #: 不要用逐点权重（例如给温度乘 0.25）来表达这件事：逐点权重与采样
    #: 频率混在一起，会意外地让某个通道主导总损失。
    channel_weights: tuple[float, float] = (1.0, 0.25)

    def __post_init__(self) -> None:
        if len(self.channel_weights) != 2:
            raise ValueError("channel_weights 必须是 (C_A, T) 两个值。")
        if any(value < 0 for value in self.channel_weights):
            raise ValueError("channel_weights 不能为负。")
        if sum(self.channel_weights) <= 0:
            raise ValueError("channel_weights 之和必须为正。")

    def signature(self) -> str:
        """返回可写进日志的确定性配置指纹。"""
        import hashlib
        import json
        from dataclasses import asdict

        payload = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def config_from_dict(payload: Mapping[str, object]) -> ModelConfig:
    """从 ``ModelConfig.signature`` 所用的嵌套字典重建配置。

    用于 Keras 层 ``get_config``/``from_config`` 往返。嵌套 dataclass 按
    显式注册表重建（不依赖运行时的字符串注解）；list 形式的 tuple 字段
    会还原为 tuple；未知键直接报错，避免版本升级后配置被静默忽略。
    """
    nested: dict[str, type] = {
        "physics": PhysicsConfig,
        "integrator": IntegratorConfig,
        "scaling": FeatureScaling,
        "rate_correction": RateCorrectionConfig,
        "slow_drift": SlowDriftConfig,
        "long_horizon": LongHorizonConfig,
    }
    tuple_fields = {
        "c_a_guard_mol_l",
        "t_guard_k",
        "derivative_clip",
        "channel_weights",
    }

    def build(cls: type, data: Mapping[str, object]) -> object:
        from dataclasses import fields

        known = {field.name for field in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"{cls.__name__} 存在未知配置项：{sorted(unknown)}")
        missing = known - set(data)
        if missing:
            raise ValueError(f"{cls.__name__} 缺少配置项：{sorted(missing)}")
        kwargs: dict[str, object] = {}
        for name in known:
            value = data[name]
            if name in nested and isinstance(value, Mapping):
                kwargs[name] = build(nested[name], value)
            elif name in tuple_fields and value is not None:
                kwargs[name] = tuple(value)  # type: ignore[arg-type]
            else:
                kwargs[name] = value
        return cls(**kwargs)  # type: ignore[arg-type]

    return build(ModelConfig, payload)  # type: ignore[return-value]


@dataclass(frozen=True)
class DelayConfig:
    """化验/在线分析仪的纯滞后配置（单位：min）。

    总死时间 = 取样管线传输滞后 + 分析/化验周转时间。关键约定：

    - 化验结果在 ``result_time`` 到达 DCS；
    - 该结果代表 ``process_time = result_time - total_dead_time`` 时刻的
      反应器出口状态；
    - **绝不能**把迟到的化验值直接和当前时刻的预测相减，否则残差里会
      混入死时间，形成虚假的模型偏差。
    """

    line_delay_min: float = 2.0
    analysis_delay_min: float = 10.0
    lab_interval_min: float = 5.0

    def __post_init__(self) -> None:
        if self.line_delay_min < 0 or self.analysis_delay_min < 0:
            raise ValueError("滞后时间不能为负。")
        if self.lab_interval_min <= 0:
            raise ValueError("lab_interval_min 必须为正。")

    @property
    def total_dead_time_min(self) -> float:
        return self.line_delay_min + self.analysis_delay_min


@dataclass(frozen=True)
class LimitRule:
    """硬约束/物理边界规则。"""

    column: str
    lower: float
    upper: float
    action: Literal["nan", "clip", "flag"] = "nan"

    def __post_init__(self) -> None:
        if not self.lower < self.upper:
            raise ValueError(f"{self.column}: lower 必须小于 upper。")
        if self.action not in ("nan", "clip", "flag"):
            raise ValueError(f"{self.column}: 未知 action={self.action}")


@dataclass(frozen=True)
class RateOfChangeRule:
    """变化率/突变规则。

    ``max_abs_per_min`` 限制物理上不可能的斜率；``max_abs_jump`` 限制单个
    采样间隔内的绝对跳变。

    **默认只标记（``action='flag'``），不删除。** 原因是 DCS 里存在大量
    *真实的* 阶跃：操作员改设定值、切牌号、启停泵都会让温度/浓度在几个
    采样点内大幅跳变。若把它们当仪表故障删掉，训练数据会系统性缺少工况
    切换段，模型也就学不会动态响应。只有确认某条规则只可能由仪表故障
    触发（例如超出变送器量程变化速率）时，才改成 ``'nan'``。
    """

    column: str
    max_abs_per_min: float
    max_abs_jump: float | None = None
    action: Literal["flag", "nan"] = "flag"

    def __post_init__(self) -> None:
        if self.max_abs_per_min <= 0:
            raise ValueError(f"{self.column}: max_abs_per_min 必须为正。")
        if self.max_abs_jump is not None and self.max_abs_jump <= 0:
            raise ValueError(f"{self.column}: max_abs_jump 必须为正。")
        if self.action not in ("flag", "nan"):
            raise ValueError(f"{self.column}: 未知 action={self.action}")


@dataclass(frozen=True)
class DeadbandRule:
    """仪表死区 / 假恒定状态识别。

    用滑动窗口内的标准差和极差判断“信号长时间几乎不动”。注意：真实稳态
    也会产生低方差，因此：

    - 默认 ``action='flag'`` 只标记不删除，并且**不计入“可疑值”总数**，
      而是单独作为“信息性标记”，供仪表工程师复核；
    - 只有当窗口跨越 ``min_duration_min`` 以上持续不变时才算命中，避免把
      一次正常的设定值保持判成故障；
    - 只有与工艺知识（该变量在时间常数内必然变化）结合后，才可改为
      ``'nan'``。
    """

    column: str
    window_min: float = 10.0
    min_duration_min: float = 15.0
    std_tol: float = 1e-4
    span_tol: float = 5e-4
    action: Literal["flag", "nan"] = "flag"

    def __post_init__(self) -> None:
        if self.window_min <= 0:
            raise ValueError(f"{self.column}: window_min 必须为正。")
        if self.min_duration_min < self.window_min:
            raise ValueError(
                f"{self.column}: min_duration_min 不能小于 window_min，"
                "否则短窗口会频繁误报。"
            )
        if self.std_tol < 0 or self.span_tol < 0:
            raise ValueError(f"{self.column}: 容差不能为负。")


@dataclass(frozen=True)
class SpikeRule:
    """基于 Hampel（滑动中位数 + MAD）的孤立毛刺检测。"""

    column: str
    window_samples: int = 7
    n_sigma: float = 6.0

    def __post_init__(self) -> None:
        if self.window_samples < 3 or self.window_samples % 2 == 0:
            raise ValueError("window_samples 必须是 >=3 的奇数。")
        if self.n_sigma <= 0:
            raise ValueError("n_sigma 必须为正。")


@dataclass(frozen=True)
class HygieneConfig:
    """工业数据清洗流水线配置。

    默认规则只覆盖本教程的 CSTR 变量。接入现场前应逐列重写边界和变化率，
    并把规则、阈值和审核人写进数据字典。
    """

    limits: tuple[LimitRule, ...] = field(
        default_factory=lambda: (
            LimitRule(COL_C_A_ONLINE, 0.0, 5.0),
            LimitRule(COL_T_REACTOR, 280.0, 520.0),
            LimitRule(COL_C_A0, 0.0, 5.0),
            LimitRule(COL_T_IN, 280.0, 520.0),
            LimitRule(COL_T_C, 250.0, 420.0),
            LimitRule(COL_TAU, 0.2, 60.0),
        )
    )
    rate_rules: tuple[RateOfChangeRule, ...] = field(
        default_factory=lambda: (
            RateOfChangeRule(COL_T_REACTOR, 10.0, 8.0),
            RateOfChangeRule(COL_C_A_ONLINE, 0.5, 0.4),
            RateOfChangeRule(COL_C_A0, 1.0, 0.8),
            RateOfChangeRule(COL_T_IN, 5.0, 5.0),
            RateOfChangeRule(COL_T_C, 5.0, 5.0),
            RateOfChangeRule(COL_TAU, 5.0, 3.0),
        )
    )
    deadband_rules: tuple[DeadbandRule, ...] = field(
        default_factory=lambda: (
            DeadbandRule(
                COL_T_REACTOR,
                window_min=5.0,
                min_duration_min=15.0,
                std_tol=2e-3,
                span_tol=8e-3,
            ),
            DeadbandRule(
                COL_C_A_ONLINE,
                window_min=10.0,
                min_duration_min=30.0,
                std_tol=1e-4,
                span_tol=5e-4,
            ),
        )
    )
    spike_rules: tuple[SpikeRule, ...] = field(
        default_factory=lambda: (
            SpikeRule(COL_T_REACTOR, window_samples=7, n_sigma=6.0),
            SpikeRule(COL_C_A0, window_samples=7, n_sigma=6.0),
            SpikeRule(COL_C_A_ONLINE, window_samples=7, n_sigma=6.0),
        )
    )
    #: 被清洗成 NaN 后允许线性插值的最大连续点数；超过则保留 NaN，由上层
    #: 决定丢弃整条运行还是要求人工处理。
    max_interpolate_gap: int = 3
    #: 只对这些列做插值；质量目标列（化验）不做插值，以免伪造标签。
    impute_columns: tuple[str, ...] = (
        COL_T_REACTOR,
        COL_C_A0,
        COL_T_IN,
        COL_T_C,
        COL_TAU,
        COL_C_A_ONLINE,
    )
    #: 时间网格允许的相对偏差。超过则拒绝按均匀网格构建训练张量。
    dt_tolerance: float = 1e-3


@dataclass(frozen=True)
class ShadowConfig:
    """影子运行与报表配置。"""

    group_by: Literal["run", "day", "window"] = "day"
    output_dir: Path = Path("shadow_runs")
    domain_margin: float = 0.05
    flush_every: int = 25
    #: 上线初期没有 C_A 化验时的初始估计；None 表示等第一条化验结果。
    initial_c_a_mol_l: float | None = None
    #: 温度测量是否作为状态反馈（推荐 True：温度通常高频可测）。
    use_temperature_feedback: bool = True
    #: 状态重锚（soft-sensor update）。默认 ``True``：收到一条真实化验后，
    #: 把浓度状态偏移量记入 ``bias_state_mol_l``，并从未观测的浓度通道中
    #: **扣除**该偏移，使后续预测从最新的可观测真值出发。
    #:
    #: 为什么需要：影子运行从运行起点一次锚定、随后纯开环滚动上百分钟，
    #: 而训练窗口只有几十分钟。若不重锚，误差会单调累积，任何模型都会
    #: “越跑越差”，这时比较模型优劣毫无意义。工业软测量的通行做法就是
    #: 每次化验到达时更新状态。
    #:
    #: 物理边界：偏移量会被限制在 ``max_bias_correction_mol_l`` 内。若某次
    #: 化验与模型差距远超该限值，说明模型/数据有更深层问题（工况外推、
    #: 仪表故障、死时间错误），此时**不应**强行用重锚掩盖。
    reseed_on_lab_result: bool = True
    max_bias_correction_mol_l: float = 0.15

    # ---- 自适应重锚（自适应置信度 + 指数平滑过渡）----------------------
    #: 自适应重锚模式：
    #:
    #: - ``"fixed"``：旧行为。``bias = clip(residual, ±max)`` 直接施加。
    #: - ``"adaptive"``：门控置信度 + 指数平滑（推荐）。
    #:
    #: 固定截断有两个真实问题：
    #:
    #: 1. 无法区分"仪表坏值"与"真实工况漂移"——两者都表现为大残差；
    #: 2. 即便残差可信，一次性加上全部偏差会让积分器状态阶跃，在刚性
    #:    工况下可能触发数值抖动。
    reseed_mode: Literal["fixed", "adaptive"] = "adaptive"

    #: 残差超过多少倍"正常波动"时，开始压缩置信度（单位：除以稳健尺度）。
    #: 用稳健尺度（MAD/1.4826）而不是标准差：单次坏值不会污染尺度估计。
    residual_gate_sigma: float = 4.0
    #: 置信度衰减的过渡宽度，同样是稳健尺度的倍数。
    residual_gate_taper_sigma: float = 4.0
    #: 连续多少次"大残差"才判定为真实漂移而非孤立坏值。
    #: 单个坏值只有一次大残差；真实漂移会连续出现。
    drift_confirm_count: int = 3
    #: 判定"大残差"的幅度阈值（相对稳健尺度的倍数）。
    #:
    #: 这个阈值与 ``residual_gate_sigma`` **分开**是必要的：门控阈值决定
    #: "降低多少信任"，而确认阈值决定"是否认定持续偏差"。若两者共用同一个
    #: 数值，会形成自锁——门控关闭后修正被压到 0，残差不会自动变小，于是
    #: 永远达不到确认条件，模型既不敢修正、也认不出漂移。
    #:
    #: 取 6 倍：孤立坏值通常远超此值（10σ+），而真实的稳定偏差往往在
    #: 4–8 倍尺度之间。若现场仪表噪声极大，应上调该值。
    drift_magnitude_sigma: float = 6.0
    #: 稳健尺度估计的滑动窗口长度（样本数）。太短会被坏值带偏，
    #: 太长无法跟上工况变化。
    residual_scale_window: int = 20
    #: 指数平滑系数 α ∈ (0, 1]。每来一条化验，施加 ``α · bias``：
    #:
    #: - α = 1 退化为瞬时修正（等同旧的固定模式，但不建议）；
    #: - α 小 → 过渡平滑，但收敛慢，需要更频繁的化验。
    #:
    #: 默认 0.5：两条化验内收敛到约 75% 的偏差。
    reseed_smoothing: float = 0.5

    def __post_init__(self) -> None:
        if self.group_by not in ("run", "day", "window"):
            raise ValueError("group_by 必须是 run/day/window。")
        if not 0 <= self.domain_margin <= 1:
            raise ValueError("domain_margin 应在 [0, 1]。")
        if self.flush_every < 1:
            raise ValueError("flush_every 必须 >= 1。")
        if self.max_bias_correction_mol_l < 0:
            raise ValueError(
                "max_bias_correction_mol_l 不能为负；0 表示关闭偏差修正。"
            )
        if self.reseed_mode not in ("fixed", "adaptive"):
            raise ValueError("reseed_mode 必须是 'fixed' 或 'adaptive'。")
        if self.residual_gate_sigma <= 0 or self.residual_gate_taper_sigma <= 0:
            raise ValueError("门控阈值必须是正的稳健尺度倍数。")
        if self.drift_confirm_count < 1:
            raise ValueError("drift_confirm_count 必须 >= 1。")
        if self.drift_magnitude_sigma <= 0:
            raise ValueError("drift_magnitude_sigma 必须为正。")
        if self.residual_scale_window < 3:
            raise ValueError("residual_scale_window 至少为 3。")
        if not 0 < self.reseed_smoothing <= 1:
            raise ValueError("reseed_smoothing 必须在 (0, 1] 内。")


#: 训练目标通道顺序：[C_A, T]。
TARGET_CHANNELS: tuple[str, str] = (COL_C_A, COL_T_REACTOR)

#: 现场 CSV 接入时建议的列名映射示例。真实项目应把映射写进数据字典。
EXAMPLE_COLUMN_MAPPING: Mapping[str, str] = {
    "run": COL_RUN_ID,
    "time": COL_TIME_MIN,
    "CA0": COL_C_A0,
    "TIN": COL_T_IN,
    "TC": COL_T_C,
    "TAU": COL_TAU,
    "T_REACTOR": COL_T_REACTOR,
    "CA_LAB": COL_C_A,
}
