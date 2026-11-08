# 工业灰箱软测量包：设计与边界

> **先读这一段。** 本包使用**合成数据**演示工程结构，所有结果都不是任何
> 真实装置的验证结论。它解决的问题是“把教学级代码变成可审查的工程骨架”，
> 不是“给出一个可以上线的模型”。上线前必须完成数据字典、单位审核、
> 独立验证、仪表校验和工艺/安全审批。

## 1. 为什么需要这一层

教程 03/05 的模型是**结构正确**的：物料与能量衡算、Arrhenius 速率、
受限网络修正、按运行切分。但它们还不足以应对真实装置的以下现实：

| 现实问题 | 教学代码的行为 | 本包的处理 |
|---|---|---|
| 准刚性工况（短停留时间、强冷却） | 固定步长显式 Euler 可能振荡/发散 | 可微 RK4（默认）+ IMEX 线性隐式 + 子步 |
| 长周期滚动累积误差 | 数值误差无隔离 | NaN/Inf 回退到上一步、状态守护区间、诊断计数 |
| 化验死时间（GC/离线分析 10–30 min） | 默认状态同步 | 结果时刻减死时间对齐；在线用时间戳缓冲回查 |
| 历史数据毛刺/阶跃/死区/越界 | 不处理 | 硬边界、Hampel 毛刺、变化率、死区、限长插值，全部留痕 |
| UA/活性缓慢漂移 | 参数固定 | 结垢/老化慢变修正，结构上单调不增 |
| 上线前无法评估 | 只有离线误差 | 因果流式影子运行 + 分层滚动报表 |

## 2. 模块与职责

```text
industrial_greybox/
├── config.py            全部参数、单位约定、列名、配置指纹
├── numerics.py          RK4 / IMEX / Euler、数值守护、稳定比诊断
├── delays.py            化验死时间对齐（离线）、ShadowBuffer（在线）
├── reanchor.py          自适应状态重锚（稳健门控 + 指数平滑）
├── adaptation.py        在线增量微调 + 安全熔断回滚
├── hygiene.py           工业清洗流水线（只删坏值，不伪造标签）
├── slow_features.py     运行年龄、累计进料、温度 EMA
├── mechanistic_layer.py 可微 CSTR：速率修正 + 慢变 UA/活性修正
├── data.py              按运行切分、窗口化、监督掩码
├── models.py            模型 + 逐点掩码损失（重写 compute_loss）
├── synthetic_plant.py   带漂移/时滞/故障的合成 campaign（仅测试）
├── training.py          训练循环、数值健康回调、开环误差表
├── reports.py           域内/域外、按运行/按日指标与文本报表
├── shadow.py            因果流式影子运行与诊断日志
└── pipeline.py          端到端装配
```

## 3. 五个维度的关键设计决策

### 3.1 数值积分：为什么默认 RK4 而不是 Euler 或 IMEX

CSTR 的线性部分为

```text
dC_A/dt = C_A0/τ − C_A/τ + n₁(x)
dT/dt   = (T_in − T)/τ + UA(T_c − T) + n₂(x)
```

显式 Euler 的稳定条件是 `h·(1/τ + UA) < 2`。以 `τ = 0.35 min`、
`UA = 0.6 min⁻¹`、`h = 1 min` 为例，`h·a = 3.7`，已经越过稳定边界；
温度会振荡并指数放大。

三种格式的取舍：

- **RK4**（默认）：四阶精度，对温升较快、Arrhenius 强非线性的工况误差
  最小；每个采样间隔默认 2 个子步。代价是每步 4 次 RHS 求值。
- **IMEX**：`1/τ` 与 `UA` 隐式处理，对线性刚性**无条件稳定**；反应项
  仍显式，所以极快的反应速率仍需减小步长。是一阶格式，适合作为
  “强冷却 + 短停留时间”工况的稳健选项。
- **Euler**：只用于复现教程基线和做对照，不建议在生产中使用。

**数值守护**不是物理约束：`c_a_guard_mol_l`、`t_guard_k` 只阻止积分爆炸。
出现非有限值时回退到**上一步**（零阶保持），而不是填 0——0 是有物理意义
的浓度，伪造它等于伪造标签。每次回退都计入 `nonfinite_steps` 指标。

### 3.2 时滞补偿：方向不能反

```text
化验结果在 result_time 到达 DCS
它代表 process_time = result_time − dead_time 时刻的反应器状态
```

因此：

- **离线训练**：把化验的**过程时刻**对齐到 DCS 网格，生成监督掩码；
  中间时刻的零阶保持值只用于画图，不进入损失。
- **在线影子**：把历史预测存进 `ShadowBuffer`，化验到达时按 `process_time`
  回查，计算真正的模型残差。

若反过来“用当前预测减迟到的化验值”，残差里会混入 `dead_time` 分钟的过程
变化。对快速动态工况，这会表现为**规律的滞后偏差**，而模型会被训练去
补偿这个假偏差——上线后死时间稍有变化就会失效。

执行机构延迟是另一回事：它改变过程本身，应从**操作变量**侧对齐
（`shift_controls_for_actuator_delay`），不能和化验死时间混用。

### 3.3 数据清洗：标记 ≠ 删除

清洗的目标是让错误可见且可追溯，不是让损失下降。因此：

- 每条规则只**标记**原始问题，`<列>__flag` 记录原因，
  `<列>__valid` 记录最终可用性；
- **被规则删除的值绝不插值**。否则清洗白做，且报表会掩盖问题；
- 质量目标（化验 `C_A`）从不插值——伪造化验值就是伪造标签；
- 死区/变化率默认只标记：真实稳态和真实阶跃都会触发，需要工程师复核，
  而不是自动删除。

规则顺序固定为 `硬边界 → Hampel 毛刺 → 变化率 → 死区 → 限长插值`。
硬边界优先，避免 ±1e9 的毛刺把后续的中位数/差分统计拉偏。

### 3.4 慢变参数：把物理先验写进结构

结垢使传热下降，老化使活性下降，因此

```text
UA_eff  = UA_0 · exp(δ_ua),        δ_ua  ≤ 0（主趋势）
k_eff   = k · exp(δ_rate + δ_activity), δ_activity ≤ 0（主趋势）
```

网络输出经 `sigmoid` 后取负，再叠加一个很小的符号自由的残余项（清洗、
换料等非单调事件）。这比让网络从数据里“学到”单调性更可靠，也能防止
在训练范围外外推出“结垢增强传热”。

**初始化必须让训练起点就是机理模型**：输出层偏置取 `-6`，使
`sigmoid(-6) ≈ 0.0025`。若用默认 0 偏置，`sigmoid(0) = 0.5` 会让 UA 一
上来就比名义值低约 26%，等于用一个未经数据支持的“装置已结垢一半”的假设
去拟合。这一点在单元测试里有断言。

### 3.5 影子运行：因果性、误差累积与状态重锚

影子运行只能使用当时已到达的信息：

1. `t_k` 的 DCS 采样到达后才使用它的测量与控制量；
2. 化验结果按 `process_time` 回查历史预测；
3. 首点 C_A 只能来自在线分析仪、配置初值或已到达的化验；
4. 温度可用测量做状态反馈（高频可测），浓度不订阅当前时刻的化验。

但因果性只是必要条件。还有一个常被忽略的问题：**影子运行从运行起点
锚定一次，之后滚动上百分钟**，而训练窗口可能只有 20–40 min。两者的
误差不可直接比较——同一对模型可能在“20 min 开环”和“100 min 滚动”上
排名相反。

因此本包提供两件事：

- `reports.metrics_by_horizon`：按预测时长分箱报告误差，把“短期动态”
  与“长期漂移”分开看；
- **有界状态重锚**：每条化验到达时，用过程时刻的模型误差修正当前状态
  （详见下文），把软测量变成闭环更新，而不是纯开环外推。

重锚的正确写法：

```text
化验对应 process_time，而当前状态在 time_min，中间隔着一整个死时间
bias = clip(lab − ŷ(process_time), ±max_bias_correction_mol_l)
C_A(t_now) ← C_A(t_now) + bias
```

不能把 `lab_value` 直接赋给当前状态。偏差必须有界：超出限值说明外推、
仪表故障或死时间错误，强行重锚只会掩盖问题。日志记录
`applied_bias_mol_l` 与 `bias_clipped`。

### 3.6 影子日志与修正量的可解释性

影子日志列固定为：

```text
campaign_time_min, run_id, time_min,
c_a0_mol_L, feed_temperature_K, coolant_temperature_K, residence_time_min,
reactor_temperature_K,
mechanistic_c_a, mechanistic_temperature_K,
greybox_c_a, greybox_temperature_K,
correction_c_a, correction_temperature_K,
online_c_a, lab_c_a, truth_c_a,
residual_vs_lab, residual_vs_truth,
in_domain, nonfinite_step, clipped_step, stale_history, config_signature
```

控制量与实测温度也写入日志，这样 `in_domain` 可以由日志**本身**复核；
否则把 CSV 交给离线分析时无法判断某个残差是否来自外推。
`reports.domain_flags` 在缺列时会直接报错，而不是默认“全部在域内”。

`mechanistic_*` 与 `greybox_*` 共用**同一个**积分实现（`use_corrections`
开关），因此 `correction_*` 纯粹反映网络修正，不混入数值格式差异。

## 4. 损失设计：为什么必须按通道归一化

工业标签的监督密度极不均衡：一个 4 h 窗口里，温度有 41 个采样点全程可测，
浓度却可能只有 6 个化验点。若把两个通道混在一起归一化：

```text
loss = Σ_points w·err² / Σ_points w
```

温度项会以约 `41/T_scale : 6` 的比例压倒浓度项。网络于是优先拟合温度，
让质量软测量的主目标变差——这不是网络容量不足，而是**损失写错了**。

正确的两步结构（见 `models.masked_squared_error`）：

1. **通道内归一化**：每个通道的损失 = 该通道加权平方误差 / 该通道权重和；
2. **通道间加权**：按 `ModelConfig.channel_weights`（默认 `[1.0, 0.25]`）
   合并，表达“浓度是主目标、温度是辅助约束”。

对应的约定：

- `WindowedDataset.weights` 只做 **0/1 掩码**（该点是否可监督）；
- 通道重要性只写在 `channel_weights` 里，不要用逐点权重表达——
  逐点权重会同时编码“数据密度”和“工艺重要性”，无法解释。

## 5. 训练中的一个关键陷阱（Keras 3）

**Keras 3 的 `Loss.__call__` 不会把 `sample_weight` 传给子类的 `call`。**
它先调用 `call(y_true, y_pred)`，再把权重交给 `reduce_weighted_values`
做整体缩放。因此把逐点掩码写在 `Loss.call` 里是**无效**的：

```python
# 错误：sample_weight 在这里永远是 None
class MaskedLoss(keras.losses.Loss):
    def call(self, y_true, y_pred, sample_weight=None): ...
```

而工业数据里“没有化验”和“浓度为 0”必须区分。若掩码失效、未监督点又被
填成 0，模型实际学到的是“出口浓度趋向 0”——一个在安全上很危险的偏差。

正确做法是重写 `Model.compute_loss`（它**能**拿到 `sample_weight`）：

```python
def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None, training=True):
    total = masked_squared_error(y, y_pred, sample_weight)
    for extra in self.losses:
        total = total + self._aggregate_additional_loss(extra)
    return total
```

本包在 `models.py` 中实现了这一点，并用单元测试固定语义：
未监督点的离谱误差不得影响损失值。

## 6. 长期外推、自适应重锚与在线微调

本节对应三个新增模块。它们解决的问题都是"离线训练好、上线后慢慢失配"。

### 6.1 多步滚动一致性惩罚（`config.LongHorizonConfig`）

正确的 ODE 流满足半群性质 `Φ(t2,t0) = Φ(t2,t1)∘Φ(t1,t0)`。本包在训练时
从窗口**中途**的状态重新起算、再滚到末尾，要求与原始轨迹一致：

```text
loss_rollout = ‖Φ(start→start+k | trajectory[start]) − trajectory[start:start+k+1]‖²
```

它训练的是"从任意中间状态出发都滚得对"，直接对应长周期影子运行。该惩罚
不需要额外数据，完全在当前窗口上计算。

**两个必须注意的实现细节**（都是踩过坑之后定下来的）：

1. **锚点必须加扰动，否则惩罚恒为 0。** 确定性积分器从**精确**参考状态
   重新起算，必然精确复现原轨迹（半群性质自动成立），惩罚恒等于 0——它
   没有约束任何东西。第一版实现就是这样，消融实验里 A/B 两组数字逐位
   相同，才暴露出来。

   有意义的检验是：**从被重锚扰动过的起点能否回到原轨迹？** 这正是部署
   时的真实情形（状态每次化验都被加性偏差修正）。扰动幅度
   ``rollout_anchor_noise_c_a``（默认 0.01 mol/L）应接近现场重锚的实际
   量级：太大变成"任意初值都能收敛"的过强约束，太小则梯度信号微弱。

2. **展开长度要截断。** 惩罚积分嵌套在正向积分里，若一直滚到窗口末尾，
   计算图大小近似翻倍，长窗口（40+ 步）在内存受限的机器上会
   ``MemoryError``。``rollout_max_steps``（默认 15）截断到约窗口一半，
   仍能检验扰动是否被吸回，而内存代价线性增长。

3. **权重必须归一化，否则要猜一个 1e3 量级的魔法数。** 原始惩罚量级约
   5e-6，而数据损失约 0.07；初版权重 0.1 只占主损失的 7e-6，完全不起
   作用。``rollout_penalty_normalize=True``（默认）把惩罚除以**本批轨迹
   自身的动态幅度**（相邻步变化的均方），使其与浓度同量纲、随工况自适应。
   此后权重可直接理解为"惩罚相对轨迹动态的比例"，跨装置可迁移。

**实测效果**（12 条运行、8 epoch、`stride=8`，合成数据）：

| `rollout_penalty_weight` | 开环灰箱 | 影子灰箱 | 影子机理 |
|---|---|---|---|
| 0（关闭） | 0.01202 | 0.01458 | 0.01142 |
| 0.1（默认，已归一化） | **0.00796** | **0.00962** | 0.00976 |

关闭时灰箱在影子运行上劣于机理基线（0.01458 vs 0.01142），这正是本模块
要解决的问题；开启后影子误差降到 0.00962，**首次优于机理基线**，且相对
关闭状态改善 34%。开环同步改善，说明它不是靠牺牲短期精度换长期稳定。

**实现约束**：时间循环需要 Python 整数边界，而 `keras.random.randint`
返回符号张量，无法用于 `range`。因此起算点在候选集中轮换（模块级游标），
既保留"每个 batch 看到不同起点"的增强效果，也不破坏图追踪。

### 6.2 域外渐近门控（`LongHorizonConfig.domain_gate`）

用归一化特征的**盒距离** `d = max_i |z_i|` 判断是否远离训练域，在
`box` 与 `box+taper` 之间用 smoothstep 平滑过渡（不用硬阈值——硬阈值会让
修正量阶跃，积分器抖动且梯度不连续）。

**两个必须按现场校准的点**（初版都设错了，机制形同虚设）：

1. **特征必须居中。** 浓度特征原本写成 `c_a / c_a_ref`，在域内恒约为
   +1，直接把 `max|z|` 抬到 1，掩盖其他维度的真实外推。改成
   `(c_a - c_a_ref)/c_a_ref` 后，域内各维都接近 0，距离才真实反映偏移。
2. **`domain_box` 必须匹配工艺范围。** 归一化温度按 20 K 缩放，初版
   `box=3.0` 意味着"温度偏离 60 K"才触发门控——连"进料温度高 15 K"这种
   明显外推都放行。现在默认 1.5（约 30 K），并明确要求按现场操作范围校准，
   而不是拍整数。

**关键的物理判断：两类修正的域外行为必须不同。**

| 修正项 | 域外行为 | 理由 |
|---|---|---|
| 速率修正 `δ_rate` | 衰减到 0 | 符号自由，域外无数据支持 |
| 慢变**残余**项 | 衰减到 0 | 表示清洗/换剂等事件，域外默认"未发生" |
| 慢变**主趋势** `δ_ua`、`δ_activity` | **保持（饱和），不衰减** | 结垢/老化在运行更久时本来就应该更大 |

若把主趋势也在域外回退到 0，等于假设老装置换热面自动变干净，会系统性
高估 UA、低估出口浓度——方向上危险。主趋势本身已由 `max_decay_log` 与
sigmoid 双重限幅，不存在无界外推，因此不需要门控。

### 6.3 自适应状态重锚（`reanchor.AdaptiveReanchor`）

替代 `clip(residual, ±max)` 的固定截断，分两层：

**第一层：稳健置信度门控**

- 用滑动 MAD（×1.4826）估计正常残差尺度，而非标准差——标准差的崩溃点
  是 0%，一个 10σ 坏值就能让它失效；MAD 的崩溃点是 50%；
- 幅度超过 `residual_gate_sigma` 倍时 smoothstep 降低置信度，到
  `+residual_gate_taper_sigma` 倍时降为 0；
- **连续确认**：持续超过 `drift_magnitude_sigma` 倍、连续
  `drift_confirm_count` 次，判定为真实漂移而非孤立坏值，恢复 0.8 置信度。

两个实现细节是踩过坑之后定下来的：

1. **尺度必须有下限。** 若残差几乎恒定（稳定的系统偏差），MAD≈0 会让尺度
   坍缩，正常残差被误判为十几σ异常，修正永久停止。"残差高度一致"的正确
   解读是"这是一个可信的稳定偏差"。
2. **确认计数必须与门控解耦。** 若用"门控已关闭"作为计数条件会形成自锁：
   门控关闭后修正被压到 0，残差不会因此变小，模型既不敢修正也认不出漂移。
   因此计数用**独立的幅度判据**（`drift_magnitude_sigma`）。

**第二层：指数平滑过渡**

```text
applied = α · confidence · residual     # α = reseed_smoothing, 默认 0.5
```

一次到位会让状态阶跃，在准刚性工况下可能触发积分器抖动或数值守护回退。

### 6.4 在线增量微调与安全熔断（`adaptation.OnlineAdapter`）

```python
adapter = OnlineAdapter(model, AdaptationPolicy())
adapter.observe(rolling_rmse)            # 每个评估周期记录
event = adapter.adapt(recent, guard)      # 必要时微调 + 熔断
adapter.state.history_frame()             # 审计表
```

三条安全约束：

1. **冻结白箱核心。** 只微调速率网与慢变网；`log_k_ref`、`log_ua`、
   `log_heat_release` 保持冻结。机理参数经过辨识与工程审核，让它们随
   在线残差漂移，等于让模型悄悄改写物理定律。
2. **guard 必须独立留出。** 熔断用 `guard` 数据集校验短期 RMSE；若它与
   `recent` 重叠，熔断就变成自证。
3. **回滚而非重建。** 保存/恢复修正层权重，不重建模型（重建会丢优化器
   状态）。稳定快照只在一次**成功的**微调后更新，避免把未验证的权重
   当成基准。

**一个实现陷阱**：修正层是 `Sequential`，权重在第一次前向之后才创建。
若在 `__init__` 就快照会得到空列表，之后 `_restore` 因权重数量变化而
拒绝回滚——安全熔断会在第一次真正需要它时失效。因此采用延迟快照。

## 7. 使用

```python
from industrial_greybox.synthetic_plant import simulate_campaign, CampaignSpec
from industrial_greybox.config import ModelConfig, DelayConfig
from industrial_greybox.pipeline import run_pipeline, format_pipeline_summary

dcs, lab = simulate_campaign(CampaignSpec(n_runs=12, n_steps=200, days_span=30))
result = run_pipeline(dcs, lab, config=ModelConfig(), delay=DelayConfig(), epochs=25, stride=4)
print(format_pipeline_summary(result))

# 写出影子日志
result.shadow_report  # ShadowReport
```

接入现场数据时替换 `dcs`/`lab` 即可，但必须先完成：

1. 列名映射到 `config.EXAMPLE_COLUMN_MAPPING` 约定；
2. 单位核对（尤其温标 K、停留时间 min、浓度 mol/L）；
3. 死时间 `DelayConfig` 由取样管线长度与分析方法周期确定，不能拍脑袋；
4. 清洗阈值由工艺/仪表规格给出，并写进数据字典；
5. 按 `run_id` 划分训练/验证/测试，**不要**按行随机拆分。

## 8. 已知局限

- **反应级数固定为一级**。若真实动力学是 `n 阶` 或存在副反应，必须改
  机理结构，网络不能“学会”反应级数。
- **夹套视为理想恒温边界**。真实夹套有冷却水流量、壁热惯性和滞后，
  应扩展为夹套能量衡算状态。
- **参数可辨识性未解决**。稳态数据上，UA 增大与反应热减小可以给出相同
  温度；`learn_heat_release` 默认关闭。要辨识这些参数，需要有针对性的
  动态实验（例如冷却阶跃），并由工程团队审核安全性。
- **慢特征依赖维护记录**。`run_age_days` 需要清洗/换剂日期；若用数据集
  相对时间代替，模型会把“数据窗口长度”误当成“装置老化”。
- **影子运行的缓冲区长度**必须大于死时间加化验间隔，否则会出现
  `stale_history`，此时残差不可用。
- **开环指标与影子指标可能给出相反的模型排序。** 开环只在训练窗口长度内
  滚动（例如 20 min），影子连续滚动上百分钟。本包的合成实验中，灰箱在
  20 min 开环上优于机理基线（0.0098 vs 0.0104），但在约 100 min 的影子
  滚动上更差（0.0148 vs 0.0110；8 epoch/stride=8 时为 0.0150 vs 0.0114，
  排序不变）。部署是连续滚动的，因此**选型应以影子指标为准**；开环指标
  只用于诊断动态结构是否学对。
- **同一进程连续训练多个模型会累积 TensorFlow 图。** `train_greybox`
  在构建模型前调用 `keras.backend.clear_session()`；若调用方仍持有旧模型
  并需要推理，应在清图后重建或先完成推理。

## 9. 常见故障与排查

| 现象 | 最可能的原因 | 处理 |
|---|---|---|
| 训练损失正常但预测浓度整体偏低 | 掩码没生效（见第 5 节），或温度通道压倒浓度 | 用 `masked_squared_error` 单测确认未监督点不贡献；检查 `channel_weights` |
| `nonfinite_steps` 持续增长 | 步长/子步与工况刚性不匹配 | 用 `explicit_stability_ratio` 算稳定比；改用 `imex` 或增加 `substeps` |
| `C_A` 指标全是 `nan` | 该切分没有化验监督点 | 检查死时间与化验间隔；`frame_to_windows` 的跳过计数 |
| `stale_history > 0` | 缓冲区短于死时间 + 化验间隔 | 增大 `ShadowRunner.horizon_min`，或缩短死时间 |
| 灰箱开环更好、影子更差 | 修正项在训练窗口外持续外推 | 见第 7 节“已知局限”；缩短重锚间隔或收紧 `max_log_correction` |
| 影子残差与过程时刻呈趋势 | 死时间估计错误/对齐方向反了 | 用 `diagnostics.residual_structure` 定量检查斜率 |
| 连续训练多个模型时报 `Arena alloc failed` | TensorFlow 图累积 | `train_greybox` 已清图；调用方避免同时持有多个待推理模型 |
| 参数学到 `k_ref`/`UA` 远离工艺范围 | 参数不可辨识，网络在补偿 | 用 `diagnostics.parameter_identifiability` 看条件数，用 `sensitivity_correlation_matrix` 看谁和谁共线；补充动态实验 |

## 10. 验证清单

- [ ] 单元测试通过：`python -m unittest discover -s industrial_greybox/tests -t .`
- [ ] `health_records` 中 `nonfinite_steps`、`clipped_steps` 全为 0；
- [ ] 灰箱与机理基线在同一损失、同一积分下比较；
- [ ] 域内与域外误差分别报告，附样本数；
- [ ] 化验残差无明显滞后结构（`residual_grey` 对 `process_time` 无趋势）；
- [ ] 影子日志的 `config_signature` 与模型版本一致；
- [ ] 参数（`k_ref`、`UA`）落在工艺可接受范围内；
- [ ] 已由工艺/仪表/安全人员审核，并保留回退到机理基线的开关。
