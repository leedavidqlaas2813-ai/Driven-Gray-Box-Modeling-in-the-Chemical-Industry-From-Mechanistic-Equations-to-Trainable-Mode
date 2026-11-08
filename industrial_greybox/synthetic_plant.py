"""可复现的“合成装置”数据工厂：带漂移、时滞与仪表缺陷的 campaign。

用途与边界
----------
本模块生成**教学/回归测试用**的合成 DCS + 化验数据，用于验证：

- 积分器在准刚性工况下不发散；
- 时滞对齐与监督掩码确实只使用“当时可用”的化验值；
- 清洗流水线能识别毛刺、突变、死区与越界；
- 慢变结垢/活性修正能在长 campaign 中恢复漂移；
- 影子运行报表能给出按日/按运行的滚动指标。

**它不能替代真实装置验证。** 合成数据的噪声是独立同分布的正态量，真实
工厂还有偏差、漂移、串扰、量程压缩、通讯丢包和多速率采样；这些必须用
现场数据评估。

隐藏真值故意与名义机理不同
--------------------------
- 真活化能 48 kJ/mol ≠ 名义 42 kJ/mol；
- 速率含弱温度周期项，模拟结构误差；
- 结垢使 UA 随时间指数衰减（清洗后恢复）；
- 催化剂活性随累计进料下降；
- 首末稳态的净反应热与名义值不同（总括热效应）。

这些偏差让“纯机理”必然有系统误差，从而可以观察灰箱修正是否真的有用，
而不是在完全匹配的仿真上自证。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import (
    COL_C_A,
    COL_C_A0,
    COL_C_A_ONLINE,
    COL_CAMPAIGN_TIME,
    COL_CUM_FEED,
    COL_EMA_T,
    COL_RUN_AGE,
    COL_RUN_ID,
    COL_T_C,
    COL_T_IN,
    COL_T_REACTOR,
    COL_TAU,
    COL_TIME_MIN,
    PhysicsConfig,
)

R_GAS = 8.314
T_REF = 350.0


@dataclass(frozen=True)
class PlantTruth:
    """合成装置的隐藏真值（不进入模型输入）。"""

    activation_energy_j_mol: float = 48_000.0
    heat_release_k_l_mol: float = 24.0
    ua_clean_per_min: float = 0.10
    ua_fouling_per_day: float = 0.004
    k_ref_clean_per_min: float = 0.032
    activity_loss_per_1000_m3: float = 0.03
    rate_ripple_amplitude: float = 0.10
    rate_ripple_period_k: float = 10.0
    throughput_l_per_min: float = 10.0


@dataclass(frozen=True)
class CampaignSpec:
    """campaign 生成参数。"""

    n_runs: int = 12
    days_span: float = 60.0
    dt_min: float = 0.5
    n_steps: int = 480  # 240 min / run
    seed: int = 2026
    #: 化验间隔 [min]；同时决定监督点密度。
    lab_interval_min: float = 10.0
    #: 化验结果的报告延迟（相对过程时刻）[min]。
    lab_delay_min: float = 12.0
    noise_c_a: float = 0.006
    noise_temperature: float = 0.12
    #: 是否注入坏数据（毛刺/跳变/死区/越界）。
    inject_faults: bool = True
    #: 高温外推 campaign（True 时整体提升进料/冷却温度）。
    hot_shift: bool = False


def _hall_rhs(
    c_a: np.ndarray,
    temperature: np.ndarray,
    *,
    c_a0: np.ndarray,
    t_in: np.ndarray,
    t_c: np.ndarray,
    tau: np.ndarray,
    truth: PlantTruth,
    k_scale: np.ndarray,
    ua: np.ndarray,
    heat_release: float,
) -> tuple[np.ndarray, np.ndarray]:
    k = truth.k_ref_clean_per_min * np.exp(
        -truth.activation_energy_j_mol / R_GAS * (1 / temperature - 1 / T_REF)
    )
    k *= 1 + truth.rate_ripple_amplitude * np.sin(
        (temperature - T_REF) / truth.rate_ripple_period_k
    )
    rate = k_scale * k * c_a
    d_ca = (c_a0 - c_a) / tau - rate
    d_t = (t_in - temperature) / tau + heat_release * rate + ua * (t_c - temperature)
    return d_ca, d_t


def simulate_campaign(
    spec: CampaignSpec | None = None,
    truth: PlantTruth | None = None,
    physics: PhysicsConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """返回 ``(dcs, lab)`` 两张长表。

    - ``dcs``：每个采样点一行，含操作量、釜温、真实出口浓度（用于评估）
      以及带噪声的仪表读数；带故障时含毛刺。
    - ``lab``：稀疏化验表，列含 ``run_id``、``time_min``、``c_a_out_mol_L``。
      ``time_min`` 表示**结果到达**时间（= 过程时刻 + 报告延迟），符合
      真实 LIMS 导出习惯；对齐时必须用 ``timestamp_semantics='result'``。
    """
    spec = spec or CampaignSpec()
    truth = truth or PlantTruth()
    physics = physics or PhysicsConfig()
    rng = np.random.default_rng(spec.seed)
    dt = spec.dt_min
    n_steps = spec.n_steps
    times = np.arange(n_steps + 1) * dt

    # 把 runs 均匀铺在 days_span 上，每个 run 之间有停车间隔（工艺假设：
    # 间歇取样/换批，运行之间的时间不参与轨迹内部积分）。
    run_starts_min = np.linspace(0.0, spec.days_span * 24 * 60, spec.n_runs, endpoint=False)
    dcs_rows: list[dict[str, float | str]] = []
    lab_rows: list[dict[str, float | str]] = []

    for run_index in range(spec.n_runs):
        start_min = run_starts_min[run_index]
        age_days = start_min / (24 * 60)
        # 每 30 天清洗一次换热器：UA 恢复，活性不回（换剂另计）。
        days_since_clean = age_days % 30.0
        ua = truth.ua_clean_per_min * np.exp(-truth.ua_fouling_per_day * days_since_clean)
        cumulative_feed = start_min / (24 * 60) * 24 * truth.throughput_l_per_min / 1000
        activity = np.exp(-truth.activity_loss_per_1000_m3 * cumulative_feed / 1000)

        base_c_a0 = rng.uniform(0.9, 1.3)
        base_t_in = rng.uniform(338.0, 352.0)
        base_t_c = rng.uniform(288.0, 302.0)
        base_tau = rng.uniform(3.5, 7.5)
        if spec.hot_shift:
            base_t_in += 14.0
            base_t_c += 14.0

        controls = np.empty((n_steps + 1, 4))
        controls[:, 0] = base_c_a0
        controls[:, 1] = base_t_in
        controls[:, 2] = base_t_c
        controls[:, 3] = base_tau
        # 两次设定值变化，模拟操作调整；变化点固定以避免“未来信息”泄漏。
        for change in (n_steps // 3, 2 * n_steps // 3):
            controls[change:, 0] = np.clip(base_c_a0 + rng.normal(0, 0.06), 0.8, 1.4)
            controls[change:, 1] = base_t_in + rng.normal(0, 3.0)
            controls[change:, 2] = base_t_c + rng.normal(0, 2.0)
            controls[change:, 3] = np.clip(base_tau + rng.normal(0, 0.6), 2.5, 9.0)

        # 真值积分：小步长 RK4，作为“装置”参考解。
        c_a = np.empty(n_steps + 1)
        temperature = np.empty(n_steps + 1)
        c_a[0] = controls[0, 0] * 0.75
        temperature[0] = 0.5 * (controls[0, 1] + 355.0)
        heat_release = truth.heat_release_k_l_mol * (
            1.0 + 0.15 * np.sin(run_index / 3.0)
        )
        for step in range(n_steps):
            c_a0, t_in, t_c, tau = controls[step]
            k_scale = activity
            # 装置侧的 RK4 用 4 个子步，远小于模型层的步长。
            sub = 4
            h = dt / sub
            x_c, x_t = c_a[step], temperature[step]
            for _ in range(sub):
                d1c, d1t = _hall_rhs(
                    x_c, x_t, c_a0=c_a0, t_in=t_in, t_c=t_c, tau=tau,
                    truth=truth, k_scale=k_scale, ua=ua, heat_release=heat_release,
                )
                d2c, d2t = _hall_rhs(
                    x_c + 0.5 * h * d1c, x_t + 0.5 * h * d1t,
                    c_a0=c_a0, t_in=t_in, t_c=t_c, tau=tau,
                    truth=truth, k_scale=k_scale, ua=ua, heat_release=heat_release,
                )
                d3c, d3t = _hall_rhs(
                    x_c + 0.5 * h * d2c, x_t + 0.5 * h * d2t,
                    c_a0=c_a0, t_in=t_in, t_c=t_c, tau=tau,
                    truth=truth, k_scale=k_scale, ua=ua, heat_release=heat_release,
                )
                d4c, d4t = _hall_rhs(
                    x_c + h * d3c, x_t + h * d3t,
                    c_a0=c_a0, t_in=t_in, t_c=t_c, tau=tau,
                    truth=truth, k_scale=k_scale, ua=ua, heat_release=heat_release,
                )
                x_c += h / 6 * (d1c + 2 * d2c + 2 * d3c + d4c)
                x_t += h / 6 * (d1t + 2 * d2t + 2 * d3t + d4t)
            c_a[step + 1] = max(x_c, 0.0)
            temperature[step + 1] = x_t

        # 仪表读数：温度高频可测，浓度在线分析有噪声（仅用于 DCS 展示，
        # 训练真值来自稀疏化验）。
        measured_t = temperature + rng.normal(0, spec.noise_temperature, size=len(times))
        measured_ca = c_a + rng.normal(0, spec.noise_c_a, size=len(times))

        ema_alpha = 1 - np.exp(-dt / 60.0)
        ema_t = pd.Series(measured_t).ewm(alpha=ema_alpha, adjust=False).mean().to_numpy()

        run_id = f"RUN-{run_index:03d}"
        for step, time_min in enumerate(times):
            row = {
                COL_RUN_ID: run_id,
                COL_TIME_MIN: float(time_min),
                COL_CAMPAIGN_TIME: float(start_min + time_min),
                COL_C_A0: float(controls[step, 0]),
                COL_T_IN: float(controls[step, 1]),
                COL_T_C: float(controls[step, 2]),
                COL_TAU: float(controls[step, 3]),
                COL_T_REACTOR: float(measured_t[step]),
                COL_C_A_ONLINE: float(measured_ca[step]),
                COL_RUN_AGE: float(age_days),
                COL_CUM_FEED: float(start_min / (24 * 60) * 24 * truth.throughput_l_per_min / 1000),
                COL_EMA_T: float(ema_t[step]),
                # 以下列只用于诊断/评估，不进入模型输入。
                "c_a_truth_mol_L": float(c_a[step]),
                "temperature_truth_K": float(temperature[step]),
                "ua_truth_per_min": float(ua),
                "activity_truth": float(activity),
            }
            dcs_rows.append(row)

        # 稀疏化验：过程时刻每隔 lab_interval，结果在 process+delay 到达。
        lab_process_times = np.arange(15.0, times[-1], spec.lab_interval_min)
        for process_time in lab_process_times:
            index = int(round(process_time / dt))
            if index > n_steps:
                continue
            value = c_a[index] + rng.normal(0, spec.noise_c_a * 1.5)
            lab_rows.append(
                {
                    COL_RUN_ID: run_id,
                    "time_min": float(process_time + spec.lab_delay_min),
                    COL_C_A: float(max(value, 0.0)),
                    "process_time_min": float(process_time),
                }
            )

    dcs = pd.DataFrame(dcs_rows)
    lab = pd.DataFrame(lab_rows).rename(
        columns={"process_time_min": "process_time_min_truth"}
    )
    if spec.inject_faults:
        dcs = _inject_faults(dcs, rng, fault_rate=0.004)
    return dcs, lab


def _inject_faults(
    frame: pd.DataFrame, rng: np.random.Generator, fault_rate: float
) -> pd.DataFrame:
    """注入典型仪表缺陷：孤立毛刺、阶跃跳变、死区卡值、越界。

    故障只注入到**测量列**（``T_REACTOR``、``C_A_ONLINE``、``C_A0``），
    不动 ``*_truth`` 列；这样清洗效果可以用真值量化，而不是靠肉眼判断。
    质量化验表（``lab``）不加故障：真实 LIMS 数据的问题类型不同（复核、
    稀释、检出限），需要单独的数据字典。
    """
    frame = frame.copy()
    for _, group in frame.groupby(COL_RUN_ID, sort=False):
        index = group.index.to_numpy()
        n = len(index)
        if n < 12:
            continue

        # 1) 孤立毛刺：温度 +8~20 K，浓度 +0.4 mol/L。
        for _ in range(max(1, int(n * fault_rate))):
            position = int(rng.integers(5, n - 5))
            if rng.random() < 0.5:
                frame.loc[index[position], COL_T_REACTOR] += rng.uniform(8, 20)
            else:
                frame.loc[index[position], COL_C_A_ONLINE] += rng.uniform(0.3, 0.6)

        # 2) 死区：把一段 40–80 个点（20–40 min）的在线分析仪读数冻结。
        #    注意：设定值（C_A0）本身就会长时间保持不变，不能拿它测死区，
        #    否则会把正常工艺保持误判成仪表故障。
        if n > 120:
            start = int(rng.integers(10, n - 90))
            length = int(rng.integers(40, 80))
            frame.loc[index[start : start + length], COL_C_A_ONLINE] = frame.loc[
                index[start], COL_C_A_ONLINE
            ]

        # 3) 越界：负浓度、超高温。
        if n > 40:
            position = int(rng.integers(5, n - 5))
            frame.loc[index[position], COL_T_REACTOR] = rng.uniform(560, 700)
            frame.loc[index[position], COL_C_A_ONLINE] = rng.uniform(5.5, 8.0)

        # 4) 阶跃跳变：连续两个点温度突变。
        if n > 60:
            start = int(rng.integers(10, n - 10))
            frame.loc[index[start], COL_T_REACTOR] += rng.uniform(12, 25)
            frame.loc[index[start + 1], COL_T_REACTOR] += rng.uniform(12, 25)
    return frame


def simulate_stiff_run(
    *,
    dt_min: float = 1.0,
    tau_min: float = 0.35,
    ua_per_min: float = 0.6,
    n_steps: int = 60,
    c_a0: float = 1.2,
    t_in: float = 350.0,
    t_c: float = 290.0,
) -> tuple[np.ndarray, np.ndarray]:
    """生成一个准刚性工况，用于积分器稳定性对比测试。

    停留时间 0.35 min、传热 0.6 min⁻¹ 时，``h a_max ≈ 0.95``，显式
    Euler 在 `dt=1 min` 下接近稳定边界；把 ``dt`` 提到 2 min 会越过
    边界。返回 ``(times, reference)``，参考解用 64 子步 RK4 生成。
    """
    controls = np.tile(np.array([c_a0, t_in, t_c, tau_min]), (n_steps, 1))
    state = np.array([c_a0 * 0.8, 355.0])
    reference = [state.copy()]
    # 参考解：把每步细分为 64 个 RK4 子步。
    sub = 64
    h = dt_min / sub
    for step in range(n_steps):
        c_a0_s, t_in_s, t_c_s, tau_s = controls[step]
        x_c, x_t = state
        for _ in range(sub):
            k1c, k1t = _rhs_simple(x_c, x_t, c_a0_s, t_in_s, t_c_s, tau_s)
            k2c, k2t = _rhs_simple(
                x_c + 0.5 * h * k1c, x_t + 0.5 * h * k1t, c_a0_s, t_in_s, t_c_s, tau_s
            )
            k3c, k3t = _rhs_simple(
                x_c + 0.5 * h * k2c, x_t + 0.5 * h * k2t, c_a0_s, t_in_s, t_c_s, tau_s
            )
            k4c, k4t = _rhs_simple(
                x_c + h * k3c, x_t + h * k3t, c_a0_s, t_in_s, t_c_s, tau_s
            )
            x_c += h / 6 * (k1c + 2 * k2c + 2 * k3c + k4c)
            x_t += h / 6 * (k1t + 2 * k2t + 2 * k3t + k4t)
        state = np.array([max(x_c, 0.0), x_t])
        reference.append(state.copy())
    times = np.arange(n_steps + 1) * dt_min
    return times, np.asarray(reference)


def _rhs_simple(
    c_a: float, temperature: float, c_a0: float, t_in: float, t_c: float, tau: float
) -> tuple[float, float]:
    k = 0.032 * np.exp(-48_000 / R_GAS * (1 / temperature - 1 / T_REF))
    rate = k * c_a
    d_ca = (c_a0 - c_a) / tau - rate
    d_t = (t_in - temperature) / tau + 24.0 * rate + 0.10 * (t_c - temperature)
    return d_ca, d_t


__all__ = [
    "CampaignSpec",
    "PlantTruth",
    "simulate_campaign",
    "simulate_stiff_run",
]
