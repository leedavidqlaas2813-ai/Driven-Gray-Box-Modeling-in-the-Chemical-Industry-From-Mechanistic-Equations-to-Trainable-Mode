"""从工业长表构建训练张量：按运行切分、时滞对齐、监督掩码。

张量约定
--------
每条运行被切成若干连续窗口（``n_steps`` 个控制间隔 + 1 个初始状态）::

    initial_states : (N, 2)            [C_A, T] 的观测首点
    controls       : (N, n_steps, 4)   [C_A0, T_in, T_c, tau]
    slow           : (N, n_steps, 3)   已缩放的慢特征
    targets        : (N, n_steps+1, 2) [C_A, T]
    weights        : (N, n_steps+1, 2) 0/1 监督掩码：C_A 仅在化验点为 1，
                                         温度全程为 1

为什么要窗口化而不是“一条运行一个样本”
----------------------------------------
一条 8 小时运行若作为一个样本，batch 里只有十几条轨迹，梯度噪声很大，
且一次反向传播要展开几百步。窗口化让训练可以按批次打乱，同时**同一
运行的不同窗口绝不跨训练/测试集**（先按 run_id 切分，再窗口化）。

为什么把 C_A 损失掩码化
------------------------
化验是稀疏的（例如每 10–30 分钟一次），中间时刻没有真值。若把零阶保持
的阶梯当作目标，模型会学习去拟合“取样时刻的冻结值”，产生系统性偏差。
温度通常高频可测，因此全程参与损失。两个通道**在损失内各自归一化**，
温度项不会因为采样点更多而压倒浓度项；通道间的相对重要性由
``ModelConfig.channel_weights`` 表达，而不是逐点权重。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .config import (
    COL_RUN_ID,
    COL_T_REACTOR,
    COL_TIME_MIN,
    CONTROL_COLUMNS,
    PhysicsConfig,
)
from .slow_features import normalize_slow_features


@dataclass(frozen=True)
class WindowedDataset:
    """一个数据集切分的所有窗口张量。"""

    initial_states: np.ndarray
    controls: np.ndarray
    slow: np.ndarray
    targets: np.ndarray
    weights: np.ndarray
    run_ids: np.ndarray
    window_start_times: np.ndarray

    def __len__(self) -> int:
        return int(self.initial_states.shape[0])

    def as_model_inputs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.initial_states, self.controls, self.slow

    def subset(self, indices: np.ndarray) -> "WindowedDataset":
        return WindowedDataset(
            initial_states=self.initial_states[indices],
            controls=self.controls[indices],
            slow=self.slow[indices],
            targets=self.targets[indices],
            weights=self.weights[indices],
            run_ids=self.run_ids[indices],
            window_start_times=self.window_start_times[indices],
        )

    def concat(self, other: "WindowedDataset") -> "WindowedDataset":
        return WindowedDataset(
            initial_states=np.concatenate([self.initial_states, other.initial_states]),
            controls=np.concatenate([self.controls, other.controls]),
            slow=np.concatenate([self.slow, other.slow]),
            targets=np.concatenate([self.targets, other.targets]),
            weights=np.concatenate([self.weights, other.weights]),
            run_ids=np.concatenate([self.run_ids, other.run_ids]),
            window_start_times=np.concatenate(
                [self.window_start_times, other.window_start_times]
            ),
        )


def split_runs(
    frame: pd.DataFrame,
    *,
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 7,
    run_column: str = COL_RUN_ID,
    require_validation: bool = True,
) -> dict[str, list[str]]:
    """按运行 ID 划分训练/验证/测试，绝不切割单条运行。

    这是工业验证的**最低要求**：相邻时间点强相关，按行随机拆分会让同一
    运行的两端分别进入训练和测试，测得的泛化误差会严重乐观。

    ``fractions`` 顺序为 (train, validation, test)。若运行数太少导致
    验证集为空且 ``require_validation=True``，直接报错而不是悄悄退化。
    """
    if len(fractions) != 3 or not np.isclose(sum(fractions), 1.0):
        raise ValueError("fractions 必须是三者之和为 1 的元组。")
    run_ids = np.asarray(frame[run_column].drop_duplicates())
    if len(run_ids) < 3:
        raise ValueError("至少需要 3 条独立运行才能划分训练/验证/测试。")
    order = np.random.default_rng(seed).permutation(run_ids)
    n_train = max(1, int(round(len(order) * fractions[0])))
    n_val = int(round(len(order) * fractions[1]))
    n_train = min(n_train, len(order) - 2)
    n_val = max(n_val, 1) if require_validation else n_val
    n_test = len(order) - n_train - n_val
    if n_test < 1:
        raise ValueError(
            f"划分后测试集为空（train={n_train}, val={n_val}, runs={len(order)}）。"
        )
    groups = {
        "train": order[:n_train],
        "validation": order[n_train : n_train + n_val],
        "test": order[n_train + n_val :],
    }
    return {name: [str(item) for item in ids] for name, ids in groups.items()}


def frame_to_windows(
    frame: pd.DataFrame,
    *,
    physics: PhysicsConfig,
    slow_reference: tuple[float, float],
    stride: int = 1,
    temperature_weight: float = 1.0,
    aligned_column: str = "c_a_aligned",
    mask_column: str = "c_a_mask",
    lab_time_column: str = "lab_process_time_min",
    max_anchor_age_min: float | None = None,
    require_dt_uniform: bool = True,
) -> WindowedDataset:
    """把长表切成训练窗口。

    Parameters
    ----------
    frame
        已与化验结果对齐（``delays.align_lab_results``）并清洗后的长表，
        必须含 ``CONTROL_COLUMNS``、``T_REACTOR``、慢特征三列，以及
        ``aligned_column``/``mask_column``。**喂进来的应当是 ZOH 对齐值
        加掩码**，不是把稀疏化验先 reindex 再用 ffill 成“密集真值”。
    physics
        提供 ``n_steps``、``dt_min``、``temperature_scale_k``。
    slow_reference
        ``(age_ref_days, feed_ref_m3)``，用于有界缩放。
    stride
        窗口起点步长。``1`` 表示最大重叠；生产训练可用 2–4 降低相关性。
    max_anchor_age_min
        窗口初值允许使用的“最近一条化验”的最大年龄。``None`` 时自动取
        **半个化验间隔**：既允许窗口从化验点稍微错开（否则 ``stride`` 稍大
        就可能一个窗口都取不到），又不让初值滞后超过一个化验周期的一半。
        生产使用应以化验周期的实际分布重新定这个上限；若已有高频在线
        分析仪，可放宽到 1–2 个化验周期。

    Notes
    -----
    初值来自 ZOH 对齐值：``age=0`` 表示起点恰为化验点，``age>0`` 表示
    使用最近一次化验。计算损失时，初值点不计入浓度监督（权重为 0），
    因此不会把“滞后的初值”当作真值去拟合。
    """
    if stride < 1:
        raise ValueError("stride 必须 >= 1。")
    if max_anchor_age_min is not None and max_anchor_age_min < 0:
        raise ValueError("max_anchor_age_min 不能为负。")
    required = {
        *CONTROL_COLUMNS,
        COL_T_REACTOR,
        COL_TIME_MIN,
        COL_RUN_ID,
        aligned_column,
        mask_column,
        "run_age_days",
        "cumulative_feed_m3",
        "ema_temperature_K",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"缺少构建窗口所需列：{sorted(missing)}")
    if (frame[aligned_column] < 0).any(skipna=True):
        raise ValueError("对齐后的浓度含负值；请先确认传感器/单位问题。")

    n_steps = physics.n_steps
    dt = physics.dt_min
    scale_t = physics.temperature_scale_k
    t_ref = physics.t_ref_k
    age_ref, feed_ref = slow_reference

    initial_states: list[np.ndarray] = []
    controls_out: list[np.ndarray] = []
    slow_out: list[np.ndarray] = []
    targets_out: list[np.ndarray] = []
    weights_out: list[np.ndarray] = []
    run_ids_out: list[str] = []
    start_times: list[float] = []
    skipped_short = 0
    skipped_no_anchor = 0
    skipped_gaps = 0

    for run_id, group in frame.groupby(COL_RUN_ID, sort=False):
        group = group.sort_values(COL_TIME_MIN)
        times = group[COL_TIME_MIN].to_numpy(dtype=np.float64)
        if len(times) < n_steps + 1:
            skipped_short += 1
            continue
        grid_steps = np.diff(times) / dt
        if require_dt_uniform and np.any(np.abs(grid_steps - 1.0) > 1e-3):
            bad = float(np.median(grid_steps[np.abs(grid_steps - 1.0) > 1e-3]))
            raise ValueError(
                f"运行 {run_id} 的采样间隔不是 {dt} min（中位偏差 {bad:.3f} 倍）；"
                "请先重采样到统一网格。"
            )

        controls = group[list(CONTROL_COLUMNS)].to_numpy(dtype=np.float32)
        temperature = group[COL_T_REACTOR].to_numpy(dtype=np.float32)
        concentration = group[aligned_column].to_numpy(dtype=np.float64)
        concentration_mask = group[mask_column].to_numpy(dtype=np.float64)
        slow_raw = group[
            ["run_age_days", "cumulative_feed_m3", "ema_temperature_K"]
        ].to_numpy(dtype=np.float64)
        slow_scaled = normalize_slow_features(
            slow_raw, age_ref_days=age_ref, feed_ref_m3=feed_ref
        )

        last_start = len(times) - n_steps - 1
        # ``lab_process_time_min`` 只在被吸附的化验行有值；对齐值是零阶保持，
        # 因此这里要前向填充，得到“当前 ZOH 值对应的那条化验的过程时刻”。
        lab_time_series = pd.Series(
            group[lab_time_column].to_numpy(dtype=np.float64)
            if lab_time_column in group.columns
            else np.full(len(times), np.nan)
        )
        anchor_source_times = lab_time_series.ffill().to_numpy()
        finite_lab_times = lab_time_series.dropna().to_numpy()
        if len(finite_lab_times) >= 2:
            lab_interval = float(np.median(np.diff(np.unique(finite_lab_times))))
        else:
            lab_interval = float(n_steps * dt)
        anchor_limit = (
            max_anchor_age_min
            if max_anchor_age_min is not None
            else 0.5 * lab_interval
        )
        for start in range(0, last_start + 1, stride):
            stop = start + n_steps + 1
            window_temp = temperature[start:stop]
            if not np.all(np.isfinite(window_temp)):
                # 温度通道全程监督；含 NaN 的窗口丢弃比插值更诚实
                # （插值会伪造动态响应）。
                skipped_gaps += 1
                continue
            window_slow = slow_scaled[start:stop]
            if not np.all(np.isfinite(window_slow)):
                skipped_gaps += 1
                continue
            window_controls = controls[start : start + n_steps]
            if not np.all(np.isfinite(window_controls)):
                skipped_gaps += 1
                continue
            # 浓度初值必须有化验支撑，且不能比 anchor_limit 更旧。用“产生该
            # ZOH 值的那条化验的过程时刻”计算年龄，而不是用窗口起点本身
            # 是否恰好落在化验行上——否则窗口几乎无法错开。
            anchor_value = concentration[start]
            anchor_age = times[start] - anchor_source_times[start]
            if (
                not np.isfinite(anchor_value)
                or not np.isfinite(anchor_age)
                or anchor_age > anchor_limit + 1e-9
                or anchor_age < -1e-9
            ):
                skipped_no_anchor += 1
                continue

            window_concentration = concentration[start:stop]
            window_conc_mask = concentration_mask[start:stop]
            targets = np.stack(
                [window_concentration, (window_temp - t_ref) / scale_t],
                axis=-1,
            ).astype(np.float32)
            weights = np.zeros_like(targets)
            # 浓度只在真实化验点参与损失；NaN 一律视为“无监督”。
            weights[:, 0] = np.where(
                np.isfinite(window_concentration) & (window_conc_mask > 0),
                1.0,
                0.0,
            ).astype(np.float32)
            # 初值是 ZOH 锚点（可能滞后 age 分钟），不作为浓度真值。
            weights[0, 0] = 0.0
            weights[:, 1] = temperature_weight
            targets[:, 0] = np.where(
                weights[:, 0] > 0, window_concentration, 0.0
            ).astype(np.float32)

            initial_states.append(
                np.array([anchor_value, window_temp[0]], dtype=np.float32)
            )
            controls_out.append(window_controls)
            slow_out.append(window_slow[:n_steps])
            targets_out.append(targets)
            weights_out.append(weights)
            run_ids_out.append(str(run_id))
            start_times.append(float(times[start]))

    if not initial_states:
        raise ValueError(
            "没有可用窗口：检查运行长度、缺失值和 n_steps="
            f"{n_steps}（过短运行 {skipped_short} 条，缺口 {skipped_gaps} 个，"
            f"缺初值锚点 {skipped_no_anchor} 个）。"
        )

    return WindowedDataset(
        initial_states=np.stack(initial_states).astype(np.float32),
        controls=np.stack(controls_out).astype(np.float32),
        slow=np.stack(slow_out).astype(np.float32),
        targets=np.stack(targets_out).astype(np.float32),
        weights=np.stack(weights_out).astype(np.float32),
        run_ids=np.asarray(run_ids_out),
        window_start_times=np.asarray(start_times, dtype=np.float32),
    )


def build_datasets(
    frame: pd.DataFrame,
    *,
    physics: PhysicsConfig,
    slow_reference: tuple[float, float],
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 7,
    stride: int = 1,
    temperature_weight: float = 1.0,
    aligned_column: str = "c_a_aligned",
    mask_column: str = "c_a_mask",
) -> tuple[dict[str, WindowedDataset], dict[str, list[str]]]:
    """切分运行并分别为每个切分构建窗口。

    先切分运行、再调用 ``frame_to_windows``：即使同一运行的时间戳在全局
    上连续，也不会因为窗口跨边界而泄漏。
    """
    groups = split_runs(frame, fractions=fractions, seed=seed)
    datasets: dict[str, WindowedDataset] = {}
    for split_name, run_ids in groups.items():
        subset = frame[frame[COL_RUN_ID].astype(str).isin(run_ids)]
        datasets[split_name] = frame_to_windows(
            subset,
            physics=physics,
            slow_reference=slow_reference,
            stride=stride,
            temperature_weight=temperature_weight,
            aligned_column=aligned_column,
            mask_column=mask_column,
        )
    return datasets, groups


def dataset_train_arrays(
    dataset: WindowedDataset,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray, np.ndarray]:
    """转成 ``fit`` 需要的 ``(x, y, sample_weight)``。

    用法：``model.fit(*dataset_train_arrays(dataset))``——即
    ``model.fit(x, y, sample_weight=weights)``。权重是**逐元素**的
    ``(N, T+1, C)`` 掩码，由 ``CSTRGreyBoxModel.compute_loss`` 消费。

    .. warning::
       不要把权重塞进 ``y``（例如 ``y=(targets, weights)``）：Keras 3 会
       把它当作多输出结构，与单输出预测不匹配并报错。必须用
       ``sample_weight=`` 关键字传入。
    """
    return (
        dataset.as_model_inputs(),
        dataset.targets,
        dataset.weights,
    )


def check_no_run_leakage(
    datasets: dict[str, WindowedDataset], groups: dict[str, Sequence[str]]
) -> None:
    """断言切分之间没有共享运行 ID（防御性检查，可放进单元测试）。"""
    for name, dataset in datasets.items():
        observed = set(np.unique(dataset.run_ids).tolist())
        expected = set(map(str, groups[name]))
        if observed - expected:
            raise AssertionError(
                f"{name} 中出现不属于该切分的运行：{sorted(observed - expected)}"
            )
    names = list(groups)
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            overlap = set(groups[first]) & set(groups[second])
            if overlap:
                raise AssertionError(f"{first}/{second} 运行泄漏：{sorted(overlap)}")


def stack_window_datasets(parts: Iterable[WindowedDataset]) -> WindowedDataset:
    parts = list(parts)
    if not parts:
        raise ValueError("至少需要一个数据集。")
    out = parts[0]
    for part in parts[1:]:
        out = out.concat(part)
    return out
