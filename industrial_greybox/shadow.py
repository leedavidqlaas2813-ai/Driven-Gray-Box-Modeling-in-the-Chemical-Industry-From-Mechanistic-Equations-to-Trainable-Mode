"""影子运行（Shadow Mode）：因果流式推理、残差诊断与滚动报表。

因果性约定（最重要）
--------------------
影子运行**只能使用当时已经到达的信息**：

1. DCS 采样时刻 ``t_k`` 到达后，才允许使用 ``t_k`` 的测量与控制量；
2. 化验结果在 ``result_time`` 到达，它代表 ``process_time = result_time -
   dead_time`` 时刻的反应器状态；残差必须与该**过程时刻**的历史预测比较；
3. 因此预测历史保存在 ``ShadowBuffer`` 中，化验到达时回查，而不是用当前
   预测去减迟到的化验值。

这一点与离线评估不同：离线评估可以整体对齐，影子运行不行。若未来信息
被混入，残差会变小、报表会变好看，但上线后立刻失效；这是工业软测量最
常见的“自欺”来源。

输出
----
``ShadowRunner.run`` 返回：

- ``log``：每个 DCS 采样一行，含机理基线、灰箱预测、修正量、现场真实值
  （若有）、在域标志、数值诊断标志；
- ``lab_events``：每个化验结果到达一行，含过程时刻、残差、死时间；
- ``report``：按日/运行/域内外的滚动误差表与参数摘要。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    COL_C_A,
    COL_CAMPAIGN_TIME,
    COL_C_A_ONLINE,
    COL_RUN_ID,
    COL_T_REACTOR,
    COL_TIME_MIN,
    CONTROL_COLUMNS,
    DelayConfig,
    ModelConfig,
    ShadowConfig,
)
from .delays import ShadowBuffer
from .reanchor import AdaptiveReanchor
from .reports import (
    control_bounds,
    domain_flags,
    format_table,
    metrics_by_group,
    metrics_by_horizon,
    regression_metrics,
)
from .slow_features import normalize_slow_features

#: 影子日志列（顺序固定，便于下游 BI/数据库导入）。
#:
#: ``in_domain`` 必须能由**日志自身**复核，因此控制量列也一并写出：
#: 否则把日志交给离线分析时，无法判断某个残差是否来自外推，也无法重现
#: 包络判断。控制量单位见 ``config.CONTROL_COLUMNS``。
LOG_COLUMNS: tuple[str, ...] = (
    "campaign_time_min",
    "run_id",
    "time_min",
    *CONTROL_COLUMNS,
    COL_T_REACTOR,
    "mechanistic_c_a",
    "mechanistic_temperature_K",
    "greybox_c_a",
    "greybox_temperature_K",
    "correction_c_a",
    "correction_temperature_K",
    "online_c_a",
    "lab_c_a",
    "truth_c_a",
    "residual_vs_lab",
    "residual_vs_truth",
    "in_domain",
    "nonfinite_step",
    "clipped_step",
    "stale_history",
    # 域内程度（1=域内，0=远离训练域）。修正网络在域外会平滑关闭，
    # 因此这一列直接告诉你"这条预测有多少依赖数据驱动部分"。
    "rate_gate",
    "drift_gate",
    "config_signature",
)


@dataclass
class ShadowReport:
    """影子运行报表集合。"""

    by_day: pd.DataFrame = field(default_factory=pd.DataFrame)
    by_run: pd.DataFrame = field(default_factory=pd.DataFrame)
    generalisation: pd.DataFrame = field(default_factory=pd.DataFrame)
    by_horizon: pd.DataFrame = field(default_factory=pd.DataFrame)
    lab_residual_summary: dict[str, float] = field(default_factory=dict)
    n_samples: int = 0
    n_lab_events: int = 0
    n_stale_history: int = 0
    n_nonfinite: int = 0
    n_clipped: int = 0

    def to_text(self) -> str:
        lines = [
            f"样本数 {self.n_samples}，化验事件 {self.n_lab_events}；"
            f"历史不足 {self.n_stale_history}，非有限步 {self.n_nonfinite}，"
            f"裁剪步 {self.n_clipped}",
            "",
            "[按日滚动误差]",
            format_table(self.by_day),
            "",
            "[按运行误差]",
            format_table(self.by_run),
            "",
            "[域内/域外泛化]",
            format_table(self.generalisation),
            "",
            "[按预测时长分箱（误差随时间增长是正常现象，关键是斜率）]",
            format_table(self.by_horizon),
            "",
            "[化验残差（灰箱 - 化验）]",
            json.dumps(
                {key: round(value, 4) for key, value in self.lab_residual_summary.items()},
                ensure_ascii=False,
            ),
        ]
        return "\n".join(lines)


class ShadowRunner:
    """对一条 DCS 流 + 化验流做因果推理与诊断。

    Parameters
    ----------
    model
        ``CSTRGreyBoxModel``（已训练）。
    config
        模型配置；日志写入其 ``signature()`` 以便追溯“哪个模型产生的行”。
    delay
        化验死时间配置。
    shadow
        影子运行配置（分组、输出目录、温度反馈等）。
    slow_reference
        ``(age_ref_days, feed_ref_m3)``，必须与训练时一致；否则慢特征
        缩放不一致，修正网络会输出错误趋势。
    training_bounds
        训练工况包络（``reports.control_bounds`` 的结果），用于域外标记。
    """

    def __init__(
        self,
        model: Any,
        config: ModelConfig,
        delay: DelayConfig,
        shadow: ShadowConfig | None = None,
        *,
        slow_reference: tuple[float, float],
        training_bounds: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.delay = delay
        self.shadow = shadow or ShadowConfig()
        self.slow_reference = slow_reference
        self.training_bounds = training_bounds or {}
        self.horizon_min = delay.total_dead_time_min + 3.0 * delay.lab_interval_min
        self._signature = config.signature()

    # ------------------------------------------------------------------
    # 流式推理
    # ------------------------------------------------------------------
    def run(
        self,
        dcs: pd.DataFrame,
        lab: pd.DataFrame,
        *,
        truth_column: str = "c_a_truth_mol_L",
    ) -> tuple[pd.DataFrame, pd.DataFrame, ShadowReport]:
        """执行影子运行。

        Parameters
        ----------
        dcs
            按时序到达的 DCS 长表，至少含 ``run_id``、``time_min``、
            ``CONTROL_COLUMNS``、``T_REACTOR``。可含 ``campaign_time_min``
            与 ``c_a_online_mol_L``。
        lab
            化验结果表，``time_min`` 表示**结果到达**时刻，``c_a_out_mol_L``
            为化验值。可含额外真值列。允许为空 DataFrame（纯预测模式）。
        truth_column
            **独立**真值列（例如实验室复核值或合成数据的真值），用于诊断
            ``residual_vs_truth``。默认 ``"c_a_truth_mol_L"``：只有合成
            数据或带独立复核的现场数据才有这一列；现场若没有，日志中该
            列与 ``residual_vs_truth`` 会是 NaN——这是**正确**行为，不应拿
            ``lab_c_a`` 顶替，否则会把“与化验的差”伪装成“与真值的差”。

        Returns
        -------
        (log, lab_events, report)
        """
        self._validate(dcs, lab)
        dcs = self._prepare_dcs(dcs)
        run_order = list(dict.fromkeys(dcs[COL_RUN_ID].astype(str)))
        lab_by_run = {
            str(run_id): group.sort_values(COL_TIME_MIN)
            for run_id, group in lab.groupby(COL_RUN_ID, sort=False)
        }
        lab_cursor = {run_id: 0 for run_id in run_order}

        grey_buffer = ShadowBuffer(self.horizon_min, interpolation="zoh")
        mech_buffer = ShadowBuffer(self.horizon_min, interpolation="zoh")

        log_rows: list[dict[str, Any]] = []
        lab_rows: list[dict[str, Any]] = []
        n_stale = 0
        n_nonfinite = 0
        n_clipped = 0

        for run_id in run_order:
            run_frame = dcs[dcs[COL_RUN_ID].astype(str) == run_id].sort_values(COL_TIME_MIN)
            run_lab = lab_by_run.get(run_id, pd.DataFrame())
            grey_buffer = ShadowBuffer(self.horizon_min, interpolation="zoh")
            mech_buffer = ShadowBuffer(self.horizon_min, interpolation="zoh")
            # 每条运行独立的自适应重锚器：稳健尺度与连续计数不能跨运行
            # 复用，否则一个运行的坏值会污染另一个运行的置信度估计。
            reanchor = AdaptiveReanchor(self.shadow)

            grey_state: np.ndarray | None = None
            mech_state: np.ndarray | None = None
            previous_control: np.ndarray | None = None
            previous_time: float | None = None
            lab_index = 0

            times = run_frame[COL_TIME_MIN].to_numpy(dtype=np.float64)
            controls_array = run_frame[list(CONTROL_COLUMNS)].to_numpy(dtype=np.float64)
            temperatures = run_frame[COL_T_REACTOR].to_numpy(dtype=np.float64)
            online = (
                run_frame[COL_C_A_ONLINE].to_numpy(dtype=np.float64)
                if COL_C_A_ONLINE in run_frame.columns
                else np.full(len(run_frame), np.nan)
            )
            truths = (
                run_frame[truth_column].to_numpy(dtype=np.float64)
                if truth_column in run_frame.columns
                else np.full(len(run_frame), np.nan)
            )
            slow_raw = run_frame[
                ["run_age_days", "cumulative_feed_m3", "ema_temperature_K"]
            ].to_numpy(dtype=np.float64) if {
                "run_age_days", "cumulative_feed_m3", "ema_temperature_K"
            }.issubset(run_frame.columns) else self._fallback_slow(run_frame)
            slow_scaled = normalize_slow_features(
                slow_raw,
                age_ref_days=self.slow_reference[0],
                feed_ref_m3=self.slow_reference[1],
            )
            campaign_times = (
                run_frame[COL_CAMPAIGN_TIME].to_numpy(dtype=np.float64)
                if COL_CAMPAIGN_TIME in run_frame.columns
                else times
            )

            for step, time_min in enumerate(times):
                # 1) 处理在本采样时刻之前到达的化验结果。
                while lab_index < len(run_lab):
                    lab_row = run_lab.iloc[lab_index]
                    result_time = float(lab_row[COL_TIME_MIN])
                    if result_time > time_min:
                        break
                    process_time = result_time - self.delay.total_dead_time_min
                    lab_value = float(lab_row[COL_C_A])
                    predicted_grey = grey_buffer.query(process_time)
                    predicted_mech = mech_buffer.query(process_time)
                    stale = predicted_grey is None
                    if stale:
                        n_stale += 1
                    lab_rows.append(
                        {
                            "campaign_time_min": float(
                                campaign_times[step] - (time_min - result_time)
                            ),
                            "run_id": run_id,
                            "result_time_min": result_time,
                            "process_time_min": process_time,
                            "dead_time_min": self.delay.total_dead_time_min,
                            "lab_c_a": lab_value,
                            "greybox_c_a_then": (
                                float(predicted_grey[0]) if predicted_grey is not None else np.nan
                            ),
                            "mechanistic_c_a_then": (
                                float(predicted_mech[0]) if predicted_mech is not None else np.nan
                            ),
                            "residual_grey": (
                                lab_value - float(predicted_grey[0])
                                if predicted_grey is not None
                                else np.nan
                            ),
                            "residual_mechanistic": (
                                lab_value - float(predicted_mech[0])
                                if predicted_mech is not None
                                else np.nan
                            ),
                            "stale_history": bool(stale),
                        }
                    )
                    # 状态重锚（软测量更新）。
                    #
                    # **不能**把 lab_value 直接写进当前状态：化验值对应的是
                    # ``process_time``，而当前状态已经推进到 ``time_min``，
                    # 中间隔着一整个死时间。正确做法是把“过程时刻的模型
                    # 误差”作为一个缓慢变化的加性偏差，施加到当前状态上。
                    #
                    # 修正量的计算交给 ``AdaptiveReanchor``：它用稳健残差
                    # 尺度区分“仪表坏值”与“真实工况漂移”，并用指数平滑
                    # 过渡替代瞬时跳变（避免刚性工况下积分器抖动）。
                    if (
                        self.shadow.reseed_on_lab_result
                        and grey_state is not None
                        and predicted_grey is not None
                    ):
                        decision = reanchor.update(lab_value - float(predicted_grey[0]))
                        if decision.applied_bias_mol_l != 0.0:
                            grey_state = grey_state.copy()
                            grey_state[0, 0] = grey_state[0, 0] + decision.applied_bias_mol_l
                            if mech_state is not None:
                                mech_state = mech_state.copy()
                                mech_state[0, 0] = (
                                    mech_state[0, 0] + decision.applied_bias_mol_l
                                )
                        lab_rows[-1].update(
                            {
                                "applied_bias_mol_l": decision.applied_bias_mol_l,
                                "bias_clipped": decision.clipped,
                                "reanchor_confidence": decision.confidence,
                                "reanchor_scale_mol_l": decision.scale_mol_l,
                                "drift_confirmed": decision.drift_confirmed,
                                "reseed_mode": decision.mode,
                            }
                        )
                    lab_index += 1

                # 2) 状态初始化：首点必须来自可用的测量/配置，不能“预知”未来。
                if grey_state is None:
                    initial_c_a = self._initial_c_a(online[step])
                    grey_state = np.array(
                        [[initial_c_a, temperatures[step]]], dtype=np.float32
                    )
                    mech_state = grey_state.copy()
                elif previous_control is not None:
                    # 3) 用上一步控制量推进到当前时刻（ZOH）。
                    control_batch = previous_control.reshape(1, -1).astype(np.float32)
                    slow_batch = slow_scaled[step - 1 : step].astype(np.float32)
                    grey_state, grey_diag = self._advance(
                        grey_state, control_batch, slow_batch, use_corrections=True
                    )
                    mech_state, mech_diag = self._advance(
                        mech_state, control_batch, slow_batch, use_corrections=False
                    )
                    n_nonfinite += int(self._scalar(grey_diag[0])) + int(
                        self._scalar(mech_diag[0])
                    )
                    n_clipped += int(self._scalar(grey_diag[1])) + int(
                        self._scalar(mech_diag[1])
                    )
                    # 4) 温度状态反馈：温度通常高频可测，用测量重置模型温度，
                    #    防止温度积分误差在长时间运行中累积。
                    if self.shadow.use_temperature_feedback:
                        measured = float(temperatures[step])
                        if np.isfinite(measured):
                            grey_state = grey_state.copy()
                            mech_state = mech_state.copy()
                            grey_state[0, 1] = measured
                            mech_state[0, 1] = measured

                # 5) 记录当前时刻的预测，供未来到达的化验回查。
                grey_value = float(grey_state[0, 0])
                grey_temp = float(grey_state[0, 1])
                mech_value = float(mech_state[0, 0])
                mech_temp = float(mech_state[0, 1])
                grey_buffer.push(time_min, [grey_value, grey_temp])
                mech_buffer.push(time_min, [mech_value, mech_temp])

                # 6) 与在此时刻的化验对齐值比较（若有）。这里只在掩码为真
                #    时给出 lab_c_a，避免把“最近一次化验”当成当前真值。
                aligned_lab = np.nan
                if "c_a_aligned" in run_frame.columns and "c_a_mask" in run_frame.columns:
                    if float(run_frame["c_a_mask"].iloc[step]) > 0:
                        aligned_lab = float(run_frame["c_a_aligned"].iloc[step])

                truth_value = float(truths[step])
                # 域内程度：只用当前工况（状态+控制+慢特征）计算，不读未来。
                domain_status = self._domain_status(
                    grey_state, controls_array[step], slow_scaled[min(step, len(slow_scaled) - 1)]
                )
                log_rows.append(
                    {
                        "campaign_time_min": float(campaign_times[step]),
                        "run_id": run_id,
                        "time_min": float(time_min),
                        # 控制量必须写进日志：域内/域外判断要在离线复核时
                        # 能由日志本身重算，而不是依赖内部包络对象。
                        **{
                            column: float(controls_array[step, index])
                            for index, column in enumerate(CONTROL_COLUMNS)
                        },
                        COL_T_REACTOR: float(temperatures[step]),
                        "mechanistic_c_a": mech_value,
                        "mechanistic_temperature_K": mech_temp,
                        "greybox_c_a": grey_value,
                        "greybox_temperature_K": grey_temp,
                        "correction_c_a": grey_value - mech_value,
                        "correction_temperature_K": grey_temp - mech_temp,
                        "online_c_a": float(online[step]),
                        "lab_c_a": aligned_lab,
                        "truth_c_a": truth_value,
                        "residual_vs_lab": (
                            grey_value - aligned_lab
                            if np.isfinite(aligned_lab)
                            else np.nan
                        ),
                        "residual_vs_truth": (
                            grey_value - truth_value
                            if np.isfinite(truth_value)
                            else np.nan
                        ),
                        "in_domain": True,  # 稍后由控制量包络统一填充
                        "nonfinite_step": False,
                        "clipped_step": False,
                        "stale_history": False,
                        "rate_gate": domain_status.get("rate_gate", np.nan),
                        "drift_gate": domain_status.get("drift_gate", np.nan),
                        "config_signature": self._signature,
                    }
                )
                previous_control = controls_array[step]
                previous_time = time_min

            # 运行结束后，把剩余到达的化验也处理掉（result_time 可能晚于
            # 最后一个 DCS 点；此时仍可用缓冲区回查历史预测）。
            while lab_index < len(run_lab):
                lab_row = run_lab.iloc[lab_index]
                result_time = float(lab_row[COL_TIME_MIN])
                process_time = result_time - self.delay.total_dead_time_min
                lab_value = float(lab_row[COL_C_A])
                predicted_grey = grey_buffer.query(process_time)
                predicted_mech = mech_buffer.query(process_time)
                stale = predicted_grey is None
                if stale:
                    n_stale += 1
                lab_rows.append(
                    {
                        "campaign_time_min": float(campaign_times[-1] + (result_time - times[-1])),
                        "run_id": run_id,
                        "result_time_min": result_time,
                        "process_time_min": process_time,
                        "dead_time_min": self.delay.total_dead_time_min,
                        "lab_c_a": lab_value,
                        "greybox_c_a_then": (
                            float(predicted_grey[0]) if predicted_grey is not None else np.nan
                        ),
                        "mechanistic_c_a_then": (
                            float(predicted_mech[0]) if predicted_mech is not None else np.nan
                        ),
                        "residual_grey": (
                            lab_value - float(predicted_grey[0])
                            if predicted_grey is not None
                            else np.nan
                        ),
                        "residual_mechanistic": (
                            lab_value - float(predicted_mech[0])
                            if predicted_mech is not None
                            else np.nan
                        ),
                        "stale_history": bool(stale),
                        # 运行结束后到达的化验不施加修正（状态已冻结），
                        # 但仍写全字段，保证日志列一致、可直接合并。
                        "applied_bias_mol_l": 0.0,
                        "bias_clipped": False,
                        "reanchor_confidence": np.nan,
                        "reanchor_scale_mol_l": np.nan,
                        "drift_confirmed": False,
                        "reseed_mode": self.shadow.reseed_mode,
                    }
                )
                lab_index += 1

        log = pd.DataFrame(log_rows, columns=list(LOG_COLUMNS))
        lab_events = pd.DataFrame(lab_rows)
        if not log.empty and self.training_bounds:
            log["in_domain"] = domain_flags(
                log, bounds=self.training_bounds, margin=self.shadow.domain_margin
            ).to_numpy()
        # 数值诊断：非有限值已由守护层回退，但报表中单独计数。
        if not log.empty:
            for column in ("greybox_c_a", "greybox_temperature_K", "mechanistic_c_a"):
                log[f"{column}_finite"] = np.isfinite(log[column])
        report = self._build_report(log, lab_events)
        # 数值守护计数必须真正进入报表；否则报表永远显示 0，等于把
        # “积分发过散”这一关键信息静默丢掉。
        report.n_nonfinite = n_nonfinite
        report.n_clipped = n_clipped
        return log, lab_events, report

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    def _domain_status(
        self, state: np.ndarray | None, control: np.ndarray, slow: np.ndarray
    ) -> dict[str, float]:
        """每条采样点的域内程度（失败时返回 NaN，绝不阻断影子运行）。

        这是**诊断信息**，不是安全联锁：即使门控计算失败，也应继续预测并
        记 NaN，而不是让整条运行中断。
        """
        if state is None:
            return {"rate_gate": np.nan, "drift_gate": np.nan}
        try:
            from keras import ops

            layer = self.model.physics_layer
            return layer.domain_status(
                ops.convert_to_tensor(state.reshape(1, -1), dtype="float32"),
                ops.convert_to_tensor(control.reshape(1, -1), dtype="float32"),
                ops.convert_to_tensor(slow.reshape(1, -1), dtype="float32"),
            )
        except Exception:  # noqa: BLE001 - 诊断失败不应影响预测
            return {"rate_gate": np.nan, "drift_gate": np.nan}

    def _advance(
        self, state: np.ndarray, control: np.ndarray, slow: np.ndarray, *, use_corrections: bool
    ) -> tuple[np.ndarray, tuple[Any, Any, Any]]:
        from keras import ops

        layer = self.model.physics_layer
        next_state, diagnostics = layer.step(
            ops.convert_to_tensor(state, dtype="float32"),
            ops.convert_to_tensor(control, dtype="float32"),
            ops.convert_to_tensor(slow, dtype="float32"),
            use_corrections=use_corrections,
        )
        return np.asarray(next_state, dtype=np.float64), diagnostics

    @staticmethod
    def _scalar(value: Any) -> float:
        import numpy as np

        return float(np.asarray(value).reshape(-1)[0]) if np.asarray(value).size else 0.0

    def _initial_c_a(self, online_value: float) -> float:
        """首点 C_A 的获取顺序：在线分析仪 → 配置初值 → 化验前不预测。

        绝不使用该时刻之后的化验值。若三者都没有，用在线分析仪或配置值；
        仍没有则报错，要求工艺提供初始估计（否则影子预测起点的误差会被
        误读为模型偏差）。
        """
        if np.isfinite(online_value):
            return float(online_value)
        if self.shadow.initial_c_a_mol_l is not None:
            return float(self.shadow.initial_c_a_mol_l)
        raise ValueError(
            "无法确定 C_A 初值：请提供在线分析仪列、"
            "ShadowConfig.initial_c_a_mol_l，或在首条化验到达后再开始预测。"
        )

    def _fallback_slow(self, frame: pd.DataFrame) -> np.ndarray:
        """缺少慢特征列时，用中性值并显式警告。

        中性值 = 参考尺度的一半，意味着“中等老化”；这是**占位**而非
        估计。生产使用必须从维护记录/流量计生成慢特征，否则慢变修正
        会输出无依据的趋势。
        """
        import warnings

        warnings.warn(
            "DCS 缺少 run_age_days/cumulative_feed_m3/ema_temperature_K，"
            "影子运行将使用中性慢特征占位；生产使用必须提供真实慢特征。",
            RuntimeWarning,
            stacklevel=2,
        )
        age = 0.5 * self.slow_reference[0]
        feed = 0.5 * self.slow_reference[1]
        temperature = frame[COL_T_REACTOR].to_numpy(dtype=np.float64)
        return np.column_stack(
            [
                np.full(len(frame), age),
                np.full(len(frame), feed),
                pd.Series(temperature).ewm(alpha=0.05, adjust=False).mean().to_numpy(),
            ]
        )

    def _prepare_dcs(self, dcs: pd.DataFrame) -> pd.DataFrame:
        frame = dcs.copy()
        frame[COL_RUN_ID] = frame[COL_RUN_ID].astype(str)
        frame = frame.sort_values([COL_RUN_ID, COL_TIME_MIN]).reset_index(drop=True)
        # 统一时间轴：若没有 campaign 绝对时间，用每条运行的起始顺序拼接。
        if COL_CAMPAIGN_TIME not in frame.columns:
            offsets = {}
            cursor = 0.0
            for run_id, group in frame.groupby(COL_RUN_ID, sort=False):
                offsets[run_id] = cursor
                cursor += float(group[COL_TIME_MIN].max()) + self.config.physics.dt_min
            frame[COL_CAMPAIGN_TIME] = frame.apply(
                lambda row: row[COL_TIME_MIN] + offsets[str(row[COL_RUN_ID])], axis=1
            )
        return frame

    def _validate(self, dcs: pd.DataFrame, lab: pd.DataFrame) -> None:
        required = {COL_RUN_ID, COL_TIME_MIN, COL_T_REACTOR, *CONTROL_COLUMNS}
        missing = required - set(dcs.columns)
        if missing:
            raise ValueError(f"DCS 流缺少列：{sorted(missing)}")
        if dcs.empty:
            raise ValueError("DCS 流为空。")
        if not lab.empty:
            missing_lab = {COL_RUN_ID, COL_TIME_MIN, COL_C_A} - set(lab.columns)
            if missing_lab:
                raise ValueError(f"化验流缺少列：{sorted(missing_lab)}")
            if (lab[COL_C_A] < 0).any():
                raise ValueError("化验值含负浓度；先确认单位/检出限处理。")

    def _build_report(self, log: pd.DataFrame, lab_events: pd.DataFrame) -> ShadowReport:
        if log.empty:
            return ShadowReport()
        frame = log.copy()
        # 报表使用化验对齐点计算质量误差（不与在线仪表的噪声混淆）。
        frame["_has_lab"] = frame["lab_c_a"].notna()
        frame["_day"] = np.floor(frame["campaign_time_min"] / (24 * 60)).astype(int)

        by_run = metrics_by_group(
            frame,
            prediction_column="greybox_c_a",
            truth_column="lab_c_a",
            mask_column=None,
            group_columns=["run_id"],
        )
        # 仅化验点参与质量统计。
        lab_only = frame[frame["_has_lab"]]
        by_day = (
            metrics_by_group(
                lab_only,
                prediction_column="greybox_c_a",
                truth_column="lab_c_a",
                group_columns=["_day"],
            )
            if not lab_only.empty
            else pd.DataFrame()
        )
        # 机理基线并列，便于看出灰箱修正是否真的改善。
        if not lab_only.empty:
            mech = regression_metrics(
                lab_only["mechanistic_c_a"].to_numpy(),
                lab_only["lab_c_a"].to_numpy(),
            )
            grey = regression_metrics(
                lab_only["greybox_c_a"].to_numpy(), lab_only["lab_c_a"].to_numpy()
            )
            generalisation = pd.DataFrame(
                [
                    {"模型": "机理基线", **mech},
                    {"模型": "灰箱", **grey},
                ]
            )
        else:
            generalisation = pd.DataFrame()

        if not lab_events.empty:
            residual_summary = {
                "n": float(len(lab_events)),
                "bias_grey": float(np.nanmean(lab_events["residual_grey"])),
                "rmse_grey": float(
                    np.sqrt(np.nanmean(np.square(lab_events["residual_grey"])))
                ),
                "bias_mechanistic": float(np.nanmean(lab_events["residual_mechanistic"])),
                "rmse_mechanistic": float(
                    np.sqrt(np.nanmean(np.square(lab_events["residual_mechanistic"])))
                ),
            }
        else:
            residual_summary = {}

        # 按预测时长分箱：同一模型在 20 min 开环与 100 min 滚动上可能排名
        # 相反。只报总 RMSE 会把“短期好、长期漂”和“长期稳”混为一谈。
        by_horizon = (
            metrics_by_horizon(
                lab_only,
                prediction_columns={
                    "灰箱": "greybox_c_a",
                    "机理基线": "mechanistic_c_a",
                },
                truth_column="lab_c_a",
                time_column="time_min",
                run_column="run_id",
                bins_min=(0.0, 25.0, 50.0, 100.0, 200.0, 400.0),
            )
            if not lab_only.empty
            else pd.DataFrame()
        )

        return ShadowReport(
            by_day=by_day,
            by_run=by_run,
            generalisation=generalisation,
            by_horizon=by_horizon,
            lab_residual_summary=residual_summary,
            n_samples=len(log),
            n_lab_events=len(lab_events),
            n_stale_history=int(lab_events["stale_history"].sum())
            if not lab_events.empty
            else 0,
        )

    # ------------------------------------------------------------------
    # 落盘
    # ------------------------------------------------------------------
    def write_outputs(
        self,
        log: pd.DataFrame,
        lab_events: pd.DataFrame,
        report: ShadowReport,
        *,
        output_dir: Path | None = None,
        prefix: str = "shadow",
    ) -> dict[str, Path]:
        """写出诊断 CSV 与报表文本，返回路径字典。

        文件名包含配置指纹，避免不同模型版本的日志互相覆盖；现场部署时
        还应写入模型版本、数据窗口起止与操作员信息。
        """
        directory = Path(output_dir or self.shadow.output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        stem = f"{prefix}_{self._signature}"
        paths = {
            "log": directory / f"{stem}_log.csv",
            "lab_events": directory / f"{stem}_lab_events.csv",
            "report": directory / f"{stem}_report.txt",
        }
        log.to_csv(paths["log"], index=False, encoding="utf-8-sig")
        lab_events.to_csv(paths["lab_events"], index=False, encoding="utf-8-sig")
        paths["report"].write_text(report.to_text(), encoding="utf-8")
        return paths


def build_training_bounds(frame: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """从训练集计算控制变量包络（供域外标记）。"""
    return control_bounds(frame, [*CONTROL_COLUMNS, COL_T_REACTOR])


__all__ = [
    "LOG_COLUMNS",
    "ShadowReport",
    "ShadowRunner",
    "build_training_bounds",
]
