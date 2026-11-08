"""工业历史数据清洗流水线（PI/PHD/DCS 长表）。

设计立场
--------
清洗的目标不是“让损失下降”，而是**让错误可见且可追溯**。因此：

1. 每条规则只**标记**原始问题，不静默修改；被删除的点保留原因。
2. 质量目标（化验 ``C_A``）默认不插值：伪造化验值等于伪造标签。
3. 插值只用于操作变量/温度等连续量，且限长；超过限长的缺口保留 NaN，
   由上层决定丢弃整条运行或要求人工/工艺处理。
4. 所有阈值都来自工艺知识或仪表规格，不来自测试集统计。默认阈值只适用
   于本教程的合成工况，接入现场必须重写并记录审核人。

规则顺序（顺序会影响标记结果）
------------------------------
``硬边界 → Hampel 毛刺 → 变化率/跳变 → 死区（假恒定） → 限长插值``。

先做硬边界，避免非物理的 ±1e9 毛刺把变化率/中位数统计拉偏。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from .config import (
    COL_RUN_ID,
    COL_TIME_MIN,
    HygieneConfig,
)

#: 标记列后缀。``<column>__flag`` 保存命中规则的名称（分号分隔），
#: ``<column>__valid`` 是该值的最终可用性（False 表示已是 NaN）。
FLAG_SUFFIX = "__flag"
VALID_SUFFIX = "__valid"

REASON_LIMIT = "limit"
REASON_SPIKE = "spike"
REASON_RATE = "rate_of_change"
REASON_STEP = "step_jump"
REASON_DEADBAND = "deadband"
REASON_NONFINITE = "nonfinite"
REASON_IMPUTED = "imputed"
REASON_MISSING = "missing"


@dataclass
class HygieneReport:
    """清洗结果摘要，可写入模型版本/数据质量报表。"""

    n_rows: int
    n_runs: int
    counts_by_rule: dict[str, int] = field(default_factory=dict)
    counts_by_column: dict[str, int] = field(default_factory=dict)
    imputed_points: int = 0
    informative_flags: int = 0
    unresolved_nan: dict[str, int] = field(default_factory=dict)
    run_summaries: pd.DataFrame = field(default_factory=pd.DataFrame)

    #: 这些原因只标记、不删除，属于“供人工复核的信息”，不计入可疑值。
    INFORMATIVE_REASONS: tuple[str, ...] = (REASON_DEADBAND, REASON_IMPUTED)

    def to_frame(self) -> pd.DataFrame:
        rows = [
            {"类别": "规则", "项目": rule, "次数": count}
            for rule, count in sorted(self.counts_by_rule.items())
        ]
        rows += [
            {"类别": "列", "项目": column, "次数": count}
            for column, count in sorted(self.counts_by_column.items())
        ]
        rows += [
            {"类别": "未解决 NaN", "项目": column, "次数": count}
            for column, count in sorted(self.unresolved_nan.items())
        ]
        rows.append(
            {"类别": "插值点", "项目": "全部", "次数": self.imputed_points}
        )
        rows.append(
            {"类别": "信息性标记", "项目": "死区/插值", "次数": self.informative_flags}
        )
        return pd.DataFrame(rows)

    def summary_line(self) -> str:
        suspicious = sum(
            count
            for rule, count in self.counts_by_rule.items()
            if rule not in self.INFORMATIVE_REASONS
        )
        return (
            f"{self.n_rows} 行 / {self.n_runs} 条运行；"
            f"可疑值（已删除/待处理）{suspicious} 个，"
            f"插值 {self.imputed_points} 个点，"
            f"死区等信息性标记 {self.informative_flags} 个。"
        )


class IndustrialHygienePipeline:
    """按运行清洗工业时序长表。

    输入要求
    --------
    - 至少包含 ``run_id`` 和 ``time_min``；
    - 每条运行内 ``time_min`` 严格递增（重复时间戳直接报错，不做自动
      去重，因为哪一条“正确”需要工艺判断）；
    - 其余列按 ``HygieneConfig`` 中出现的规则处理。
    """

    def __init__(self, config: HygieneConfig | None = None) -> None:
        self.config = config or HygieneConfig()

    # ------------------------------------------------------------------
    # 公共入口
    # ------------------------------------------------------------------
    def clean(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, HygieneReport]:
        """返回 ``(cleaned, report)``。

        ``cleaned`` 保留原始数值列名，增加 ``<col>__flag`` 与
        ``<col>__valid``；原始输入不被修改。
        """
        self._validate_columns(frame)
        work = frame.copy()
        work[COL_TIME_MIN] = pd.to_numeric(work[COL_TIME_MIN], errors="coerce")
        if work[COL_TIME_MIN].isna().any():
            raise ValueError("time_min 含不可解析值；先修正时间戳再清洗。")
        # 重复时间戳必须拒绝：窗口化、化验对齐和 ZOH 都假定每个时刻唯一。
        # 单条运行的**行序**则允许乱序（DCS 导出常按通道而非时间排列），
        # 排序后使用；乱序本身不改变数据内容，不应报错。
        duplicates = work.groupby(COL_RUN_ID)[COL_TIME_MIN].apply(
            lambda series: int(series.duplicated().sum())
        )
        offending = duplicates[duplicates > 0]
        if not offending.empty:
            raise ValueError(
                "以下运行存在重复时间戳，需人工确认后再清洗："
                + ", ".join(f"{run}={count}" for run, count in offending.items())
            )
        work = work.sort_values([COL_RUN_ID, COL_TIME_MIN], kind="stable").reset_index(drop=True)

        touched = {
            rule.column
            for rule in (
                *self.config.limits,
                *self.config.rate_rules,
                *self.config.deadband_rules,
                *self.config.spike_rules,
            )
        }
        # 插值列也需要标记/可用性列，否则报表无法区分“原本缺失”与“被插值”。
        touched.update(self.config.impute_columns)
        for column in touched:
            if column not in work.columns:
                continue
            work[column] = pd.to_numeric(work[column], errors="coerce")
            work[column + FLAG_SUFFIX] = ""
            work[column + VALID_SUFFIX] = work[column].notna()

        runs = [
            self._clean_run(group).set_index(group.index)
            for _, group in work.groupby(COL_RUN_ID, sort=False)
        ]
        cleaned = pd.concat(runs).sort_index()
        cleaned = cleaned.sort_values([COL_RUN_ID, COL_TIME_MIN], kind="stable").reset_index(drop=True)
        report = self._build_report(cleaned)
        return cleaned, report

    # ------------------------------------------------------------------
    # 单运行清洗
    # ------------------------------------------------------------------
    def _clean_run(self, run: pd.DataFrame) -> pd.DataFrame:
        run = run.copy()
        # 该函数假定调用方已排序且已通过重复时间戳检查；这里只在单条运行
        # 内部再做一次排序，保证插值/差分的时间方向正确。
        run = run.sort_values(COL_TIME_MIN, kind="stable")
        self._apply_limits(run)
        self._apply_spikes(run)
        self._apply_rate_rules(run)
        self._apply_deadband(run)
        self._apply_imputation(run)
        self._finalize_validity(run)
        return run

    # ------------------------------------------------------------------
    # 规则实现
    # ------------------------------------------------------------------
    def _flag(self, run: pd.DataFrame, column: str, mask: np.ndarray, reason: str) -> None:
        """记录原因；掩码为 True 的行追加该原因（保持顺序稳定）。"""
        if not mask.any():
            return
        flag_column = column + FLAG_SUFFIX
        current = run.loc[mask, flag_column]
        run.loc[mask, flag_column] = current.map(
            lambda text: f"{text};{reason}" if text else reason
        )

    def _apply_limits(self, run: pd.DataFrame) -> None:
        """硬边界：``nan`` 删除、``clip`` 截断、``flag`` 只标记。"""
        for rule in self._rules_for(run, self.config.limits):
            values = run[rule.column]
            outside = (values < rule.lower) | (values > rule.upper)
            outside = outside.fillna(False).to_numpy()
            if not outside.any():
                continue
            self._flag(run, rule.column, outside, REASON_LIMIT)
            if rule.action == "nan":
                run.loc[outside, rule.column] = np.nan
            elif rule.action == "clip":
                run[rule.column] = values.clip(rule.lower, rule.upper)
            # action == 'flag' 只标记，不修改数值。

    def _apply_spikes(self, run: pd.DataFrame) -> None:
        """Hampel 规则：偏离滑动中位数超过 ``n_sigma * 1.4826 * MAD``。"""
        for rule in self._rules_for(run, self.config.spike_rules):
            values = run[rule.column].to_numpy(dtype=np.float64)
            window = rule.window_samples
            half = window // 2
            padded = np.pad(values, half, mode="edge")
            spikes = np.zeros(len(values), dtype=bool)
            for index in range(len(values)):
                local = padded[index : index + window]
                local = local[np.isfinite(local)]
                if local.size < 3:
                    continue
                center = np.median(local)
                mad = np.median(np.abs(local - center))
                # 1.4826 使 MAD 在正态噪声下与标准差一致。
                scale = 1.4826 * mad
                if scale <= 0:
                    # 常数窗口：任何偏离都可能是量化噪声，不用 Hampel 判定。
                    continue
                if abs(values[index] - center) > rule.n_sigma * scale:
                    spikes[index] = True
            self._flag(run, rule.column, spikes, REASON_SPIKE)
            run.loc[spikes, rule.column] = np.nan

    def _apply_rate_rules(self, run: pd.DataFrame) -> None:
        """变化率/阶跃规则。

        方向固定为 **backward**（比较当前点与上一点），不做“两侧都看”的
        对称判断：对称判断会让一个真实阶跃把它前后的正常点也标成异常，
        在小批量数据上尤其明显。首个采样点没有前驱，不参与判断。

        默认 ``action='flag'``：真实工况切换本来就会触发这类规则，只提醒
        工程师复核，不删除数据。
        """
        times = run[COL_TIME_MIN].to_numpy(dtype=np.float64)
        dt = np.diff(times)
        for rule in self._rules_for(run, self.config.rate_rules):
            values = run[rule.column].to_numpy(dtype=np.float64)
            flags = np.zeros(len(values), dtype=bool)
            if len(values) < 2:
                continue
            delta = np.diff(values)
            with np.errstate(invalid="ignore", divide="ignore"):
                slope = np.where(dt > 0, delta / dt, np.inf)
            too_fast = np.abs(slope) > rule.max_abs_per_min
            flags[1:] |= too_fast
            if rule.max_abs_jump is not None:
                flags[1:] |= np.abs(delta) > rule.max_abs_jump
            self._flag(run, rule.column, flags, REASON_RATE)
            if rule.action == "nan":
                run.loc[flags, rule.column] = np.nan

    def _apply_deadband(self, run: pd.DataFrame) -> None:
        """滑动窗口方差识别“假恒定”。

        判定条件（同时满足）：

        1. 窗口 ``window_min`` 内标准差 <= ``std_tol`` 且极差 <= ``span_tol``；
        2. 从该点向前连续保持同一低方差状态已超过 ``min_duration_min``。

        条件 2 避免把一次短时保持判成死区。命中时默认只标记（``action='flag'``），
        并在报表中记为“信息性标记”，不计入“可疑值/已删除值”的计数——
        真实稳态本来就会触发，它需要仪表工程师复核，而不是自动删。
        """
        times = run[COL_TIME_MIN].to_numpy(dtype=np.float64)
        for rule in self._rules_for(run, self.config.deadband_rules):
            values = run[rule.column].to_numpy(dtype=np.float64)
            flags = np.zeros(len(values), dtype=bool)
            quiet = np.zeros(len(values), dtype=bool)
            for index in range(len(values)):
                in_window = (times >= times[index] - rule.window_min) & (
                    times <= times[index]
                )
                local = values[in_window]
                local = local[np.isfinite(local)]
                if local.size < 3:
                    continue
                std = float(np.std(local))
                span = float(np.max(local) - np.min(local))
                quiet[index] = std <= rule.std_tol and span <= rule.span_tol

            # 连续 quiet 段长度（按时间而非点数）：只有持续超过
            # min_duration_min 的段才判定为死区。
            start = 0
            while start < len(quiet):
                if not quiet[start]:
                    start += 1
                    continue
                stop = start
                while stop + 1 < len(quiet) and quiet[stop + 1]:
                    stop += 1
                duration = times[stop] - times[start]
                if duration >= rule.min_duration_min:
                    flags[start : stop + 1] = True
                start = stop + 1

            self._flag(run, rule.column, flags, REASON_DEADBAND)
            if rule.action == "nan":
                run.loc[flags, rule.column] = np.nan

    def _apply_imputation(self, run: pd.DataFrame) -> None:
        """只对连续量做限长线性插值；质量目标列默认排除。

        **只填补原本缺失的点，绝不填补规则删除的点。** 被硬边界/毛刺规则
        置为 NaN 的值，是因为它们“物理上不可信”；插值把它们换成一个
        看起来合理的中间值，等于让清洗白做，并在报表里掩盖问题。这类点
        必须保留 NaN 或整段丢弃，由上层明确决策。
        """
        limit = self.config.max_interpolate_gap
        if limit <= 0:
            return
        for column in self.config.impute_columns:
            if column not in run.columns:
                continue
            series = run[column]
            missing = series.isna()
            if not missing.any() or missing.all():
                continue
            flagged = run[column + FLAG_SUFFIX].fillna("").astype(str)
            # 被规则命中的点不插值。
            rule_flagged = flagged != ""
            imputable_mask = missing & ~rule_flagged
            if not imputable_mask.any():
                continue
            # 连续缺失段长度：只有整段 <= limit 才插值。段内若含规则命中点，
            # 整段都不插值（避免“用坏点两侧的值造中间值”）。
            groups = (missing != missing.shift()).cumsum()
            gap_length = missing.groupby(groups).transform("sum")
            gap_has_flag = rule_flagged.groupby(groups).transform("any")
            imputable = imputable_mask & (gap_length <= limit) & ~gap_has_flag
            if not imputable.any():
                continue
            filled = series.interpolate(method="linear", limit_direction="both")
            run.loc[imputable, column] = filled[imputable]
            self._flag(run, column, imputable.to_numpy(), REASON_IMPUTED)

    def _finalize_validity(self, run: pd.DataFrame) -> None:
        for column in [
            name[: -len(FLAG_SUFFIX)]
            for name in run.columns
            if name.endswith(FLAG_SUFFIX)
        ]:
            valid = run[column].notna()
            previously_invalid = run[column + VALID_SUFFIX] & run[column].isna()
            # 原本就缺失的点标为 missing，便于区分“传感器坏值”和“没采到”。
            missing_reason = previously_invalid & (
                run[column + FLAG_SUFFIX].map(lambda text: REASON_MISSING not in text)
            )
            run.loc[missing_reason, column + FLAG_SUFFIX] = (
                run.loc[missing_reason, column + FLAG_SUFFIX].map(
                    lambda text: f"{text};{REASON_MISSING}" if text else REASON_MISSING
                )
            )
            run[column + VALID_SUFFIX] = valid.to_numpy()

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------
    @staticmethod
    def _rules_for(run: pd.DataFrame, rules: Iterable) -> list:
        return [rule for rule in rules if rule.column in run.columns]

    def _validate_columns(self, frame: pd.DataFrame) -> None:
        missing = {COL_RUN_ID, COL_TIME_MIN} - set(frame.columns)
        if missing:
            raise ValueError(f"数据缺少必需列：{sorted(missing)}")
        if frame.empty:
            raise ValueError("数据为空。")
        if frame[COL_RUN_ID].isna().any():
            raise ValueError("run_id 含缺失值。")

    def _build_report(self, cleaned: pd.DataFrame) -> HygieneReport:
        flag_columns = [name for name in cleaned.columns if name.endswith(FLAG_SUFFIX)]
        counts_by_rule: dict[str, int] = {}
        counts_by_column: dict[str, int] = {}
        imputed = 0
        informative = 0
        for flag_column in flag_columns:
            column = flag_column[: -len(FLAG_SUFFIX)]
            texts = cleaned[flag_column].fillna("").astype(str)
            hits = texts != ""
            # 列级计数只统计“可疑”行；死区/插值单独计数，避免稳态工况
            # 把报告淹没在成千上万条信息性标记里。
            suspicious = texts.map(
                lambda text: any(
                    reason and reason not in HygieneReport.INFORMATIVE_REASONS
                    for reason in text.split(";")
                )
            )
            counts_by_column[column] = int(suspicious.sum())
            for text in texts[hits]:
                for reason in text.split(";"):
                    if not reason:
                        continue
                    counts_by_rule[reason] = counts_by_rule.get(reason, 0) + 1
            imputed += int(
                texts.map(lambda text: REASON_IMPUTED in text.split(";")).sum()
            )
            informative += int(
                texts.map(lambda text: REASON_DEADBAND in text.split(";")).sum()
            )
        unresolved: dict[str, int] = {}
        for column in cleaned.columns:
            if not column.endswith(VALID_SUFFIX):
                continue
            base = column[: -len(VALID_SUFFIX)]
            missing = int(cleaned[base].isna().sum())
            if missing > 0:
                unresolved[base] = missing
        run_summaries = (
            cleaned.groupby(COL_RUN_ID)
            .agg(
                rows=(COL_TIME_MIN, "size"),
                t_min=(COL_TIME_MIN, "min"),
                t_max=(COL_TIME_MIN, "max"),
            )
            .reset_index()
        )
        return HygieneReport(
            n_rows=len(cleaned),
            n_runs=cleaned[COL_RUN_ID].nunique(),
            counts_by_rule=counts_by_rule,
            counts_by_column=counts_by_column,
            imputed_points=imputed,
            informative_flags=informative,
            unresolved_nan=unresolved,
            run_summaries=run_summaries,
        )


def apply_column_mapping(
    frame: pd.DataFrame, mapping: Mapping[str, str], *, drop_extra: bool = False
) -> pd.DataFrame:
    """把现场列名映射为统一列名；冲突时直接报错。"""
    collisions = set(mapping) & set(mapping.values())
    if collisions:
        raise ValueError(f"列名映射存在自映射/冲突：{sorted(collisions)}")
    missing = set(mapping) - set(frame.columns)
    if missing:
        raise ValueError(f"现场数据缺少映射源列：{sorted(missing)}")
    renamed = frame.rename(columns=dict(mapping))
    if drop_extra:
        renamed = renamed[list(mapping.values())]
    return renamed
