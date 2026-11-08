"""工业灰箱软测量工具包。

模块地图
--------
``config``
    全部可调参数、单位约定、列名与配置指纹。
``numerics``
    可微 RK4 / IMEX 积分与数值守护（NaN/Inf 回退、状态裁剪）。
``delays``
    化验死时间对齐（离线）与时间戳查询缓冲（在线）。
``hygiene``
    工业数据清洗（硬边界、Hampel 毛刺、变化率、死区、限长插值）。
``slow_features``
    结垢/老化的可解释慢特征（运行年龄、累计进料、温度 EMA）。
``mechanistic_layer``
    可微 CSTR 机理层：速率修正 + 慢变 UA/活性修正 + 积分。
``data``
    按运行切分的窗口化数据集与监督掩码。
``models``
    Keras 模型、掩码物理损失与编译辅助。
``synthetic_plant``
    带漂移/时滞/仪表故障的合成 campaign（仅用于回归测试与教学）。
``training``
    训练循环、数值健康回调与开环误差表。
``reports``
    分层指标（域内/域外、按运行/按日）与文本报表。
``diagnostics``
    参数灵敏度/可辨识性、残差结构检查。
``shadow``
    因果流式影子运行、诊断日志与滚动报表。
``pipeline``
    端到端装配：清洗 → 对齐 → 窗口 → 训练 → 评估 → 影子运行。
``cli``
    命令行入口（合成数据或现场 CSV，只读评估）。

最低使用示例::

    from industrial_greybox import run_demo

    result = run_demo(quick=True)
    print(result.summary)

**边界声明**：本包使用合成数据演示工程结构，不是任何真实装置的
验证结果；上线前必须完成数据字典、单位审核、独立验证与工艺/安全审批。
"""

from __future__ import annotations

from typing import Any

from .config import ModelConfig

__all__ = ["ModelConfig", "run_demo", "demo_summary"]


def run_demo(quick: bool = True, output_dir: str | None = None) -> Any:
    """跑通一条最小端到端流程，用于冒烟测试/教学。

    Parameters
    ----------
    quick
        True 时用较小的 campaign（8 条运行、160 步、10 个 epoch、stride=4），
        在普通 CPU 上约 1–2 分钟；False 时用更大的 campaign（20 条运行、
        480 步、40 个 epoch）。
    output_dir
        若提供，写出影子日志与报表文件。
    """
    import os

    os.environ.setdefault("KERAS_BACKEND", "tensorflow")

    from pathlib import Path

    from .config import DelayConfig, ModelConfig, ShadowConfig
    from .pipeline import format_pipeline_summary, run_pipeline
    from .synthetic_plant import CampaignSpec, simulate_campaign

    spec = (
        CampaignSpec(n_runs=8, n_steps=160, days_span=30, lab_interval_min=10)
        if quick
        else CampaignSpec(n_runs=20, n_steps=480, days_span=60, lab_interval_min=10)
    )
    dcs, lab = simulate_campaign(spec)
    result = run_pipeline(
        dcs,
        lab,
        config=ModelConfig(),
        delay=DelayConfig(),
        shadow=ShadowConfig(
            output_dir=Path(output_dir) if output_dir else Path("shadow_runs"),
            group_by="day",
        ),
        epochs=10 if quick else 40,
        stride=4 if quick else 2,
        verbose=1,
    )
    result.summary = format_pipeline_summary(result)
    return result


def demo_summary(result: Any) -> str:
    """返回 ``run_demo`` 的文本摘要（便于 Notebook 显示）。"""
    return getattr(result, "summary", "(无摘要)")
