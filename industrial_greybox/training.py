"""训练流程：分层数据准备、灰箱/机理/黑箱对照、早停与诊断回调。

训练目标
--------
对每条窗口最小化（按通道归一化后再合并）::

    L = Σ_c α_c · [ Σ_t m_c(t)·(ŷ_c(t) − y_c(t))² / Σ_t m_c(t) ] / Σ_c α_c

其中 ``m_c(t) ∈ {0,1}`` 是监督掩码（浓度只在真实化验点为 1，温度全程为 1），
``α_c`` 是 ``ModelConfig.channel_weights``（默认 ``[1, 0.25]``）。

先做通道内归一化再按 ``α`` 合并，是为了不让采样频率决定权重：否则温度
因为点多会以数倍权重压倒质量目标，网络会被训练成“拟合温度、牺牲浓度”。
梯度经 RK4/IMEX 反传，因此机理参数（k_ref、UA）与网络修正被**联合**辨识。

工业训练惯例
------------
- 按运行切分，验证集用于早停，测试集只在最后评估一次；
- 梯度范数裁剪；学习率在验证损失平台期下降；
- 记录数值诊断（非有限步、裁剪步）与修正幅度，防止“网络补偿物理错误”；
- 所有随机性由 ``keras.utils.set_random_seed`` 固定。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import keras
import numpy as np
import pandas as pd

from .config import ModelConfig
from .data import WindowedDataset
from .models import CSTRGreyBoxModel, compile_greybox, predict_trajectory
from .reports import regression_metrics, summarise_parameters


@dataclass
class TrainingResult:
    """训练输出与诊断。"""

    model: CSTRGreyBoxModel
    history: dict[str, list[float]]
    metrics: pd.DataFrame
    parameters: pd.DataFrame
    predictions: dict[str, np.ndarray] = field(default_factory=dict)
    health_records: pd.DataFrame = field(default_factory=pd.DataFrame)
    notes: list[str] = field(default_factory=list)


class NumericalHealthCallback(keras.callbacks.Callback):
    """每个 epoch 记录数值守护与修正幅度。

    若 ``nonfinite`` 或 ``clipped`` 持续增长，说明积分步长/守护区间设置
    与数据不匹配；此时继续训练只会得到“被回退状态伪装的稳定轨迹”。
    """

    def __init__(self) -> None:
        super().__init__()
        self.records: list[dict[str, float]] = []

    def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None) -> None:
        logs = logs or {}
        layer = getattr(self.model, "physics_layer", None)
        record = {"epoch": float(epoch)}
        if layer is not None:
            record["nonfinite_steps"] = float(layer.nonfinite_steps.result())
            record["clipped_steps"] = float(layer.clipped_steps.result())
            record["log_k_ref"] = float(np.asarray(layer.log_k_ref))
            record["log_ua"] = float(np.asarray(layer.log_ua))
        record["val_loss"] = float(logs.get("val_loss", np.nan))
        record["loss"] = float(logs.get("loss", np.nan))
        self.records.append(record)


def train_greybox(
    datasets: dict[str, WindowedDataset],
    config: ModelConfig,
    *,
    epochs: int = 60,
    batch_size: int = 32,
    learning_rate: float = 5e-3,
    patience: int = 10,
    seed: int = 42,
    verbose: int = 2,
    enable_corrections: bool = True,
) -> TrainingResult:
    """训练灰箱模型（或仅机理参数，当 ``enable_corrections=False``）。

    ``enable_corrections=False`` 时把 ``rate_correction.hidden_units`` 与
    ``enable_slow_drift`` 关闭，得到纯机理基线；其余流程（切分、损失、
    早停）完全一致，因此两者误差可直接比较。
    """
    if not enable_corrections:
        from dataclasses import replace

        from .config import RateCorrectionConfig

        config = replace(
            config,
            rate_correction=RateCorrectionConfig(hidden_units=0, l2_penalty=0.0),
            enable_slow_drift=False,
        )
    keras.utils.set_random_seed(seed)
    # 同一进程连续训练多个模型（例如灰箱 + 基线、或多组消融）时，
    # TensorFlow 会累积 tf.function 图，最终可能报
    # ``Arena alloc failed`` / ``FunctionDef`` 解析失败。每个模型训练前
    # 清一次图，既释放内存也避免不同模型共享残留的 tracing 状态。
    # 注意：调用方应在一个进程内只保留**正在使用**的模型；旧模型若还要
    # 推理，请在清图后再重新构建或先完成推理。
    keras.backend.clear_session()
    model = CSTRGreyBoxModel(config)
    compile_greybox(model, learning_rate=learning_rate)

    train = datasets["train"]
    validation = datasets.get("validation")
    callbacks: list[keras.callbacks.Callback] = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss" if validation is not None else "loss",
            patience=patience,
            restore_best_weights=True,
            min_delta=1e-6,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss" if validation is not None else "loss",
            factor=0.5,
            patience=max(3, patience // 2),
            min_lr=1e-5,
        ),
        NumericalHealthCallback(),
    ]
    health = callbacks[-1]
    fit_kwargs: dict[str, Any] = dict(
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose,
        shuffle=True,
    )
    if validation is not None:
        fit_kwargs["validation_data"] = (
            validation.as_model_inputs(),
            validation.targets,
            validation.weights,
        )
    history = model.fit(
        train.as_model_inputs(),
        train.targets,
        sample_weight=train.weights,
        **fit_kwargs,
    )

    predictions = {
        split: predict_trajectory(model, dataset)
        for split, dataset in datasets.items()
    }
    metrics_rows = []
    for split, prediction in predictions.items():
        dataset = datasets[split]
        conc = regression_metrics(
            prediction[:, :, 0],
            dataset.targets[:, :, 0],
            mask=dataset.weights[:, :, 0],
        )
        temp = regression_metrics(
            prediction[:, :, 1],
            dataset.targets[:, :, 1],
            mask=dataset.weights[:, :, 1],
        )
        metrics_rows.append(
            {
                "数据切分": split,
                "C_A RMSE (mol/L)": conc["rmse"],
                "C_A MAE (mol/L)": conc["mae"],
                "C_A bias (mol/L)": conc["bias"],
                "C_A 监督点数": conc["n"],
                "T RMSE (K)": temp["rmse"] * config.physics.temperature_scale_k,
                "T MAE (K)": temp["mae"] * config.physics.temperature_scale_k,
                "T 点数": temp["n"],
            }
        )
    metrics = pd.DataFrame(metrics_rows)
    parameters = summarise_parameters(model.fitted_parameters())
    return TrainingResult(
        model=model,
        history={key: list(values) for key, values in history.history.items()},
        metrics=metrics,
        parameters=parameters,
        predictions=predictions,
        health_records=pd.DataFrame(health.records),
        notes=[
            "``C_A 监督点数`` 为有效化验点数；温度误差已换算回 K。",
            "若 ``C_A`` 指标为空（nan），说明该切分没有化验监督点，"
            "不能据此宣称模型在该工况有效。",
            "请同时检查 ``health_records`` 中 nonfinite/clipped 是否为 0。",
        ],
    )


def _safe(value: float) -> float:
    return float(value) if np.isfinite(value) else float("nan")


def open_loop_error_table(
    predictions: dict[str, np.ndarray],
    targets: np.ndarray,
    weights: np.ndarray,
    *,
    temperature_scale_k: float,
) -> pd.DataFrame:
    """滚动（开环）误差表：多个模型、浓度与温度分开。

    开环误差是软测量的核心指标：从化验锚点出发，模型要独立滚动到下一个
    化验时刻。若模型只在“一步预测”上好，开环会迅速漂移。
    """
    rows = []
    for name, prediction in predictions.items():
        conc = regression_metrics(
            prediction[:, :, 0], targets[:, :, 0], mask=weights[:, :, 0]
        )
        temp = regression_metrics(
            prediction[:, :, 1], targets[:, :, 1], mask=weights[:, :, 1]
        )
        rows.append(
            {
                "模型": name,
                "C_A RMSE (mol/L)": conc["rmse"],
                "C_A bias (mol/L)": conc["bias"],
                "C_A P90 |err| (mol/L)": conc["p90_abs"],
                "C_A 点数": conc["n"],
                "T RMSE (K)": temp["rmse"] * temperature_scale_k,
                "T bias (K)": temp["bias"] * temperature_scale_k,
            }
        )
    return pd.DataFrame(rows).sort_values("C_A RMSE (mol/L)", ignore_index=True)


__all__ = [
    "NumericalHealthCallback",
    "TrainingResult",
    "format_table",
    "open_loop_error_table",
    "train_greybox",
]
