from typing import List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime


# ==================== Request Schemas ====================

class LayerConfig(BaseModel):
    type: str = Field(..., description="层类型: lstm / dropout / dense")
    neurons: Optional[int] = Field(default=None, description="神经元数量（lstm/dense需要）")
    rate: Optional[float] = Field(default=None, description="dropout比率（dropout需要）")
    activation: Optional[str] = Field(default=None, description="激活函数（dense需要）")
    return_seq: Optional[bool] = Field(default=None, description="是否返回序列（lstm需要）")
    input_timesteps: Optional[int] = Field(default=None, description="输入时间步（仅第一层lstm需要）")
    input_dim: Optional[int] = Field(default=None, description="输入维度（仅第一层lstm需要）")


class TrainingRunRequest(BaseModel):
    """训练请求参数"""
    # 数据源选择（三选一，优先级：dataset_id > data_filename > config.json默认）
    dataset_id: Optional[int] = Field(default=None, description="数据集ID（优先级最高，首次使用自动下载数据）")
    # 数据配置
    data_filename: str = Field(default="nasdaq.csv", description="数据文件名（data目录下）")
    data_columns: List[str] = Field(default=["Close", "Volume"], description="使用的特征列")
    sequence_length: int = Field(default=50, ge=5, le=500, description="序列长度（时间窗口）")
    train_test_split: float = Field(default=0.85, gt=0.1, lt=1.0, description="训练集比例")
    normalise: bool = Field(default=True, description="是否标准化数据")

    # 训练配置
    epochs: int = Field(default=100, ge=1, le=1000, description="训练轮数")
    batch_size: int = Field(default=32, ge=1, le=512, description="批量大小")
    loss: str = Field(default="mse", description="损失函数")
    optimizer: str = Field(default="adam", description="优化器")

    # 模型结构
    layers: Optional[List[LayerConfig]] = Field(default=None, description="自定义模型结构（为空则使用默认结构）")


# ==================== Response Schemas ====================

class TrainingRunResponse(BaseModel):
    """训练启动后立即返回的响应"""
    id: int
    status: str

    class Config:
        from_attributes = True


class EpochLogResponse(BaseModel):
    epoch: int
    loss: float
    val_loss: Optional[float] = None

    class Config:
        from_attributes = True


class TrainingSummary(BaseModel):
    """训练列表中的摘要信息"""
    id: int
    dataset_id: Optional[int] = None
    data_filename: str
    sequence_length: int
    epochs: int
    batch_size: int
    status: str
    final_loss: Optional[float] = None
    best_epoch: Optional[int] = None
    training_time_seconds: Optional[float] = None
    saved_model_path: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class TrainingDetail(BaseModel):
    """训练结果详情"""
    id: int
    dataset_id: Optional[int] = None
    data_filename: str
    data_columns: List[str]
    sequence_length: int
    train_test_split: float
    normalise: bool
    epochs: int
    batch_size: int
    loss: str
    optimizer: str
    network_config: Any
    status: str
    saved_model_path: Optional[str] = None
    final_loss: Optional[float] = None
    final_val_loss: Optional[float] = None
    best_epoch: Optional[int] = None
    training_time_seconds: Optional[float] = None
    error_message: Optional[str] = None
    total_train_samples: Optional[int] = None
    total_test_samples: Optional[int] = None
    created_at: datetime
    updated_at: datetime
    epoch_logs: List[EpochLogResponse] = []

    class Config:
        from_attributes = True


class TrainingListResponse(BaseModel):
    total: int
    items: List[TrainingSummary]
