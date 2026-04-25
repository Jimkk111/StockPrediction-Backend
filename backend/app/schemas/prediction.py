from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


# ==================== Request Schemas ====================

class PredictionRunRequest(BaseModel):
    """预测请求参数"""
    # 数据源选择（优先级：dataset_id > data_filename > config.json默认）
    dataset_id: Optional[int] = Field(default=None, description="数据集ID（优先级最高，首次使用自动下载数据）")
    data_filename: Optional[str] = Field(default=None, description="股票数据文件名（data目录下），不传则使用config.json配置")
    # 股票代码信息（用于标记，从 dataset 或文件名推断）
    stock_code: Optional[str] = Field(default=None, description="股票代码，如 600519")
    stock_name: Optional[str] = Field(default=None, description="股票名称，如 贵州茅台")

    # 模型选择（必选）
    training_job_id: int = Field(..., description="训练任务ID（使用该任务的模型进行LSTM预测）")

    # 策略选择
    strategy_type: str = Field(default="trend_rider", description="策略类型: trend_rider / ema_cross / macd")
    strategy_params: Optional[dict] = Field(default=None, description="策略参数覆盖")


# ==================== Response Schemas ====================

class PredictionPointResponse(BaseModel):
    """单个预测数据点"""
    date_index: int
    date_label: Optional[str] = None
    actual_close: Optional[float] = None
    predicted_close: Optional[float] = None
    predicted_norm: Optional[float] = None
    signal: int = 0

    class Config:
        from_attributes = True


class PredictionSummary(BaseModel):
    """预测列表中的摘要信息"""
    id: int
    stock_code: Optional[str] = None
    stock_name: Optional[str] = None
    data_source: Optional[str] = None
    training_job_id: Optional[int] = None
    strategy_type: str
    predicted_date: Optional[str] = None
    predicted_price: Optional[float] = None
    predicted_direction: Optional[int] = None
    predicted_signal: Optional[int] = None
    last_close: Optional[float] = None
    actual_price: Optional[float] = None
    actual_direction: Optional[int] = None
    is_matched: int = 0
    mae: Optional[float] = None
    rmse: Optional[float] = None
    direction_accuracy: Optional[float] = None
    n_points: Optional[int] = None
    created_at: datetime

    class Config:
        from_attributes = True


class PredictionDetail(BaseModel):
    """预测结果详情（含图表数据点）"""
    id: int
    stock_code: Optional[str] = None
    stock_name: Optional[str] = None
    data_source: Optional[str] = None
    training_job_id: Optional[int] = None
    saved_model_path: Optional[str] = None
    sequence_length: Optional[int] = None
    strategy_type: str
    strategy_params: Optional[dict] = None
    predicted_date: Optional[str] = None
    predicted_price: Optional[float] = None
    predicted_direction: Optional[int] = None
    predicted_signal: Optional[int] = None
    last_close: Optional[float] = None
    actual_price: Optional[float] = None
    actual_direction: Optional[int] = None
    is_matched: int = 0
    mae: Optional[float] = None
    rmse: Optional[float] = None
    direction_accuracy: Optional[float] = None
    n_points: Optional[int] = None
    status: str
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    points: List[PredictionPointResponse] = []

    class Config:
        from_attributes = True


class PredictionListResponse(BaseModel):
    """预测结果列表"""
    total: int
    items: List[PredictionSummary]


class PredictionMatchRequest(BaseModel):
    """匹配实际数据请求"""
    actual_price: float = Field(..., description="次日实际收盘价")
