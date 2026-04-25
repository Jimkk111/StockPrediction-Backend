from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


# ==================== Request Schemas ====================

class BacktestRunRequest(BaseModel):
    """回测请求参数"""
    strategy_type: str = Field(..., description="策略类型: trend_rider / ema_cross / macd")
    strategy_params: Optional[dict] = Field(default=None, description="策略参数覆盖")
    initial_capital: float = Field(default=100000.0, gt=0, description="初始资金")
    commission_rate: float = Field(default=0.001, ge=0, lt=0.1, description="手续费率")
    # 模型选择
    training_job_id: Optional[int] = Field(default=None, description="指定训练任务ID（使用该任务的模型进行LSTM预测，不传则尝试自动加载最新模型）")
    # 数据源选择（三选一，优先级：dataset_id > data_filename > config.json默认）
    dataset_id: Optional[int] = Field(default=None, description="数据集ID（优先级最高，首次使用自动下载数据）")
    data_filename: Optional[str] = Field(default=None, description="股票数据文件名（data目录下），不传则使用config.json配置或nasdaq.csv")


# ==================== Response Schemas ====================

class BacktestSummary(BaseModel):
    """回测列表中的摘要信息"""
    id: int
    strategy_name: str
    strategy_type: str
    data_filename: Optional[str] = None
    training_job_id: Optional[int] = None
    saved_model_path: Optional[str] = None
    total_return: float
    annual_return: float
    max_drawdown: float
    n_trades: int
    win_rate: float
    benchmark_total_return: float
    benchmark_annual: float
    initial_capital: float
    commission_rate: float
    strategy_params: Optional[dict] = None
    created_at: datetime

    class Config:
        from_attributes = True


class PortfolioSnapshotResponse(BaseModel):
    date_index: int
    date_label: Optional[str] = None
    portfolio_value: float
    benchmark_value: float
    drawdown: float
    price: Optional[float] = None

    class Config:
        from_attributes = True


class TradeRecordResponse(BaseModel):
    trade_type: str
    date_index: int
    date_label: Optional[str] = None
    price: float

    class Config:
        from_attributes = True


class BacktestDetail(BaseModel):
    """回测结果详情"""
    id: int
    strategy_name: str
    strategy_type: str
    data_filename: Optional[str] = None
    training_job_id: Optional[int] = None
    saved_model_path: Optional[str] = None
    total_return: float
    annual_return: float
    max_drawdown: float
    n_trades: int
    win_rate: float
    benchmark_total_return: float
    benchmark_annual: float
    initial_capital: float
    commission_rate: float
    strategy_params: Optional[dict] = None
    created_at: datetime
    portfolio_snapshots: List[PortfolioSnapshotResponse] = []
    trade_records: List[TradeRecordResponse] = []

    class Config:
        from_attributes = True


class BacktestListResponse(BaseModel):
    total: int
    items: List[BacktestSummary]
