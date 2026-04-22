from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class BacktestSummary(BaseModel):
    id: int
    strategy_name: str
    strategy_type: str
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
    id: int
    strategy_name: str
    strategy_type: str
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


class BacktestRunRequest(BaseModel):
    strategy_type: str = Field(..., description="策略类型: trend_rider / ema_cross / macd")
    strategy_params: Optional[dict] = Field(default=None, description="策略参数覆盖")
    initial_capital: float = Field(default=100000.0)
    commission_rate: float = Field(default=0.001)
