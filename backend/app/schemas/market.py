from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


# ==================== Market Schemas ====================

class StockSearchResult(BaseModel):
    code: str
    name: str


class StockSearchResponse(BaseModel):
    items: List[StockSearchResult]


class FeatureInfo(BaseModel):
    """单个特征的信息"""
    key: str
    name: str
    category: str
    selectable: bool = True
    default_selected: bool = False


class FeatureListResponse(BaseModel):
    """特征列表响应"""
    type: str = "stock"
    features: List[FeatureInfo]


class KlineDataItem(BaseModel):
    """单条 K 线数据"""
    date: str
    open: Optional[float] = None
    close: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    volume: Optional[float] = None
    amount: Optional[float] = None
    turnover: Optional[float] = None
    pct_change: Optional[float] = None
    change: Optional[float] = None
    amplitude: Optional[float] = None


class KlineResponse(BaseModel):
    """K 线预览响应"""
    symbol: str
    name: str
    period: str
    count: int
    columns: List[str]
    data: List[KlineDataItem]
