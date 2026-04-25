from typing import List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime


# ==================== Request Schemas ====================

class DatasetConfig(BaseModel):
    """数据集配置"""
    symbol: str = Field(..., description="股票代码，如 600519")
    symbol_name: str = Field(default="", description="股票名称")
    period: str = Field(default="daily", description="周期: daily / weekly / monthly")
    start_date: str = Field(..., description="起始日期 YYYYMMDD")
    end_date: str = Field(..., description="结束日期 YYYYMMDD")
    adjust: str = Field(default="qfq", description="复权: qfq / hfq / 空字符串")
    columns: List[str] = Field(default=["Close", "Volume"], description="使用的特征列")


class DatasetCreateRequest(BaseModel):
    """创建数据集请求"""
    name: str = Field(..., min_length=1, max_length=200, description="数据集名称")
    type: str = Field(default="stock", description="类型: stock / fund")
    config: DatasetConfig


# ==================== Response Schemas ====================

class DatasetSummary(BaseModel):
    """数据集摘要"""
    id: int
    name: str
    type: str
    config: Any
    status: str
    rows: Optional[int] = None
    columns_info: Optional[List[str]] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class DatasetDetail(BaseModel):
    """数据集详情"""
    id: int
    name: str
    type: str
    config: Any
    status: str
    saved_path: Optional[str] = None
    rows: Optional[int] = None
    columns_info: Optional[List[str]] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class DatasetListResponse(BaseModel):
    total: int
    items: List[DatasetSummary]


class DatasetDownloadResponse(BaseModel):
    """下载结果响应"""
    id: int
    name: str
    status: str
    saved_path: Optional[str] = None
    rows: Optional[int] = None
    columns_info: Optional[List[str]] = None
    error_message: Optional[str] = None
