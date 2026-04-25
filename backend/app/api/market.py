from fastapi import APIRouter, Depends, Query
from fastapi.exceptions import HTTPException
from app.models.user import User
from app.utils.security import get_current_user
from app.schemas.market import (
    StockSearchResponse, FeatureListResponse, KlineResponse,
)
from app.services.market import search_stocks, get_feature_list, get_kline

router = APIRouter(prefix="/market", tags=["行情数据"])


@router.get("/search", response_model=StockSearchResponse, summary="搜索股票")
def search(
    keyword: str = Query(..., min_length=1, description="搜索关键词（代码或名称）"),
    limit: int = Query(default=20, ge=1, le=50),
    current_user: User = Depends(get_current_user),
):
    """模糊搜索 A 股代码或名称，返回匹配的股票列表。"""
    try:
        return search_stocks(keyword, limit)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=f"股票搜索服务暂时不可用，请稍后重试: {e}")


@router.get("/features", response_model=FeatureListResponse, summary="内置特征列表")
def features(
    type: str = Query(default="stock", description="数据类型: stock / fund"),
    current_user: User = Depends(get_current_user),
):
    """
    获取可选的特征列列表。
    用于前端让用户勾选需要的数据列。
    """
    return get_feature_list(type)


@router.get("/kline", response_model=KlineResponse, summary="预览K线行情")
def kline(
    symbol: str = Query(..., description="股票代码，如 600519"),
    period: str = Query(default="daily", description="周期: daily / weekly / monthly"),
    start_date: str = Query(default="", description="起始日期 YYYYMMDD"),
    end_date: str = Query(default="", description="结束日期 YYYYMMDD"),
    adjust: str = Query(default="qfq", description="复权: qfq / hfq / 空字符串"),
    columns: str = Query(default="", description="需要的列（逗号分隔），如 Close,Volume,Turnover"),
    current_user: User = Depends(get_current_user),
):
    """
    预览股票行情数据，不落盘。
    columns 不传则返回全部可选列。
    """
    col_list = [c.strip() for c in columns.split(",") if c.strip()] or None
    try:
        return get_kline(
            symbol=symbol,
            period=period,
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
            columns=col_list,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=f"行情数据获取失败，请稍后重试: {e}")
