from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.user import User
from app.utils.security import get_current_user
from app.schemas.backtest import (
    BacktestSummary, BacktestDetail, BacktestListResponse, BacktestRunRequest,
)
from app.services.backtest import (
    run_and_save_backtest, list_backtests, get_backtest_detail, delete_backtest,
    list_available_data_files,
)

router = APIRouter(prefix="/backtest", tags=["回测管理"])


@router.post("/run", response_model=BacktestDetail, summary="执行回测")
def run_backtest(
    request: BacktestRunRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    执行交易策略回测。

    - **strategy_type**: 策略类型 (trend_rider / ema_cross / macd)
    - **training_job_id**: 可选，指定训练任务的模型用于LSTM预测，不传则尝试自动加载最新模型
    - **data_filename**: 可选，指定股票数据文件，不传则使用config.json配置
    """
    return run_and_save_backtest(db, current_user, request)


@router.get("/list", response_model=BacktestListResponse, summary="回测结果列表")
def get_backtest_list(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return list_backtests(db, current_user, skip, limit)


@router.get("/data-files", summary="可用数据文件列表")
def get_available_data_files(
    current_user: User = Depends(get_current_user),
):
    """
    获取 data 目录下可用的 CSV 数据文件列表。
    用于前端回测页面的数据文件选择。
    """
    return list_available_data_files()


@router.get("/{backtest_id}", response_model=BacktestDetail, summary="回测结果详情")
def get_backtest(
    backtest_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return get_backtest_detail(db, current_user, backtest_id)


@router.delete("/{backtest_id}", summary="删除回测记录")
def remove_backtest(
    backtest_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    delete_backtest(db, current_user, backtest_id)
    return {"message": "删除成功"}
