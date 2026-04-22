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
)

router = APIRouter(prefix="/backtest", tags=["回测管理"])


@router.post("/run", response_model=BacktestDetail, summary="执行回测")
def run_backtest(
    request: BacktestRunRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return run_and_save_backtest(db, current_user, request)


@router.get("/list", response_model=BacktestListResponse, summary="回测结果列表")
def get_backtest_list(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return list_backtests(db, current_user, skip, limit)


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
