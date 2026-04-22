import os
import sys
import json
import shutil
import tempfile
import numpy as np
import pandas as pd
from typing import List, Optional
from sqlalchemy.orm import Session
from fastapi import HTTPException, status

from app.models.backtest import BacktestResult, PortfolioSnapshot, TradeRecord
from app.models.user import User
from app.schemas.backtest import (
    BacktestSummary, BacktestDetail, BacktestListResponse,
    PortfolioSnapshotResponse, TradeRecordResponse, BacktestRunRequest,
)

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_DIR = os.path.dirname(BACKEND_DIR)

sys.path.insert(0, PROJECT_DIR)

from trading_strategy import (
    load_all_data,
    load_lstm_full_predictions,
    generate_signals_trend_rider,
    generate_signals_ema_cross_wide,
    generate_signals_macd_wide,
    backtest as run_backtest,
)


STRATEGY_MAP = {
    "trend_rider": generate_signals_trend_rider,
    "ema_cross": generate_signals_ema_cross_wide,
    "macd": generate_signals_macd_wide,
}

DEFAULT_PARAMS = {
    "trend_rider": {
        "trend_ema": 50, "confirm_ema": 20,
        "bull_trail": 0.15, "bear_trail": 0.03,
        "stop_loss_pct": 0.12, "rsi_period": 14,
    },
    "ema_cross": {
        "fast_ema": 8, "slow_ema": 21,
        "bull_trail": 0.15, "bear_trail": 0.03,
        "stop_loss_pct": 0.12, "rsi_period": 14,
    },
    "macd": {
        "trend_ema": 50,
        "bull_trail": 0.15, "bear_trail": 0.03,
        "stop_loss_pct": 0.12, "rsi_period": 14,
    },
}


def run_and_save_backtest(db: Session, user: User, request: BacktestRunRequest) -> BacktestDetail:
    strategy_type = request.strategy_type
    if strategy_type not in STRATEGY_MAP:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"不支持的策略类型: {strategy_type}")

    merged_params = dict(DEFAULT_PARAMS.get(strategy_type, {}))
    if request.strategy_params:
        merged_params.update(request.strategy_params)

    original_cwd = os.getcwd()
    try:
        os.chdir(PROJECT_DIR)
        dataframe = load_all_data()
    except Exception as e:
        os.chdir(original_cwd)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"加载数据失败: {str(e)}")

    prices = dataframe['Close'].values
    high = dataframe['High'].values
    low = dataframe['Low'].values
    close = prices
    dates = dataframe['Date'].values

    try:
        lstm_data = load_lstm_full_predictions(dataframe)
    except Exception:
        lstm_data = None

    strategy_fn = STRATEGY_MAP[strategy_type]

    try:
        signals = strategy_fn(
            prices, high, low, close, lstm_data,
            **merged_params,
        )
    except Exception as e:
        os.chdir(original_cwd)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"策略计算失败: {str(e)}")
    finally:
        os.chdir(original_cwd)

    results = run_backtest(signals, initial_capital=request.initial_capital, commission_rate=request.commission_rate)

    strategy_name = f"{strategy_type}(" + ",".join(f"{k}={v}" for k, v in merged_params.items()) + ")"

    backtest_record = BacktestResult(
        user_id=user.id,
        strategy_name=strategy_name,
        strategy_type=strategy_type,
        total_return=results['total_return'],
        annual_return=results['annual_return'],
        max_drawdown=results['max_drawdown'],
        n_trades=results['n_trades'],
        win_rate=results['win_rate'],
        benchmark_total_return=results['benchmark_total_return'],
        benchmark_annual=results['benchmark_annual'],
        initial_capital=request.initial_capital,
        commission_rate=request.commission_rate,
        strategy_params=merged_params,
    )
    db.add(backtest_record)
    db.flush()

    portfolio_values = results['portfolio_values']
    benchmark_values = results['benchmark_values']
    drawdown_arr = results['drawdown']
    price_arr = signals['price'].values

    snapshots = []
    batch_size = max(1, len(portfolio_values) // 500)
    for i in range(0, len(portfolio_values), batch_size):
        date_label = dates[i] if i < len(dates) else None
        snapshots.append(PortfolioSnapshot(
            backtest_id=backtest_record.id,
            date_index=i,
            date_label=str(date_label) if date_label is not None else None,
            portfolio_value=float(portfolio_values[i]),
            benchmark_value=float(benchmark_values[i]),
            drawdown=float(drawdown_arr[i]),
            price=float(price_arr[i]),
        ))
    db.bulk_save_objects(snapshots)

    trade_log = results['trade_log']
    trade_records = []
    for trade in trade_log:
        idx = trade['idx']
        date_label = dates[idx] if idx < len(dates) else None
        trade_records.append(TradeRecord(
            backtest_id=backtest_record.id,
            trade_type=trade['type'],
            date_index=idx,
            date_label=str(date_label) if date_label is not None else None,
            price=float(trade['price']),
        ))
    db.bulk_save_objects(trade_records)

    db.commit()
    db.refresh(backtest_record)

    return _build_detail(backtest_record)


def list_backtests(db: Session, user: User, skip: int = 0, limit: int = 20) -> BacktestListResponse:
    query = db.query(BacktestResult).filter(BacktestResult.user_id == user.id)
    total = query.count()
    items = query.order_by(BacktestResult.created_at.desc()).offset(skip).limit(limit).all()
    return BacktestListResponse(
        total=total,
        items=[BacktestSummary.model_validate(r) for r in items],
    )


def get_backtest_detail(db: Session, user: User, backtest_id: int) -> BacktestDetail:
    record = db.query(BacktestResult).filter(
        BacktestResult.id == backtest_id,
        BacktestResult.user_id == user.id,
    ).first()
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="回测记录不存在")
    return _build_detail(record)


def delete_backtest(db: Session, user: User, backtest_id: int) -> None:
    record = db.query(BacktestResult).filter(
        BacktestResult.id == backtest_id,
        BacktestResult.user_id == user.id,
    ).first()
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="回测记录不存在")
    db.delete(record)
    db.commit()


def _build_detail(record: BacktestResult) -> BacktestDetail:
    return BacktestDetail(
        id=record.id,
        strategy_name=record.strategy_name,
        strategy_type=record.strategy_type,
        total_return=record.total_return,
        annual_return=record.annual_return,
        max_drawdown=record.max_drawdown,
        n_trades=record.n_trades,
        win_rate=record.win_rate,
        benchmark_total_return=record.benchmark_total_return,
        benchmark_annual=record.benchmark_annual,
        initial_capital=record.initial_capital,
        commission_rate=record.commission_rate,
        strategy_params=record.strategy_params,
        created_at=record.created_at,
        portfolio_snapshots=[PortfolioSnapshotResponse.model_validate(s) for s in record.portfolio_snapshots],
        trade_records=[TradeRecordResponse.model_validate(t) for t in record.trade_records],
    )
