import os
import sys
import json
import numpy as np
import pandas as pd
from typing import List, Optional
from sqlalchemy.orm import Session
from fastapi import HTTPException, status

from app.models.backtest import BacktestResult, PortfolioSnapshot, TradeRecord
from app.models.training import TrainingJob
from app.models.dataset import Dataset
from app.schemas.backtest import (
    BacktestSummary, BacktestDetail, BacktestListResponse,
    PortfolioSnapshotResponse, TradeRecordResponse, BacktestRunRequest,
)
from app.services.dataset import ensure_dataset_downloaded

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_DIR = os.path.dirname(BACKEND_DIR)

sys.path.insert(0, PROJECT_DIR)

from trading_strategy import (
    load_all_data,
    load_lstm_full_predictions,
    load_lstm_predictions_from_model,
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

# 默认数据文件
DEFAULT_DATA_FILENAME = "nasdaq.csv"


def _resolve_data_source(request: BacktestRunRequest, db: Session) -> tuple:
    """
    解析数据源：dataset_id > data_filename > config.json默认
    
    返回: (data_filepath, data_filename_or_label, dataset_columns)
    - data_filepath: CSV 文件绝对路径
    - data_filename_or_label: 用于记录的数据来源标识
    - dataset_columns: 数据集指定的特征列（dataset 场景下有用），None 表示使用全部列
    """
    # 优先级 1: dataset_id
    if request.dataset_id:
        dataset = db.query(Dataset).filter(
            Dataset.id == request.dataset_id,
        ).first()
        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"数据集不存在: #{request.dataset_id}",
            )
        # 懒加载：自动触发下载
        csv_path = ensure_dataset_downloaded(db, dataset)
        label = f"dataset:{dataset.name}({dataset.id})"
        columns = dataset.config.get("columns") if dataset.config else None
        return csv_path, label, columns

    # 优先级 2: data_filename
    data_filename = request.data_filename
    if not data_filename:
        # 优先级 3: config.json
        config_path = os.path.join(PROJECT_DIR, 'config.json')
        if os.path.isfile(config_path):
            try:
                configs = json.load(open(config_path, 'r'))
                fname = configs.get('data', {}).get('filename')
                if fname:
                    data_filename = fname
            except Exception:
                pass
        if not data_filename:
            data_filename = DEFAULT_DATA_FILENAME

    data_filepath = os.path.join(PROJECT_DIR, "data", data_filename)
    if not os.path.isfile(data_filepath):
        data_dir = os.path.join(PROJECT_DIR, "data")
        available = []
        if os.path.isdir(data_dir):
            available = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"数据文件不存在: data/{data_filename}，可用文件: {', '.join(available)}",
        )
    return data_filepath, data_filename, None


def _load_lstm_data(dataframe, request: BacktestRunRequest, db: Session):
    """
    加载 LSTM 预测数据：
    1. 如果指定了 training_job_id → 用该任务的模型
    2. 否则 → 尝试自动加载最新模型（兼容旧行为）
    """
    if request.training_job_id:
        # 查找指定的训练任务
        job = db.query(TrainingJob).filter(
            TrainingJob.id == request.training_job_id,
        ).first()
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"训练任务不存在: #{request.training_job_id}",
            )
        if job.status != "completed":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"训练任务 #{request.training_job_id} 状态为 '{job.status}'，仅支持使用已完成的模型",
            )
        if not job.saved_model_path or not os.path.isfile(job.saved_model_path):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"训练任务 #{request.training_job_id} 的模型文件不存在或已被删除",
            )

        # 使用指定模型生成预测
        original_cwd = os.getcwd()
        try:
            os.chdir(PROJECT_DIR)
            lstm_data = load_lstm_predictions_from_model(
                dataframe, job.saved_model_path, job.sequence_length
            )
        finally:
            os.chdir(original_cwd)

        return lstm_data, job.saved_model_path, job.id
    else:
        # 自动加载最新模型（兼容旧行为）
        original_cwd = os.getcwd()
        try:
            os.chdir(PROJECT_DIR)
            lstm_data = load_lstm_full_predictions(dataframe)
        finally:
            os.chdir(original_cwd)

        return lstm_data, None, None


def run_and_save_backtest(db: Session, request: BacktestRunRequest) -> BacktestDetail:
    """执行回测并保存结果"""
    strategy_type = request.strategy_type
    if strategy_type not in STRATEGY_MAP:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"不支持的策略类型: {strategy_type}")

    merged_params = dict(DEFAULT_PARAMS.get(strategy_type, {}))
    if request.strategy_params:
        merged_params.update(request.strategy_params)

    # 解析数据源（支持 dataset_id / data_filename / config.json）
    try:
        data_filepath, data_label, dataset_columns = _resolve_data_source(request, db)
    except HTTPException:
        raise

    # 加载数据
    original_cwd = os.getcwd()
    try:
        os.chdir(PROJECT_DIR)
        dataframe = pd.read_csv(data_filepath)
    except Exception as e:
        os.chdir(original_cwd)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"加载数据失败: {str(e)}")

    prices = dataframe['Close'].values
    high = dataframe['High'].values
    low = dataframe['Low'].values
    close = prices
    dates = dataframe['Date'].values

    # 加载 LSTM 预测数据（核心改动：支持模型选择）
    try:
        lstm_data, used_model_path, used_job_id = _load_lstm_data(dataframe, request, db)
    except HTTPException:
        raise
    except Exception as e:
        os.chdir(original_cwd)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"加载LSTM模型失败: {str(e)}")

    # 生成交易信号
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

    # 执行回测
    results = run_backtest(signals, initial_capital=request.initial_capital, commission_rate=request.commission_rate)

    strategy_name = f"{strategy_type}(" + ",".join(f"{k}={v}" for k, v in merged_params.items()) + ")"

    # 保存回测结果
    # data_filename: 如果来自 dataset 则用 label，否则用文件名
    record_data_filename = data_label if request.dataset_id else (
        request.data_filename or data_label
    )
    backtest_record = BacktestResult(
        strategy_name=strategy_name,
        strategy_type=strategy_type,
        data_filename=record_data_filename,
        training_job_id=used_job_id,
        saved_model_path=used_model_path,
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

    # 保存组合快照（降采样到最多500个点）
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

    # 保存交易记录
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


def list_backtests(db: Session, skip: int = 0, limit: int = 20) -> BacktestListResponse:
    query = db.query(BacktestResult)
    total = query.count()
    items = query.order_by(BacktestResult.created_at.desc()).offset(skip).limit(limit).all()
    return BacktestListResponse(
        total=total,
        items=[BacktestSummary.model_validate(r) for r in items],
    )


def get_backtest_detail(db: Session, backtest_id: int) -> BacktestDetail:
    record = db.query(BacktestResult).filter(
        BacktestResult.id == backtest_id,
    ).first()
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="回测记录不存在")
    return _build_detail(record)


def delete_backtest(db: Session, backtest_id: int) -> None:
    record = db.query(BacktestResult).filter(
        BacktestResult.id == backtest_id,
    ).first()
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="回测记录不存在")
    db.delete(record)
    db.commit()


def list_available_data_files() -> dict:
    """获取可用的数据文件列表"""
    data_dir = os.path.join(PROJECT_DIR, "data")
    if not os.path.isdir(data_dir):
        return {"files": []}

    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    file_info = []
    for f in sorted(csv_files):
        filepath = os.path.join(data_dir, f)
        size = os.path.getsize(filepath)
        # 读取列信息
        try:
            df = pd.read_csv(filepath, nrows=1)
            columns = list(df.columns)
            rows = sum(1 for _ in open(filepath)) - 1  # 快速行数统计
        except Exception:
            columns = []
            rows = 0
        file_info.append({
            "filename": f,
            "size_bytes": size,
            "rows": rows,
            "columns": columns,
        })

    return {"files": file_info}


def _build_detail(record: BacktestResult) -> BacktestDetail:
    return BacktestDetail(
        id=record.id,
        strategy_name=record.strategy_name,
        strategy_type=record.strategy_type,
        data_filename=record.data_filename,
        training_job_id=record.training_job_id,
        saved_model_path=record.saved_model_path,
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
