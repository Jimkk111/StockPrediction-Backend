"""
Prediction Service: 预测功能核心逻辑

职责：
- 加载股票数据 + 训练好的模型
- 生成全量 LSTM 预测（反标准化为真实价格）
- 应用交易策略生成信号
- 计算次日预测结果
- 保存预测数据（预测+实际）到数据库
- 支持后续匹配实际数据
"""
import os
import sys
import json
import shutil
import tempfile
import logging
import numpy as np
import pandas as pd
from typing import Optional, Tuple
from sqlalchemy.orm import Session
from fastapi import HTTPException, status

from app.models.prediction import PredictionResult, PredictionPoint
from app.models.training import TrainingJob
from app.models.dataset import Dataset
from app.models.user import User
from app.schemas.prediction import (
    PredictionRunRequest, PredictionDetail, PredictionSummary,
    PredictionListResponse, PredictionPointResponse, PredictionMatchRequest,
)
from app.services.dataset import ensure_dataset_downloaded

logger = logging.getLogger(__name__)

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_DIR = os.path.dirname(BACKEND_DIR)

sys.path.insert(0, PROJECT_DIR)

from trading_strategy import (
    generate_signals_trend_rider,
    generate_signals_ema_cross_wide,
    generate_signals_macd_wide,
)


# 策略映射（与 backtest service 一致）
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


# ==================== 数据源解析 ====================

def _resolve_data_source(request: PredictionRunRequest, db: Session, user: User) -> Tuple[str, str, Optional[list]]:
    """
    解析数据源：dataset_id > data_filename > config.json默认
    
    返回: (data_filepath, data_source_label, dataset_columns)
    """
    # 优先级 1: dataset_id
    if request.dataset_id:
        dataset = db.query(Dataset).filter(
            Dataset.id == request.dataset_id,
            Dataset.user_id == user.id,
        ).first()
        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"数据集不存在: #{request.dataset_id}",
            )
        csv_path = ensure_dataset_downloaded(db, dataset)
        label = f"dataset:{dataset.name}({dataset.id})"
        columns = dataset.config.get("columns") if dataset.config else None
        return csv_path, label, columns

    # 优先级 2: data_filename
    data_filename = request.data_filename
    if not data_filename:
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
            data_filename = "nasdaq.csv"

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


# ==================== 模型加载与预测 ====================

def _load_model_and_predict(dataframe: pd.DataFrame, model_path: str, seq_len: int) -> dict:
    """
    加载训练好的 LSTM 模型，生成全量预测，并反标准化为真实价格。
    
    返回:
        {
            'predictions_norm': np.array,      # 标准化预测值
            'predictions_price': np.array,     # 反标准化后的预测价格
            'lstm_signal': np.array,           # LSTM 信号
            'base_prices': np.array,           # 每个预测点对应的基准价格
            'n_features': int,                 # 模型使用的特征数
            'use_cols': list,                  # 使用的列名
        }
    """
    if not os.path.isfile(model_path):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"模型文件不存在: {model_path}",
        )

    temp_dir = tempfile.mkdtemp(prefix='lstm_pred_')

    try:
        temp_model_path = os.path.join(temp_dir, 'model.h5')
        shutil.copy2(model_path, temp_model_path)

        from core.model import Model

        model = Model()
        model.load_model(temp_model_path)

        input_shape = model.model.input_shape
        n_features = input_shape[2] if len(input_shape) == 3 else 2
        logger.info(f"模型输入形状: {input_shape}, 序列长度: {seq_len}")

        # 根据模型输入维度选择特征列
        all_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        available_cols = [c for c in all_cols if c in dataframe.columns]
        use_cols = available_cols[:n_features]
        logger.info(f"使用特征列: {use_cols}")

        # 确定 Close 列在 use_cols 中的位置
        close_idx = use_cols.index('Close') if 'Close' in use_cols else 0

        data_raw = dataframe[use_cols].values.astype(float)

        # 滑动窗口标准化 + 记录基准价格用于反标准化
        normalised_data = []
        base_prices = []  # 每个窗口的基准价格（窗口第一个 Close）

        for i in range(len(data_raw) - seq_len + 1):
            window = data_raw[i:i + seq_len]
            norm_window = np.zeros_like(window)
            base = window[0, close_idx]  # Close 列的第一个值作为基准
            base_prices.append(base)
            for col in range(window.shape[1]):
                base_val = window[0, col]
                if base_val != 0:
                    norm_window[:, col] = (window[:, col] / base_val) - 1
                else:
                    norm_window[:, col] = 0
            normalised_data.append(norm_window)

        normalised_data = np.array(normalised_data)
        base_prices = np.array(base_prices)

        x_all = normalised_data[:, :-1, :]

        logger.info(f"全量数据: {x_all.shape}, 生成预测中...")
        predictions_norm = model.predict_point_by_point(x_all)
        logger.info(f"生成 {len(predictions_norm)} 个预测点")

        # 反标准化：predicted_price = base_price * (1 + predicted_norm)
        predictions_price = base_prices * (1 + predictions_norm)

        # 生成 LSTM 信号（EMA 平滑）
        pred_series = pd.Series(predictions_norm)
        smooth_fast = pred_series.ewm(span=3, adjust=False).mean()
        smooth_slow = pred_series.ewm(span=10, adjust=False).mean()

        lstm_signal = np.zeros(len(dataframe))
        for i in range(len(predictions_norm)):
            idx = i + seq_len - 1
            if idx < len(lstm_signal):
                if smooth_fast.iloc[i] > smooth_slow.iloc[i]:
                    lstm_signal[idx] = 1
                else:
                    lstm_signal[idx] = -1

        return {
            'predictions_norm': predictions_norm,
            'predictions_price': predictions_price,
            'lstm_signal': lstm_signal,
            'base_prices': base_prices,
            'n_features': n_features,
            'use_cols': use_cols,
            'close_idx': close_idx,
        }

    except Exception as e:
        logger.error(f"加载模型预测失败: {str(e)[:200]}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"模型预测失败: {str(e)[:200]}",
        )
    finally:
        try:
            shutil.rmtree(temp_dir)
        except:
            pass


# ==================== 次日预测 ====================

def _predict_next_day(dataframe: pd.DataFrame, model_path: str, seq_len: int,
                      use_cols: list, close_idx: int) -> dict:
    """
    使用最后一组数据预测次日价格。
    取最后 seq_len-1 个时间步作为输入，预测下一个时间步的值。
    """
    temp_dir = tempfile.mkdtemp(prefix='lstm_pred_next_')

    try:
        temp_model_path = os.path.join(temp_dir, 'model.h5')
        shutil.copy2(model_path, temp_model_path)

        from core.model import Model

        model = Model()
        model.load_model(temp_model_path)

        data_raw = dataframe[use_cols].values.astype(float)
        n = len(data_raw)

        if n < seq_len:
            return None

        # 取最后 seq_len-1 个数据点作为输入
        last_window_data = data_raw[n - seq_len + 1:n]  # shape: (seq_len-1, n_features)
        base = data_raw[n - seq_len + 1, close_idx]      # 基准价格

        # 标准化
        norm_window = np.zeros_like(last_window_data)
        for col in range(last_window_data.shape[1]):
            base_val = last_window_data[0, col]
            if base_val != 0:
                norm_window[:, col] = (last_window_data[:, col] / base_val) - 1
            else:
                norm_window[:, col] = 0

        # 构造输入 shape: (1, seq_len-1, n_features)
        x_input = norm_window[np.newaxis, :, :]

        # 预测
        pred_norm = model.predict_point_by_point(x_input)
        pred_norm_val = float(pred_norm[0])

        # 反标准化
        pred_price = base * (1 + pred_norm_val)

        return {
            'predicted_norm': pred_norm_val,
            'predicted_price': pred_price,
            'base_price': base,
        }

    except Exception as e:
        logger.error(f"次日预测失败: {str(e)[:200]}", exc_info=True)
        return None
    finally:
        try:
            shutil.rmtree(temp_dir)
        except:
            pass


# ==================== 核心业务逻辑 ====================

def run_and_save_prediction(db: Session, user: User, request: PredictionRunRequest) -> PredictionDetail:
    """执行预测并保存结果"""

    strategy_type = request.strategy_type
    if strategy_type not in STRATEGY_MAP:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"不支持的策略类型: {strategy_type}，可选: {', '.join(STRATEGY_MAP.keys())}",
        )

    # 1. 查找训练任务
    job = db.query(TrainingJob).filter(
        TrainingJob.id == request.training_job_id,
        TrainingJob.user_id == user.id,
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

    # 2. 解析数据源
    data_filepath, data_label, dataset_columns = _resolve_data_source(request, db, user)

    # 3. 加载数据
    original_cwd = os.getcwd()
    try:
        os.chdir(PROJECT_DIR)
        dataframe = pd.read_csv(data_filepath)
    except Exception as e:
        os.chdir(original_cwd)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"加载数据失败: {str(e)}",
        )

    seq_len = job.sequence_length

    # 4. 加载模型 + 生成全量预测
    try:
        pred_data = _load_model_and_predict(dataframe, job.saved_model_path, seq_len)
    except HTTPException:
        os.chdir(original_cwd)
        raise
    except Exception as e:
        os.chdir(original_cwd)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"LSTM预测失败: {str(e)[:200]}",
        )

    # 5. 生成次日预测
    next_day_pred = _predict_next_day(
        dataframe, job.saved_model_path, seq_len,
        pred_data['use_cols'], pred_data['close_idx'],
    )

    # 6. 应用交易策略
    merged_params = dict(DEFAULT_PARAMS.get(strategy_type, {}))
    if request.strategy_params:
        merged_params.update(request.strategy_params)

    prices = dataframe['Close'].values
    high = dataframe['High'].values
    low = dataframe['Low'].values
    close = prices

    lstm_data = {
        'predictions': pred_data['predictions_norm'],
        'seq_len': seq_len,
        'lstm_signal': pred_data['lstm_signal'],
    }

    try:
        strategy_fn = STRATEGY_MAP[strategy_type]
        signals = strategy_fn(prices, high, low, close, lstm_data, **merged_params)
    except Exception as e:
        os.chdir(original_cwd)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"策略计算失败: {str(e)[:200]}",
        )
    finally:
        os.chdir(original_cwd)

    # 7. 计算预测精度
    predictions_price = pred_data['predictions_price']
    n_pred = len(predictions_price)

    # 对齐：predictions[i] 对应 dataframe 的第 (i + seq_len - 1) 行
    actual_prices_aligned = []
    predicted_prices_aligned = []
    for i in range(n_pred):
        idx = i + seq_len - 1
        if idx < len(prices):
            actual_prices_aligned.append(prices[idx])
            predicted_prices_aligned.append(predictions_price[i])

    actual_arr = np.array(actual_prices_aligned)
    predicted_arr = np.array(predicted_prices_aligned)

    # 计算误差指标
    errors = actual_arr - predicted_arr
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    # 方向准确率：预测方向与实际方向是否一致
    actual_directions = np.sign(np.diff(actual_arr))
    predicted_directions = np.sign(np.diff(predicted_arr))
    # 只在两者都不为0时比较
    valid_mask = (actual_directions != 0) & (predicted_directions != 0)
    if valid_mask.sum() > 0:
        direction_accuracy = float(np.mean(actual_directions[valid_mask] == predicted_directions[valid_mask]))
    else:
        direction_accuracy = 0.0

    # 8. 确定次日预测结果
    last_close = float(prices[-1])
    if next_day_pred:
        predicted_price = next_day_pred['predicted_price']
        predicted_direction = 1 if predicted_price > last_close else (-1 if predicted_price < last_close else 0)
    else:
        predicted_price = None
        predicted_direction = None

    # 获取最后一天的策略信号
    last_signal = int(signals['signal'].iloc[-1]) if len(signals) > 0 else 0
    predicted_signal = last_signal

    # 预测目标日期（下一个交易日）
    dates = dataframe['Date'].values
    predicted_date = str(dates[-1]) if len(dates) > 0 else None

    # 9. 推断股票代码/名称
    stock_code = request.stock_code
    stock_name = request.stock_name

    # 如果没有手动指定，尝试从 dataset 推断
    if not stock_code and request.dataset_id:
        dataset = db.query(Dataset).filter(Dataset.id == request.dataset_id).first()
        if dataset and dataset.config:
            stock_code = dataset.config.get("symbol", "")
            stock_name = dataset.config.get("symbol_name", "")

    # 10. 保存预测结果
    prediction_record = PredictionResult(
        user_id=user.id,
        stock_code=stock_code,
        stock_name=stock_name,
        data_source=data_label,
        training_job_id=job.id,
        saved_model_path=job.saved_model_path,
        sequence_length=seq_len,
        strategy_type=strategy_type,
        strategy_params=merged_params,
        predicted_date=predicted_date,
        predicted_price=predicted_price,
        predicted_direction=predicted_direction,
        predicted_signal=predicted_signal,
        last_close=last_close,
        mae=mae,
        rmse=rmse,
        direction_accuracy=direction_accuracy,
        n_points=n_pred,
        status="completed",
    )
    db.add(prediction_record)
    db.flush()

    # 11. 保存预测数据点（降采样到最多 500 个点）
    strategy_signals = signals['signal'].values
    batch_size = max(1, n_pred // 500)

    points = []
    for i in range(0, n_pred, batch_size):
        data_idx = i + seq_len - 1  # 对应原始数据中的索引
        date_label = str(dates[data_idx]) if data_idx < len(dates) else None
        actual_close = float(prices[data_idx]) if data_idx < len(prices) else None
        signal_val = int(strategy_signals[data_idx]) if data_idx < len(strategy_signals) else 0

        points.append(PredictionPoint(
            prediction_id=prediction_record.id,
            date_index=data_idx,
            date_label=date_label,
            actual_close=actual_close,
            predicted_close=float(predictions_price[i]),
            predicted_norm=float(pred_data['predictions_norm'][i]),
            signal=signal_val,
        ))

    db.bulk_save_objects(points)
    db.commit()
    db.refresh(prediction_record)

    return _build_detail(prediction_record)


# ==================== 匹配实际数据 ====================

def match_actual_data(db: Session, user: User, prediction_id: int, request: PredictionMatchRequest) -> PredictionDetail:
    """用实际次日收盘价更新预测记录"""
    record = db.query(PredictionResult).filter(
        PredictionResult.id == prediction_id,
        PredictionResult.user_id == user.id,
    ).first()
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="预测记录不存在")

    actual_price = request.actual_price
    actual_direction = 1 if actual_price > record.last_close else (-1 if actual_price < record.last_close else 0)

    record.actual_price = actual_price
    record.actual_direction = actual_direction
    record.is_matched = 1
    db.commit()
    db.refresh(record)

    return _build_detail(record)


# ==================== 列表/详情/删除 ====================

def list_predictions(db: Session, user: User, skip: int = 0, limit: int = 20) -> PredictionListResponse:
    """获取用户的预测记录列表"""
    query = db.query(PredictionResult).filter(PredictionResult.user_id == user.id)
    total = query.count()
    items = query.order_by(PredictionResult.created_at.desc()).offset(skip).limit(limit).all()
    return PredictionListResponse(
        total=total,
        items=[PredictionSummary.model_validate(r) for r in items],
    )


def get_prediction_detail(db: Session, user: User, prediction_id: int) -> PredictionDetail:
    """获取预测详情"""
    record = db.query(PredictionResult).filter(
        PredictionResult.id == prediction_id,
        PredictionResult.user_id == user.id,
    ).first()
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="预测记录不存在")
    return _build_detail(record)


def delete_prediction(db: Session, user: User, prediction_id: int) -> None:
    """删除预测记录"""
    record = db.query(PredictionResult).filter(
        PredictionResult.id == prediction_id,
        PredictionResult.user_id == user.id,
    ).first()
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="预测记录不存在")
    db.delete(record)
    db.commit()


# ==================== 辅助函数 ====================

def _build_detail(record: PredictionResult) -> PredictionDetail:
    """构建预测详情响应"""
    return PredictionDetail(
        id=record.id,
        stock_code=record.stock_code,
        stock_name=record.stock_name,
        data_source=record.data_source,
        training_job_id=record.training_job_id,
        saved_model_path=record.saved_model_path,
        sequence_length=record.sequence_length,
        strategy_type=record.strategy_type,
        strategy_params=record.strategy_params,
        predicted_date=record.predicted_date,
        predicted_price=record.predicted_price,
        predicted_direction=record.predicted_direction,
        predicted_signal=record.predicted_signal,
        last_close=record.last_close,
        actual_price=record.actual_price,
        actual_direction=record.actual_direction,
        is_matched=record.is_matched,
        mae=record.mae,
        rmse=record.rmse,
        direction_accuracy=record.direction_accuracy,
        n_points=record.n_points,
        status=record.status,
        error_message=record.error_message,
        created_at=record.created_at,
        updated_at=record.updated_at,
        points=[PredictionPointResponse.model_validate(p) for p in record.points],
    )
