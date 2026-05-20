from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from app.database import get_db
from app.schemas.prediction import (
    PredictionSummary, PredictionDetail, PredictionListResponse,
    PredictionRunRequest, PredictionMatchRequest,
)
from app.services.prediction import (
    run_and_save_prediction, list_predictions, get_prediction_detail,
    delete_prediction, match_actual_data,
)

router = APIRouter(prefix="/prediction", tags=["预测管理"])


@router.post("/run", response_model=PredictionDetail, summary="执行预测")
def run_prediction(
    request: PredictionRunRequest,
    db: Session = Depends(get_db),
):
    """
    选择股票+模型+策略，生成次日预测结果。

    - **training_job_id**: 必填，指定训练任务的模型用于LSTM预测
    - **strategy_type**: 策略类型 (trend_rider / ema_cross / macd)
    - **dataset_id**: 可选，数据集ID（优先级最高，首次使用自动下载数据）
    - **data_filename**: 可选，指定股票数据文件，不传则使用config.json配置
    - **stock_code / stock_name**: 可选，股票代码/名称，不传则从dataset推断
    """
    return run_and_save_prediction(db, request)


@router.get("/list", response_model=PredictionListResponse, summary="预测结果列表")
def get_prediction_list(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    return list_predictions(db, skip, limit)


@router.get("/{prediction_id}", response_model=PredictionDetail, summary="预测结果详情")
def get_prediction(
    prediction_id: int,
    db: Session = Depends(get_db),
):
    return get_prediction_detail(db, prediction_id)


@router.put("/{prediction_id}/match", response_model=PredictionDetail, summary="匹配实际数据")
def match_prediction(
    prediction_id: int,
    request: PredictionMatchRequest,
    db: Session = Depends(get_db),
):
    """
    用实际次日收盘价更新预测记录，计算预测是否准确。
    
    - **actual_price**: 次日实际收盘价
    """
    return match_actual_data(db, prediction_id, request)


@router.delete("/{prediction_id}", summary="删除预测记录")
def remove_prediction(
    prediction_id: int,
    db: Session = Depends(get_db),
):
    delete_prediction(db, prediction_id)
    return {"message": "删除成功"}
