import asyncio
from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.training import TrainingJob
from app.schemas.training import (
    TrainingDetail, TrainingSummary, TrainingListResponse,
    TrainingRunRequest, TrainingRunResponse,
)
from app.services.training import (
    start_training, list_trainings, get_training_detail, delete_training,
    TrainingEventBus,
)

router = APIRouter(prefix="/training", tags=["模型训练"])


# ==================== REST API ====================

@router.post("/run", response_model=TrainingRunResponse, summary="训练模型（异步）")
def run_training(
    request: TrainingRunRequest,
    db: Session = Depends(get_db),
):
    """
    创建训练任务并立即返回 job ID，训练在后台执行。
    通过 WebSocket `/api/training/{id}/ws` 实时接收训练进度。
    """
    return start_training(db, request)


@router.get("/list", response_model=TrainingListResponse, summary="训练记录列表")
def get_training_list(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    return list_trainings(db, skip, limit)


@router.get("/{training_id}", response_model=TrainingDetail, summary="训练结果详情")
def get_training(
    training_id: int,
    db: Session = Depends(get_db),
):
    return get_training_detail(db, training_id)


@router.delete("/{training_id}", summary="删除训练记录")
def remove_training(
    training_id: int,
    db: Session = Depends(get_db),
):
    delete_training(db, training_id)
    return {"message": "删除成功"}


# ==================== WebSocket 实时进度 ====================

@router.websocket("/{training_id}/ws")
async def training_websocket(websocket: WebSocket, training_id: int):
    """
    WebSocket 端点：实时推送训练进度。

    连接方式：ws://host/api/training/{id}/ws

    推送消息格式：
    - { "type": "started", "job_id": 1, "total_epochs": 100 }
    - { "type": "epoch", "epoch": 3, "total_epochs": 100, "loss": 0.023, "val_loss": null, "best_epoch": 3, "best_loss": 0.023 }
    - { "type": "completed", "job_id": 1, "final_loss": 0.0003, "best_epoch": 87, "training_time_seconds": 45.2, "saved_model_path": "..." }
    - { "type": "failed", "job_id": 1, "error_message": "..." }
    - { "type": "ping" }  （心跳）
    """

    # 1. 接受连接
    await websocket.accept()

    # 2. 验证训练记录存在
    from app.database import SessionLocal
    db = SessionLocal()
    try:
        job = db.query(TrainingJob).filter(TrainingJob.id == training_id).first()
        if not job:
            await websocket.send_json({"type": "failed", "job_id": training_id, "error_message": "训练记录不存在"})
            await websocket.close()
            return
    finally:
        db.close()

    # 3. 获取事件队列
    queue = TrainingEventBus.get_queue(training_id)

    # 如果 job 已经完成/失败（训练已结束），直接发最终状态然后关闭
    db = SessionLocal()
    try:
        job = db.query(TrainingJob).filter(TrainingJob.id == training_id).first()
        if job and job.status in ("completed", "failed"):
            if job.status == "completed":
                await websocket.send_json({
                    "type": "completed",
                    "job_id": training_id,
                    "final_loss": job.final_loss,
                    "best_epoch": job.best_epoch,
                    "training_time_seconds": job.training_time_seconds,
                    "saved_model_path": job.saved_model_path,
                })
            else:
                await websocket.send_json({
                    "type": "failed",
                    "job_id": training_id,
                    "error_message": job.error_message,
                })
            # 发送已有的 epoch 日志
            for log in job.epoch_logs:
                await websocket.send_json({
                    "type": "epoch",
                    "epoch": log.epoch,
                    "total_epochs": job.epochs,
                    "loss": log.loss,
                    "val_loss": log.val_loss,
                })
            await websocket.close()
            return
    finally:
        db.close()

    # 4. 如果没有队列（异常情况），也直接关闭
    if not queue:
        await websocket.send_json({
            "type": "failed",
            "job_id": training_id,
            "error_message": "训练事件队列不存在",
        })
        await websocket.close()
        return

    # 5. 实时推送循环
    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=30.0)
                await websocket.send_json(event)

                if event.get("type") in ("completed", "failed"):
                    TrainingEventBus.remove_queue(training_id)
                    break
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "ping"})
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
