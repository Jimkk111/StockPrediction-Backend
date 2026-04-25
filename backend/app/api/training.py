import asyncio
from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.user import User
from app.models.training import TrainingJob
from app.utils.security import decode_access_token, get_current_user
from app.config import get_settings
from app.schemas.training import (
    TrainingDetail, TrainingSummary, TrainingListResponse,
    TrainingRunRequest, TrainingRunResponse,
)
from app.services.training import (
    start_training, list_trainings, get_training_detail, delete_training,
    TrainingEventBus,
)

router = APIRouter(prefix="/training", tags=["模型训练"])

settings = get_settings()


# ==================== WebSocket 鉴权辅助 ====================

from typing import Optional


def _authenticate_ws_token(token: str) -> Optional[str]:
    """验证 WebSocket query param 传入的 token，返回 username 或 None"""
    payload = decode_access_token(token)
    if payload is None:
        return None
    username = payload.get("sub")
    return username


# ==================== REST API ====================

@router.post("/run", response_model=TrainingRunResponse, summary="训练模型（异步）")
def run_training(
    request: TrainingRunRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    创建训练任务并立即返回 job ID，训练在后台执行。
    通过 WebSocket `/api/training/{id}/ws` 实时接收训练进度。
    """
    return start_training(db, current_user, request)


@router.get("/list", response_model=TrainingListResponse, summary="训练记录列表")
def get_training_list(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return list_trainings(db, current_user, skip, limit)


@router.get("/{training_id}", response_model=TrainingDetail, summary="训练结果详情")
def get_training(
    training_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return get_training_detail(db, current_user, training_id)


@router.delete("/{training_id}", summary="删除训练记录")
def remove_training(
    training_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    delete_training(db, current_user, training_id)
    return {"message": "删除成功"}


# ==================== WebSocket 实时进度 ====================

@router.websocket("/{training_id}/ws")
async def training_websocket(websocket: WebSocket, training_id: int, token: str = ""):
    """
    WebSocket 端点：实时推送训练进度。

    连接方式：ws://host/api/training/{id}/ws?token=xxx

    推送消息格式：
    - { "type": "started", "job_id": 1, "total_epochs": 100 }
    - { "type": "epoch", "epoch": 3, "total_epochs": 100, "loss": 0.023, "val_loss": null, "best_epoch": 3, "best_loss": 0.023 }
    - { "type": "completed", "job_id": 1, "final_loss": 0.0003, "best_epoch": 87, "training_time_seconds": 45.2, "saved_model_path": "..." }
    - { "type": "failed", "job_id": 1, "error_message": "..." }
    - { "type": "ping" }  （心跳）
    """

    # 1. 鉴权
    username = _authenticate_ws_token(token)
    if not username:
        await websocket.close(code=4001, reason="认证失败：无效 token")
        return

    # 2. 验证 job 归属
    from app.database import SessionLocal
    db = SessionLocal()
    try:
        job = db.query(TrainingJob).filter(TrainingJob.id == training_id).first()
        if not job:
            await websocket.close(code=4004, reason="训练记录不存在")
            return
        user = db.query(User).filter(User.username == username).first()
        if not user or job.user_id != user.id:
            await websocket.close(code=4003, reason="无权访问此训练记录")
            return
    finally:
        db.close()

    # 3. 接受连接
    await websocket.accept()

    # 4. 获取事件队列
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

    # 5. 如果没有队列（异常情况），也直接关闭
    if not queue:
        await websocket.send_json({
            "type": "failed",
            "job_id": training_id,
            "error_message": "训练事件队列不存在",
        })
        await websocket.close()
        return

    # 6. 实时推送循环
    try:
        while True:
            try:
                # 等待事件，带超时用于心跳
                event = await asyncio.wait_for(queue.get(), timeout=30.0)
                await websocket.send_json(event)

                # 训练完成或失败，关闭连接
                if event.get("type") in ("completed", "failed"):
                    # 清理队列
                    TrainingEventBus.remove_queue(training_id)
                    break
            except asyncio.TimeoutError:
                # 心跳：防止连接因空闲超时被断开
                await websocket.send_json({"type": "ping"})
    except WebSocketDisconnect:
        logger_msg = f"WebSocket disconnected for training #{training_id}"
        pass
    except Exception as e:
        pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
