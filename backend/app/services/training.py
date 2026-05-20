import os
import sys
import time
import math
import asyncio
import logging
import threading
from typing import Dict, List, Optional
from sqlalchemy.orm import Session
from fastapi import HTTPException, status

from app.models.training import TrainingJob, TrainingEpochLog
from app.models.dataset import Dataset
from app.schemas.training import (
    TrainingRunRequest, TrainingRunResponse, TrainingDetail, TrainingSummary,
    TrainingListResponse, EpochLogResponse, LayerConfig,
)
from app.services.dataset import ensure_dataset_downloaded

logger = logging.getLogger(__name__)

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_DIR = os.path.dirname(BACKEND_DIR)

# 默认模型结构（与 config.json 保持一致）
DEFAULT_LAYERS = [
    {"type": "lstm", "neurons": 100, "input_timesteps": 49, "input_dim": 2, "return_seq": True},
    {"type": "dropout", "rate": 0.2},
    {"type": "lstm", "neurons": 100, "return_seq": True},
    {"type": "lstm", "neurons": 100, "return_seq": False},
    {"type": "dropout", "rate": 0.2},
    {"type": "dense", "neurons": 1, "activation": "linear"},
]


# ==================== 事件总线 ====================

class TrainingEventBus:
    """
    训练事件总线：桥接 Keras 训练线程与 WebSocket 异步端点。
    每个 job 有一个 asyncio.Queue，Keras callback 通过线程安全方式写入，
    WebSocket 端点从队列消费推送给前端。
    """
    _queues: Dict[int, asyncio.Queue] = {}
    _loop: Optional[asyncio.AbstractEventLoop] = None

    @classmethod
    def set_loop(cls, loop: asyncio.AbstractEventLoop):
        """由 FastAPI startup 事件调用，保存主事件循环引用"""
        cls._loop = loop

    @classmethod
    def create_queue(cls, job_id: int) -> asyncio.Queue:
        q = asyncio.Queue()
        cls._queues[job_id] = q
        return q

    @classmethod
    def get_queue(cls, job_id: int) -> Optional[asyncio.Queue]:
        return cls._queues.get(job_id)

    @classmethod
    def remove_queue(cls, job_id: int):
        cls._queues.pop(job_id, None)

    @classmethod
    def emit(cls, job_id: int, event: dict):
        """从训练线程中调用，线程安全地向队列放入事件"""
        q = cls._queues.get(job_id)
        if q and cls._loop and cls._loop.is_running():
            cls._loop.call_soon_threadsafe(q.put_nowait, event)


# ==================== 工具函数 ====================

def _resolve_layers(request: TrainingRunRequest) -> List[dict]:
    """解析模型结构：用户自定义 > 默认结构"""
    if request.layers:
        return [layer.model_dump(exclude_none=True) for layer in request.layers]
    layers = []
    for layer in DEFAULT_LAYERS:
        layer = dict(layer)
        if layer["type"] == "lstm" and "input_dim" in layer:
            layer["input_dim"] = len(request.data_columns)
            layer["input_timesteps"] = request.sequence_length - 1
        layers.append(layer)
    return layers


def _build_configs(request: TrainingRunRequest, layers: List[dict]) -> dict:
    """构建传给 core 模块的 configs 字典"""
    return {
        "data": {
            "filename": request.data_filename,
            "columns": request.data_columns,
            "sequence_length": request.sequence_length,
            "train_test_split": request.train_test_split,
            "normalise": request.normalise,
        },
        "training": {
            "epochs": request.epochs,
            "batch_size": request.batch_size,
        },
        "model": {
            "loss": request.loss,
            "optimizer": request.optimizer,
            "save_dir": "saved_models",
            "layers": layers,
        },
    }


# ==================== 后台训练执行 ====================

def _execute_training(job_id: int, configs: dict, save_dir: str):
    """
    在后台线程中执行的训练逻辑。
    通过 TrainingEventBus 推送实时进度，通过独立的数据库 session 操作记录。
    """
    from app.database import SessionLocal

    db = SessionLocal()
    try:
        training_job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
        if not training_job:
            return

        # 通知前端：训练开始
        TrainingEventBus.emit(job_id, {
            "type": "started",
            "job_id": job_id,
            "total_epochs": configs["training"]["epochs"],
        })

        original_cwd = os.getcwd()
        try:
            os.chdir(PROJECT_DIR)
            sys.path.insert(0, PROJECT_DIR)

            from core.data_processor import DataLoader
            from core.model import Model

            data = DataLoader(
                os.path.join("data", configs["data"]["filename"]),
                configs["data"]["train_test_split"],
                configs["data"]["columns"],
            )

            training_job.total_train_samples = data.len_train
            training_job.total_test_samples = data.len_test
            db.commit()

            model = Model()
            model.build_model(configs)

            seq_len = configs["data"]["sequence_length"]
            batch_size = configs["training"]["batch_size"]
            epochs = configs["training"]["epochs"]
            steps_per_epoch = math.ceil((data.len_train - seq_len) / batch_size)

            import datetime as dt
            save_fname = os.path.join(
                save_dir,
                f'{dt.datetime.now().strftime("%d%m%Y-%H%M%S")}-e{epochs}.h5',
            )

            from keras.callbacks import ModelCheckpoint, Callback

            checkpoint = ModelCheckpoint(
                filepath=save_fname,
                monitor="loss",
                save_best_only=True,
            )

            class WsEpochCallback(Callback):
                """Keras Callback：同时写入数据库 + 推送事件总线"""
                def __init__(self, job_id: int, total_epochs: int, db_session: Session):
                    self.job_id = job_id
                    self.total_epochs = total_epochs
                    self.db = db_session
                    self.best_loss = float("inf")
                    self.best_epoch = 0

                def on_epoch_end(self, epoch, logs=None):
                    logs = logs or {}
                    loss = float(logs.get("loss", 0))
                    val_loss = float(logs.get("val_loss", 0)) if "val_loss" in logs else None

                    # 写入数据库
                    epoch_log = TrainingEpochLog(
                        job_id=self.job_id,
                        epoch=epoch + 1,
                        loss=loss,
                        val_loss=val_loss,
                    )
                    self.db.add(epoch_log)
                    self.db.commit()

                    # 追踪最佳
                    compare_loss = val_loss if val_loss is not None else loss
                    if compare_loss < self.best_loss:
                        self.best_loss = compare_loss
                        self.best_epoch = epoch + 1

                    # 推送事件总线
                    TrainingEventBus.emit(self.job_id, {
                        "type": "epoch",
                        "epoch": epoch + 1,
                        "total_epochs": self.total_epochs,
                        "loss": loss,
                        "val_loss": val_loss,
                        "best_epoch": self.best_epoch,
                        "best_loss": self.best_loss,
                    })

                    logger.info(
                        f"[Training #{self.job_id}] Epoch {epoch+1}/{self.total_epochs}"
                        f" - loss: {loss:.6f}"
                        + (f" - val_loss: {val_loss:.6f}" if val_loss is not None else "")
                    )

            ws_callback = WsEpochCallback(job_id, epochs, db)
            callbacks = [checkpoint, ws_callback]

            start_time = time.time()

            model.model.fit(
                data.generate_train_batch(
                    seq_len=seq_len,
                    batch_size=batch_size,
                    normalise=configs["data"]["normalise"],
                ),
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                callbacks=callbacks,
            )

            elapsed = time.time() - start_time

            # 测试集评估
            x_test, y_test = data.get_test_data(
                seq_len=seq_len,
                normalise=configs["data"]["normalise"],
            )
            test_loss = float(model.model.evaluate(x_test, y_test, verbose=0))

            # 更新训练记录
            training_job.status = "completed"
            training_job.saved_model_path = save_fname
            training_job.final_loss = test_loss
            training_job.best_epoch = ws_callback.best_epoch
            training_job.training_time_seconds = round(elapsed, 2)
            db.commit()
            db.refresh(training_job)

            # 推送完成事件
            TrainingEventBus.emit(job_id, {
                "type": "completed",
                "job_id": job_id,
                "final_loss": test_loss,
                "best_epoch": ws_callback.best_epoch,
                "training_time_seconds": round(elapsed, 2),
                "saved_model_path": save_fname,
            })

        except Exception as e:
            logger.error(f"[Training #{job_id}] 训练失败: {str(e)}", exc_info=True)
            training_job.status = "failed"
            training_job.error_message = str(e)[:2000]
            db.commit()

            TrainingEventBus.emit(job_id, {
                "type": "failed",
                "job_id": job_id,
                "error_message": str(e)[:2000],
            })
        finally:
            os.chdir(original_cwd)

    except Exception as outer_e:
        logger.error(f"[Training #{job_id}] 外层异常: {str(outer_e)}", exc_info=True)
        TrainingEventBus.emit(job_id, {
            "type": "failed",
            "job_id": job_id,
            "error_message": str(outer_e)[:2000],
        })
    finally:
        db.close()


# ==================== API 调用的入口函数 ====================

def start_training(db: Session, request: TrainingRunRequest) -> TrainingRunResponse:
    """创建训练任务并启动后台线程，立即返回"""

    # 1. 解析数据源：dataset_id > data_filename
    actual_data_filename = request.data_filename
    actual_data_columns = request.data_columns

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
        # 从数据集配置覆盖数据列
        if dataset.config and dataset.config.get("columns"):
            actual_data_columns = dataset.config["columns"]
        # 用实际 CSV 文件名替代
        actual_data_filename = os.path.basename(csv_path)
    else:
        # 验证数据文件
        data_filepath = os.path.join(PROJECT_DIR, "data", request.data_filename)
        if not os.path.isfile(data_filepath):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"数据文件不存在: data/{request.data_filename}，可用文件: "
                       + ", ".join(f for f in os.listdir(os.path.join(PROJECT_DIR, "data")) if f.endswith(".csv")),
            )

    # 2. 解析模型结构
    layers = _resolve_layers(request)
    configs = _build_configs(request, layers)
    # 用实际解析的 data_filename 和 data_columns 覆盖 configs
    configs["data"]["filename"] = actual_data_filename
    configs["data"]["columns"] = actual_data_columns

    # 3. 确保 saved_models 目录存在
    save_dir = os.path.join(PROJECT_DIR, configs["model"]["save_dir"])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 4. 创建训练记录
    training_job = TrainingJob(
        dataset_id=request.dataset_id if request.dataset_id else None,
        data_filename=actual_data_filename,
        data_columns=actual_data_columns,
        sequence_length=request.sequence_length,
        train_test_split=request.train_test_split,
        normalise=1 if request.normalise else 0,
        epochs=request.epochs,
        batch_size=request.batch_size,
        loss=request.loss,
        optimizer=request.optimizer,
        network_config=layers,
        status="training",
    )
    db.add(training_job)
    db.commit()
    db.refresh(training_job)

    # 5. 创建事件队列
    TrainingEventBus.create_queue(training_job.id)

    # 6. 启动后台线程
    thread = threading.Thread(
        target=_execute_training,
        args=(training_job.id, configs, save_dir),
        daemon=True,
    )
    thread.start()

    return TrainingRunResponse(id=training_job.id, status="training")


def list_trainings(db: Session, skip: int = 0, limit: int = 20) -> TrainingListResponse:
    """获取训练记录列表"""
    query = db.query(TrainingJob)
    total = query.count()
    items = query.order_by(TrainingJob.created_at.desc()).offset(skip).limit(limit).all()
    return TrainingListResponse(
        total=total,
        items=[TrainingSummary.model_validate(r) for r in items],
    )


def get_training_detail(db: Session, training_id: int) -> TrainingDetail:
    """获取训练结果详情"""
    record = db.query(TrainingJob).filter(
        TrainingJob.id == training_id,
    ).first()
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="训练记录不存在")
    return _build_detail(record)


def delete_training(db: Session, training_id: int) -> None:
    """删除训练记录（同时删除模型文件）"""
    record = db.query(TrainingJob).filter(
        TrainingJob.id == training_id,
    ).first()
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="训练记录不存在")

    # 删除模型文件
    if record.saved_model_path and os.path.isfile(record.saved_model_path):
        os.remove(record.saved_model_path)

    db.delete(record)
    db.commit()


def _build_detail(record: TrainingJob) -> TrainingDetail:
    """构建训练详情响应"""
    return TrainingDetail(
        id=record.id,
        dataset_id=record.dataset_id,
        data_filename=record.data_filename,
        data_columns=record.data_columns,
        sequence_length=record.sequence_length,
        train_test_split=record.train_test_split,
        normalise=bool(record.normalise),
        epochs=record.epochs,
        batch_size=record.batch_size,
        loss=record.loss,
        optimizer=record.optimizer,
        network_config=record.network_config,
        status=record.status,
        saved_model_path=record.saved_model_path,
        final_loss=record.final_loss,
        final_val_loss=record.final_val_loss,
        best_epoch=record.best_epoch,
        training_time_seconds=record.training_time_seconds,
        error_message=record.error_message,
        total_train_samples=record.total_train_samples,
        total_test_samples=record.total_test_samples,
        created_at=record.created_at,
        updated_at=record.updated_at,
        epoch_logs=[EpochLogResponse.model_validate(log) for log in record.epoch_logs],
    )
