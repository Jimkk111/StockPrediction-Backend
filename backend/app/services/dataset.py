"""
Dataset Service: 数据集 CRUD + 懒加载下载

职责：
- 创建/列表/详情/删除数据集
- 手动触发下载
- 懒加载：回测/训练首次使用时自动触发下载
"""
import os
import logging
from typing import Optional
from sqlalchemy.orm import Session
from fastapi import HTTPException, status

from app.models.dataset import Dataset
from app.schemas.dataset import (
    DatasetCreateRequest, DatasetSummary, DatasetDetail,
    DatasetListResponse, DatasetDownloadResponse,
)
from app.services.market import download_stock_data

logger = logging.getLogger(__name__)

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_DIR = os.path.dirname(BACKEND_DIR)


# ==================== CRUD ====================

def create_dataset(db: Session, request: DatasetCreateRequest) -> DatasetDetail:
    """创建数据集（仅存配置，不下载）"""
    dataset = Dataset(
        name=request.name,
        type=request.type,
        config=request.config.model_dump(),
        status="pending",
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return _build_detail(dataset)


def list_datasets(db: Session, skip: int = 0, limit: int = 20) -> DatasetListResponse:
    """数据集列表"""
    query = db.query(Dataset)
    total = query.count()
    items = query.order_by(Dataset.created_at.desc()).offset(skip).limit(limit).all()
    return DatasetListResponse(
        total=total,
        items=[DatasetSummary.model_validate(r) for r in items],
    )


def get_dataset_detail(db: Session, dataset_id: int) -> DatasetDetail:
    """数据集详情"""
    record = db.query(Dataset).filter(
        Dataset.id == dataset_id,
    ).first()
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="数据集不存在")
    return _build_detail(record)


def delete_dataset(db: Session, dataset_id: int) -> None:
    """删除数据集（同时删 CSV 文件）"""
    record = db.query(Dataset).filter(
        Dataset.id == dataset_id,
    ).first()
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="数据集不存在")

    # 删除 CSV 文件
    if record.saved_path and os.path.isfile(record.saved_path):
        try:
            os.remove(record.saved_path)
        except Exception as e:
            logger.warning(f"删除数据集文件失败: {record.saved_path}, error={e}")

    db.delete(record)
    db.commit()


# ==================== 下载 ====================

def download_dataset(db: Session, dataset_id: int) -> DatasetDownloadResponse:
    """手动触发下载/刷新数据集"""
    record = db.query(Dataset).filter(
        Dataset.id == dataset_id,
    ).first()
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="数据集不存在")

    _do_download(db, record)
    return DatasetDownloadResponse(
        id=record.id,
        name=record.name,
        status=record.status,
        saved_path=record.saved_path,
        rows=record.rows,
        columns_info=record.columns_info,
        error_message=record.error_message,
    )


def ensure_dataset_downloaded(db: Session, dataset: Dataset) -> str:
    """
    懒加载：确保数据集已下载。
    如果 status=pending 则自动触发下载。
    返回 CSV 文件路径。
    """
    if dataset.status == "downloaded" and dataset.saved_path and os.path.isfile(dataset.saved_path):
        return dataset.saved_path

    if dataset.status == "pending" or dataset.status == "failed":
        _do_download(db, dataset)

    if dataset.status == "downloaded" and dataset.saved_path:
        return dataset.saved_path

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"数据集 #{dataset.id} 数据下载失败: {dataset.error_message or '未知错误'}",
    )


def _do_download(db: Session, dataset: Dataset) -> None:
    """执行实际下载逻辑"""
    config = dataset.config
    dataset.status = "downloading"
    db.commit()

    try:
        if dataset.type == "stock":
            df = download_stock_data(
                symbol=config["symbol"],
                period=config.get("period", "daily"),
                start_date=config.get("start_date", ""),
                end_date=config.get("end_date", ""),
                adjust=config.get("adjust", "qfq"),
                columns=config.get("columns"),
            )
        else:
            raise ValueError(f"暂不支持的数据类型: {dataset.type}")

        # 保存到 data/ 目录
        data_dir = os.path.join(PROJECT_DIR, "data")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        filename = f"dataset_{dataset.id}_{config['symbol']}.csv"
        save_path = os.path.join(data_dir, filename)
        df.to_csv(save_path, index=False)

        # 更新记录
        dataset.status = "downloaded"
        dataset.saved_path = save_path
        dataset.rows = len(df)
        dataset.columns_info = list(df.columns)
        dataset.error_message = None
        db.commit()

        logger.info(f"数据集 #{dataset.id} 下载成功: {save_path}, {len(df)} 行")

    except Exception as e:
        logger.error(f"数据集 #{dataset.id} 下载失败: {e}")
        dataset.status = "failed"
        dataset.error_message = str(e)[:2000]
        db.commit()


# ==================== 辅助 ====================

def _build_detail(record: Dataset) -> DatasetDetail:
    return DatasetDetail(
        id=record.id,
        name=record.name,
        type=record.type,
        config=record.config,
        status=record.status,
        saved_path=record.saved_path,
        rows=record.rows,
        columns_info=record.columns_info,
        error_message=record.error_message,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )
