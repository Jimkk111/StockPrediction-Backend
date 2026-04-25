from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.user import User
from app.utils.security import get_current_user
from app.schemas.dataset import (
    DatasetCreateRequest, DatasetSummary, DatasetDetail,
    DatasetListResponse, DatasetDownloadResponse,
)
from app.services.dataset import (
    create_dataset, list_datasets, get_dataset_detail,
    delete_dataset, download_dataset,
)

router = APIRouter(prefix="/dataset", tags=["数据集管理"])


@router.post("/create", response_model=DatasetDetail, summary="创建数据集")
def create(
    request: DatasetCreateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    创建数据集（仅存配置元信息，不立即下载数据）。
    首次用于回测/训练时会自动触发下载。
    """
    return create_dataset(db, current_user, request)


@router.get("/list", response_model=DatasetListResponse, summary="数据集列表")
def list_all(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return list_datasets(db, current_user, skip, limit)


@router.get("/{dataset_id}", response_model=DatasetDetail, summary="数据集详情")
def get_detail(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return get_dataset_detail(db, current_user, dataset_id)


@router.post("/{dataset_id}/download", response_model=DatasetDownloadResponse, summary="下载/刷新数据")
def download(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    手动触发数据下载或刷新已有数据。
    对于 status=pending 的数据集，会从 akshare 下载数据并保存为 CSV。
    对于 status=downloaded 的数据集，会重新下载覆盖旧数据（刷新）。
    """
    return download_dataset(db, current_user, dataset_id)


@router.delete("/{dataset_id}", summary="删除数据集")
def remove(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    delete_dataset(db, current_user, dataset_id)
    return {"message": "删除成功"}
