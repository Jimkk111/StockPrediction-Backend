from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base


class Dataset(Base):
    """数据集：存储用户自定义的数据配置，按需下载数据"""
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # 基本信息
    name = Column(String(200), nullable=False)
    type = Column(String(20), nullable=False, default="stock")  # stock / fund（预留）

    # 数据集配置（描述"要什么数据"）
    config = Column(JSON, nullable=False)
    # config 示例:
    # {
    #   "symbol": "600519",
    #   "symbol_name": "贵州茅台",
    #   "period": "daily",
    #   "start_date": "20240101",
    #   "end_date": "20260424",
    #   "adjust": "qfq",
    #   "columns": ["Close", "Volume"]
    # }

    # 下载状态
    status = Column(String(20), nullable=False, default="pending")  # pending / downloading / downloaded / failed
    saved_path = Column(String(500), nullable=True)       # 下载后的 CSV 文件路径
    rows = Column("rows", Integer, nullable=True)           # 数据行数（rows 是 MySQL 保留字，需指定列名）
    columns_info = Column(JSON, nullable=True)            # 实际列信息
    error_message = Column(Text, nullable=True)           # 下载失败原因

    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)

    training_jobs = relationship("TrainingJob", back_populates="dataset")
