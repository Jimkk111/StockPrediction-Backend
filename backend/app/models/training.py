from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base


class TrainingJob(Base):
    """训练任务记录"""
    __tablename__ = "training_jobs"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # 数据配置
    dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="SET NULL"), nullable=True, index=True)  # 关联的数据集
    data_filename = Column(String(200), nullable=False)
    data_columns = Column(JSON, nullable=False)           # ["Close", "Volume"]
    sequence_length = Column(Integer, nullable=False)      # 50
    train_test_split = Column(Float, nullable=False)       # 0.85
    normalise = Column(Integer, nullable=False)            # 1=True, 0=False

    # 训练配置
    epochs = Column(Integer, nullable=False)
    batch_size = Column(Integer, nullable=False)
    loss = Column(String(50), nullable=False)              # mse
    optimizer = Column(String(50), nullable=False)         # adam

    # 模型网络结构
    network_config = Column(JSON, nullable=False)

    # 训练结果
    status = Column(String(20), nullable=False, default="pending")  # pending / training / completed / failed
    saved_model_path = Column(String(500), nullable=True)    # 保存的模型文件路径
    final_loss = Column(Float, nullable=True)
    final_val_loss = Column(Float, nullable=True)
    best_epoch = Column(Integer, nullable=True)
    training_time_seconds = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)

    # 训练参数摘要（方便前端展示）
    total_train_samples = Column(Integer, nullable=True)
    total_test_samples = Column(Integer, nullable=True)

    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)

    dataset = relationship("Dataset", back_populates="training_jobs")
    epoch_logs = relationship("TrainingEpochLog", back_populates="training_job", cascade="all, delete-orphan")
    backtest_results = relationship("BacktestResult", back_populates="training_job")
    prediction_results = relationship("PredictionResult", back_populates="training_job")


class TrainingEpochLog(Base):
    """每个 epoch 的训练日志"""
    __tablename__ = "training_epoch_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(Integer, ForeignKey("training_jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    epoch = Column(Integer, nullable=False)
    loss = Column(Float, nullable=False)
    val_loss = Column(Float, nullable=True)

    training_job = relationship("TrainingJob", back_populates="epoch_logs")
