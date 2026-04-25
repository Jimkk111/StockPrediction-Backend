from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base


class PredictionResult(Base):
    """预测结果：用户选择股票+模型+策略生成次日预测"""
    __tablename__ = "prediction_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)

    # 股票信息
    stock_code = Column(String(20), nullable=True)          # 股票代码，如 "600519"
    stock_name = Column(String(100), nullable=True)         # 股票名称，如 "贵州茅台"
    data_source = Column(String(200), nullable=True)        # 数据来源标识（文件名/dataset标签）

    # 模型信息
    training_job_id = Column(Integer, ForeignKey("training_jobs.id", ondelete="SET NULL"), nullable=True, index=True)
    saved_model_path = Column(String(500), nullable=True)
    sequence_length = Column(Integer, nullable=True)

    # 策略信息
    strategy_type = Column(String(50), nullable=False)      # trend_rider / ema_cross / macd
    strategy_params = Column(JSON, nullable=True)

    # 次日预测核心结果
    predicted_date = Column(String(20), nullable=True)      # 预测的目标日期（次日）
    predicted_price = Column(Float, nullable=True)          # 次日预测价格（反标准化后）
    predicted_direction = Column(Integer, nullable=True)    # 1=涨, -1=跌, 0=平
    predicted_signal = Column(Integer, nullable=True)       # 策略信号: 1=买入, -1=卖出, 0=持有
    last_close = Column(Float, nullable=True)               # 预测基准价（最后一日收盘价）

    # 实际结果（后续匹配更新）
    actual_price = Column(Float, nullable=True)             # 次日实际收盘价
    actual_direction = Column(Integer, nullable=True)       # 次日实际涨跌方向
    is_matched = Column(Integer, default=0)                 # 0=未匹配, 1=已匹配

    # 预测精度统计
    mae = Column(Float, nullable=True)                      # 平均绝对误差
    rmse = Column(Float, nullable=True)                     # 均方根误差
    direction_accuracy = Column(Float, nullable=True)       # 方向预测准确率

    # 数据点数量
    n_points = Column(Integer, nullable=True)               # 预测数据点总数

    status = Column(String(20), nullable=False, default="completed")
    error_message = Column(Text, nullable=True)

    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)

    user = relationship("User", back_populates="prediction_results")
    training_job = relationship("TrainingJob", back_populates="prediction_results")
    points = relationship("PredictionPoint", back_populates="prediction_result", cascade="all, delete-orphan")


class PredictionPoint(Base):
    """预测数据点：用于图表展示，每条记录对应一个时间点的实际值与预测值"""
    __tablename__ = "prediction_points"

    id = Column(Integer, primary_key=True, autoincrement=True)
    prediction_id = Column(Integer, ForeignKey("prediction_results.id", ondelete="CASCADE"), nullable=False, index=True)

    date_index = Column(Integer, nullable=False)            # 在原始数据中的索引
    date_label = Column(String(20), nullable=True)          # 日期字符串
    actual_close = Column(Float, nullable=True)             # 真实收盘价
    predicted_close = Column(Float, nullable=True)          # 预测收盘价（反标准化后）
    predicted_norm = Column(Float, nullable=True)           # 标准化预测值（原始模型输出）
    signal = Column(Integer, default=0)                     # 策略信号: 1=买入, -1=卖出, 0=持有

    prediction_result = relationship("PredictionResult", back_populates="points")
