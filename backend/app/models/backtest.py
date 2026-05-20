from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base


class BacktestResult(Base):
    __tablename__ = "backtest_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_name = Column(String(200), nullable=False)
    strategy_type = Column(String(50), nullable=False)

    # 数据与模型信息
    data_filename = Column(String(200), nullable=True)                          # 使用的股票数据文件
    training_job_id = Column(Integer, ForeignKey("training_jobs.id", ondelete="SET NULL"), nullable=True, index=True)  # 关联的训练任务
    saved_model_path = Column(String(500), nullable=True)                       # 实际使用的模型文件路径

    total_return = Column(Float, nullable=False)
    annual_return = Column(Float, nullable=False)
    max_drawdown = Column(Float, nullable=False)
    n_trades = Column(Integer, nullable=False)
    win_rate = Column(Float, nullable=False)
    benchmark_total_return = Column(Float, nullable=False)
    benchmark_annual = Column(Float, nullable=False)

    initial_capital = Column(Float, default=100000.0)
    commission_rate = Column(Float, default=0.001)

    strategy_params = Column(JSON, nullable=True)

    created_at = Column(DateTime, server_default=func.now(), nullable=False)

    training_job = relationship("TrainingJob", back_populates="backtest_results")
    portfolio_snapshots = relationship("PortfolioSnapshot", back_populates="backtest_result", cascade="all, delete-orphan")
    trade_records = relationship("TradeRecord", back_populates="backtest_result", cascade="all, delete-orphan")


class PortfolioSnapshot(Base):
    __tablename__ = "portfolio_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    backtest_id = Column(Integer, ForeignKey("backtest_results.id", ondelete="CASCADE"), nullable=False, index=True)
    date_index = Column(Integer, nullable=False)
    date_label = Column(String(20), nullable=True)
    portfolio_value = Column(Float, nullable=False)
    benchmark_value = Column(Float, nullable=False)
    drawdown = Column(Float, nullable=False)
    price = Column(Float, nullable=True)

    backtest_result = relationship("BacktestResult", back_populates="portfolio_snapshots")


class TradeRecord(Base):
    __tablename__ = "trade_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    backtest_id = Column(Integer, ForeignKey("backtest_results.id", ondelete="CASCADE"), nullable=False, index=True)
    trade_type = Column(String(10), nullable=False)
    date_index = Column(Integer, nullable=False)
    date_label = Column(String(20), nullable=True)
    price = Column(Float, nullable=False)

    backtest_result = relationship("BacktestResult", back_populates="trade_records")
