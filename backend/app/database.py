from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.config import get_settings

settings = get_settings()

engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def ensure_models_registered():
    """确保所有模型已注册到 Base.metadata（用于 create_all）。
    放在函数内避免循环导入：database.py <-> models/__init__.py
    """
    from app.models import (  # noqa: F401
        User, BacktestResult, PortfolioSnapshot, TradeRecord,
        TrainingJob, TrainingEpochLog, Dataset,
        PredictionResult, PredictionPoint,
    )


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
