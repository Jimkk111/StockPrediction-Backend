import sys
import os
from dotenv import load_dotenv

backend_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(backend_dir, '.env')
load_dotenv(env_path)

sys.path.insert(0, backend_dir)

from sqlalchemy import create_engine, text
from app.config import get_settings

settings = get_settings()


def create_database_if_not_exists():
    base_url = (
        f"mysql+pymysql://{settings.MYSQL_USER}:{settings.MYSQL_PASSWORD}"
        f"@{settings.MYSQL_HOST}:{settings.MYSQL_PORT}"
    )
    engine = create_engine(base_url, pool_pre_ping=True)
    
    with engine.connect() as conn:
        conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {settings.MYSQL_DATABASE} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"))
        conn.commit()
    print(f"数据库 '{settings.MYSQL_DATABASE}' 已就绪")
    engine.dispose()


def init_database():
    print("=" * 60)
    print("LSTM Stock Prediction - 数据库初始化")
    print("=" * 60)

    print("\n检查并创建数据库...")
    create_database_if_not_exists()

    from app.database import engine, Base
    from app.models.user import User
    from app.models.backtest import BacktestResult, PortfolioSnapshot, TradeRecord
    from app.models.training import TrainingJob, TrainingEpochLog
    from app.models.dataset import Dataset
    from app.models.prediction import PredictionResult, PredictionPoint
    from app.utils.security import get_password_hash
    from sqlalchemy.orm import sessionmaker

    print("\n正在创建数据库表...")
    Base.metadata.create_all(bind=engine)
    print("数据库表创建完成")

    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()

    try:
        test_user = db.query(User).filter(User.username == "test").first()
        if not test_user:
            test_user = User(
                username="test",
                email="test@example.com",
                hashed_password=get_password_hash("test123"),
            )
            db.add(test_user)
            db.commit()
            print("测试用户已创建 (test / test123)")
        else:
            print("测试用户已存在，跳过创建")

        print("\n数据库表结构:")
        for table_name in Base.metadata.tables:
            print(f"  - {table_name}")

    except Exception as e:
        print(f"初始化失败: {str(e)}")
        db.rollback()
    finally:
        db.close()

    print("\n" + "=" * 60)
    print("数据库初始化完成")
    print("=" * 60)


if __name__ == "__main__":
    init_database()
