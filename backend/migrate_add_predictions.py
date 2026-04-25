"""
迁移脚本：新增 prediction_results 和 prediction_points 表

执行方式：
    cd backend
    python migrate_add_predictions.py
"""
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
load_dotenv(env_path)

from app.database import engine, Base
from app.models.prediction import PredictionResult, PredictionPoint


def migrate():
    print("=" * 60)
    print("迁移：新增 prediction_results 和 prediction_points 表")
    print("=" * 60)

    # 只创建新表（不会影响已有表）
    Base.metadata.create_all(bind=engine, tables=[
        PredictionResult.__table__,
        PredictionPoint.__table__,
    ])

    print("[OK] prediction_results 表已创建")
    print("[OK] prediction_points 表已创建")
    print("=" * 60)
    print("迁移完成！")


if __name__ == "__main__":
    migrate()
