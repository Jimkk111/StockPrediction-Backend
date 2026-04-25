"""
数据库迁移脚本：为 training_jobs 表添加数据集关联字段

新增字段：
- dataset_id: 关联的数据集ID（FK → datasets.id, ON DELETE SET NULL）

使用方式：python backend/migrate_training_add_dataset.py
"""
import sys
import os

backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)

from dotenv import load_dotenv
env_path = os.path.join(backend_dir, '.env')
load_dotenv(env_path)

from sqlalchemy import create_engine, text, inspect
from app.config import get_settings

settings = get_settings()


def migrate():
    engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True)

    with engine.connect() as conn:
        # 检查 training_jobs 表是否存在
        inspector = inspect(engine)
        if 'training_jobs' not in inspector.get_table_names():
            print("training_jobs 表不存在，将通过 init_db.py 创建")
            return

        existing_columns = {col['name'] for col in inspector.get_columns('training_jobs')}

        # 添加新字段
        if 'dataset_id' not in existing_columns:
            try:
                conn.execute(text(
                    "ALTER TABLE training_jobs ADD COLUMN dataset_id INTEGER NULL"
                ))
                conn.commit()
                print("  [OK] 已添加字段: dataset_id")
            except Exception as e:
                print(f"  [FAIL] 添加字段 dataset_id 失败: {e}")
        else:
            print("  - 字段已存在，跳过: dataset_id")

        # 添加索引
        indexes = [idx['name'] for idx in inspector.get_indexes('training_jobs')]
        if 'ix_training_jobs_dataset_id' not in indexes:
            try:
                conn.execute(text(
                    "CREATE INDEX ix_training_jobs_dataset_id "
                    "ON training_jobs (dataset_id)"
                ))
                conn.commit()
                print("  [OK] 已创建索引: ix_training_jobs_dataset_id")
            except Exception as e:
                print(f"  - 索引创建跳过: {e}")

        # 添加外键约束
        fks = inspector.get_foreign_keys('training_jobs')
        fk_columns = {fk['constrained_columns'][0] for fk in fks if fk['constrained_columns']}
        if 'dataset_id' not in fk_columns:
            try:
                conn.execute(text(
                    "ALTER TABLE training_jobs ADD CONSTRAINT "
                    "fk_training_jobs_dataset_id "
                    "FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE SET NULL"
                ))
                conn.commit()
                print("  [OK] 已添加外键约束: fk_training_jobs_dataset_id")
            except Exception as e:
                print(f"  - 外键添加跳过: {e}")

    engine.dispose()
    print("\n迁移完成！")


if __name__ == "__main__":
    print("=" * 60)
    print("数据库迁移：training_jobs 添加数据集关联字段")
    print("=" * 60)
    migrate()
