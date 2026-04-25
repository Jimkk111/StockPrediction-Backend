"""
数据库迁移脚本：为 backtest_results 表添加模型选择相关字段

新增字段：
- data_filename: 使用的数据文件名
- training_job_id: 关联的训练任务ID（FK → training_jobs.id）
- saved_model_path: 实际使用的模型文件路径

使用方式：python backend/migrate_backtest_add_model.py
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
        # 检查 backtest_results 表是否存在
        inspector = inspect(engine)
        if 'backtest_results' not in inspector.get_table_names():
            print("backtest_results 表不存在，将通过 init_db.py 创建")
            return

        existing_columns = {col['name'] for col in inspector.get_columns('backtest_results')}

        # 添加新字段
        new_columns = [
            ("data_filename", "ALTER TABLE backtest_results ADD COLUMN data_filename VARCHAR(200) NULL"),
            ("training_job_id", "ALTER TABLE backtest_results ADD COLUMN training_job_id INTEGER NULL"),
            ("saved_model_path", "ALTER TABLE backtest_results ADD COLUMN saved_model_path VARCHAR(500) NULL"),
        ]

        for col_name, alter_sql in new_columns:
            if col_name not in existing_columns:
                try:
                    conn.execute(text(alter_sql))
                    conn.commit()
                    print(f"  [OK] 已添加字段: {col_name}")
                except Exception as e:
                    print(f"  [FAIL] 添加字段 {col_name} 失败: {e}")
            else:
                print(f"  - 字段已存在，跳过: {col_name}")

        # 添加外键索引（如果 training_job_id 已存在但无索引）
        indexes = [idx['name'] for idx in inspector.get_indexes('backtest_results')]
        if 'ix_backtest_results_training_job_id' not in indexes:
            try:
                conn.execute(text(
                    "CREATE INDEX ix_backtest_results_training_job_id "
                    "ON backtest_results (training_job_id)"
                ))
                conn.commit()
                print("  [OK] 已创建索引: ix_backtest_results_training_job_id")
            except Exception as e:
                print(f"  - 索引创建跳过: {e}")

        # 添加外键约束
        fks = inspector.get_foreign_keys('backtest_results')
        fk_columns = {fk['constrained_columns'][0] for fk in fks if fk['constrained_columns']}
        if 'training_job_id' not in fk_columns:
            try:
                conn.execute(text(
                    "ALTER TABLE backtest_results ADD CONSTRAINT "
                    "fk_backtest_results_training_job_id "
                    "FOREIGN KEY (training_job_id) REFERENCES training_jobs(id) ON DELETE SET NULL"
                ))
                conn.commit()
                print("  [OK] 已添加外键约束: fk_backtest_results_training_job_id")
            except Exception as e:
                print(f"  - 外键添加跳过: {e}")

    engine.dispose()
    print("\n迁移完成！")


if __name__ == "__main__":
    print("=" * 60)
    print("数据库迁移：backtest_results 添加模型选择字段")
    print("=" * 60)
    migrate()
