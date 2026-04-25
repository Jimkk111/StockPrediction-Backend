"""
数据库迁移脚本：创建 datasets 表

使用方式：python backend/migrate_add_datasets.py
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
        inspector = inspect(engine)
        existing_tables = inspector.get_table_names()

        if 'datasets' in existing_tables:
            print("datasets table already exists, skipping")
            engine.dispose()
            return

        conn.execute(text("""
            CREATE TABLE datasets (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT NOT NULL,
                name VARCHAR(200) NOT NULL,
                type VARCHAR(20) NOT NULL DEFAULT 'stock',
                config JSON NOT NULL,
                status VARCHAR(20) NOT NULL DEFAULT 'pending',
                saved_path VARCHAR(500) NULL,
                `rows` INT NULL,
                columns_info JSON NULL,
                error_message TEXT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP NOT NULL,
                INDEX ix_datasets_user_id (user_id),
                CONSTRAINT fk_datasets_user_id FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
        """))
        conn.commit()
        print("[OK] datasets table created")

    engine.dispose()
    print("\nMigration done!")


if __name__ == "__main__":
    print("=" * 60)
    print("Database Migration: Create datasets table")
    print("=" * 60)
    migrate()
