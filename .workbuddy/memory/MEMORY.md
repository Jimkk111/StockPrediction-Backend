# 项目长期记忆

## 项目结构
- **LSTM 股票预测系统**：基于 LSTM 的量化交易回测系统
- 后端：FastAPI + SQLAlchemy + MySQL（lstm_stock 数据库）
- 核心训练模块：`core/` 目录（data_processor.py, model.py, utils.py）
- 数据文件：`data/` 目录（nasdaq.csv, sinewave.csv, sp500.csv）
- 模型保存：`saved_models/` 目录（.h5 文件）
- 交易策略：`trading_strategy.py`（趋势追踪/EMA交叉/MACD）

## 数据库表
- `users` - 用户表
- `backtest_results` / `portfolio_snapshots` / `trade_records` - 回测相关
  - backtest_results 2026-04-24 新增字段：data_filename, training_job_id(FK→training_jobs), saved_model_path
- `training_jobs` / `training_epoch_logs` - 模型训练相关（2026-04-23 新增）
  - training_jobs 2026-04-24 新增字段：dataset_id(FK→datasets, ON DELETE SET NULL)
- `datasets` - 数据集管理（2026-04-24 新增，懒加载模式：创建时 pending，首次使用时下载）
  - rows 是 MySQL 保留字，需用 Column("rows", Integer)
- `prediction_results` / `prediction_points` - 预测功能（2026-04-24 新增）
  - prediction_results: 存储预测摘要（股票/模型/策略/次日预测价/方向/实际价/匹配状态/精度指标）
  - prediction_points: 存储预测vs实际数据点（用于图表可视化），降采样最多500点

## API 路由
- `/api/auth/*` - 注册、登录、获取用户信息
- `/api/backtest/*` - 回测执行、列表、详情、删除
- `/api/backtest/data-files` - 可用数据文件列表（2026-04-24 新增）
- `/api/training/*` - 模型训练执行、列表、详情、删除（2026-04-23 新增）
- `/api/market/search` - A 股模糊搜索（2026-04-24 新增）
- `/api/market/features` - 内置特征列表（2026-04-24 新增）
- `/api/market/kline` - K 线预览（2026-04-24 新增）
- `/api/dataset/*` - 数据集 CRUD + 下载（2026-04-24 新增）
- `/api/prediction/*` - 预测功能（2026-04-24 新增）
  - POST /run - 执行预测（选股票+模型+策略→生成次日预测）
  - GET /list - 预测结果列表
  - GET /{id} - 预测详情（含图表数据点）
  - PUT /{id}/match - 匹配实际次日收盘价
  - DELETE /{id} - 删除预测记录

## 技术要点
- 训练使用 Keras 3.x 的 `model.fit()`（非 fit_generator，已废弃）
- 模型文件命名格式：`DDMMYYYY-HHMMSS-e{epochs}.h5`
- 训练通过 WsEpochCallback（Keras Callback）记录每个 epoch 的 loss 到数据库 + 推送事件总线
- 数据库配置通过 `.env` 文件（backend/.env）
- WebSocket 实时训练进度：`TrainingEventBus` 事件总线桥接训练线程与 WS 端点
- Pydantic V2 保留字段：不能用 `model_path`（→ `saved_model_path`）和 `model_config`（→ `network_config`）
- 运行环境：FastAPI 跑在 Python 3.10.10（`C:\Users\momo\AppData\Local\Programs\Python\Python310`）
- WebSocket 依赖：需安装 `websockets` 库（uvicorn 默认不含 WebSocket 支持）
- 回测支持模型选择：通过 `training_job_id` 指定训练好的模型，`trading_strategy.py` 新增 `load_lstm_predictions_from_model()` 函数
- 回测支持数据文件选择：通过 `data_filename` 指定，优先级：请求参数 > config.json > 默认nasdaq.csv
- 数据集懒加载模式：创建时存配置(status=pending)，首次被回测/训练使用时自动下载 akshare 数据
- 数据源优先级：dataset_id > data_filename > config.json
- akshare 封装：services/market.py，列名映射 中文→英文(Date/Open/High/Low/Close/Volume...)
- **A 股搜索接口**：使用 `ak.stock_info_a_code_name()`（只返回code+name），**不用** `ak.stock_zh_a_spot_em()`（拉全量行情，易断连）
- A 股列表三级降级缓存：内存(1h) → 本地CSV(6h, backend/cache/stock_list.csv) → API+写缓存
- 搜索列名为英文 code/name（非中文 代码/名称）
- 行情数据双源降级：东财源(`stock_zh_a_hist`) → 新浪源(`stock_zh_a_daily`)
  - 新浪源 symbol 需转换：6/9开头→sh前缀，0/3开头→sz前缀
  - 新浪源降级时不复权（qfq因子请求走东财被封域名）
  - 通用 `_call_akshare()` 包装器：3次重试，2秒间隔
- 东财 `push2his.eastmoney.com` 在部分网络环境被封（校园网等），会导致东财源不可用
- 内置特征列表：GET /api/market/features，支持按类型(stock/fund)返回可选列
- akshare 版本：1.18.57
- **预测反标准化**：LSTM 输出为标准化值 `(p/base)-1`，反标准化为 `base * (1 + pred)`
- **次日预测**：取最后 seq_len-1 个数据点构造输入，预测下一个时间步
- **预测精度指标**：MAE、RMSE、方向准确率（direction accuracy）
- database.py 使用 `ensure_models_registered()` 函数延迟导入模型，避免循环依赖
- 数据库迁移脚本：`backend/migrate_backtest_add_model.py`（已执行）
- 数据库迁移脚本：`backend/migrate_add_datasets.py`（已执行）
- 数据库迁移脚本：`backend/migrate_training_add_dataset.py`（已执行，training_jobs 添加 dataset_id FK）
- 数据库迁移脚本：`backend/migrate_add_predictions.py`（已执行）

## 数据库迁移记录
- backtest_results 表新增 data_filename、training_job_id、saved_model_path 列（migrate_backtest_add_model.py）
- training_jobs/training_epoch_logs 表曾因字段重命名（model_config→network_config）需 DROP 重建
- training_jobs 表新增 dataset_id 列（FK→datasets.id, ON DELETE SET NULL）（migrate_training_add_dataset.py）
- prediction_results + prediction_points 表新建（migrate_add_predictions.py）

## 用户偏好
- 中文交流
- 关注交互细节和视觉状态一致性
