# LSTM 神经网络时间序列预测与股票交易系统

基于 **LSTM（长短期记忆神经网络）** 的股票价格预测与量化交易回测系统。支持多种交易策略、LSTM 信号增强、异步训练、WebSocket 实时进度推送，并提供完整的 RESTful API。

---

## 功能特性

### 深度学习预测
- 基于 Keras/TensorFlow 构建多层 LSTM 神经网络
- 支持自定义网络结构（层数、神经元数、Dropout）
- 时间序列滑窗处理与归一化
- 异步训练任务，WebSocket 实时推送训练进度
- 模型自动保存与早停机制

### 交易策略
支持三种内置策略，均可选配 LSTM 预测信号增强：

| 策略 | 说明 |
|------|------|
| **Trend Rider（趋势追踪）** | 基于 EMA 趋势跟踪，附带 trailing stop-loss，辅助 EMA 确认 |
| **EMA Cross（均线交叉）** | 快慢 EMA 金叉/死叉信号 |
| **MACD** | MACD 快线/信号线交叉信号 |

### 回测引擎
- 自动遍历 500+ 参数组合，按收益率/回撤评分优选
- 生成三面板回测图表（价格与交易信号、组合净值、回撤曲线）
- 记录完整的交易记录与组合快照
- 支持基准对比

### RESTful API
- 市场数据查询（A 股搜索、K 线数据）
- 数据集管理（创建、下载、删除，支持延迟加载）
- 训练任务管理（启动、查询、删除）
- 预测管理（生成预测、准确性评估）
- 回测管理（执行、查询、删除）
- **API 文档**：启动后端后访问 `http://localhost:8000/docs`

---

## 技术栈

| 技术 | 版本 | 用途 |
|------|------|------|
| Python | 3.10+ | 运行时 |
| Keras | 3.x | 神经网络框架 |
| TensorFlow | (Keras 后端) | 深度学习引擎 |
| FastAPI | 0.115+ | REST API 框架 |
| Uvicorn | 0.30+ | ASGI 服务器 |
| SQLAlchemy | 2.0+ | ORM / 数据库抽象 |
| PyMySQL | 1.1+ | MySQL 驱动 |
| Pandas | 2.2+ | 数据处理 |
| NumPy | 1.26+ | 数值计算 |
| Matplotlib | 3.x | 回测图表生成 |
| akshare | latest | A 股市场数据接口 |
| WebSocket | (via Uvicorn) | 实时训练进度推送 |

---

## 项目结构

```
├── backend/                          # FastAPI 后端
│   ├── app/
│   │   ├── main.py                   # 应用入口
│   │   ├── config.py                 # 环境配置
│   │   ├── database.py               # 数据库连接
│   │   ├── api/                      # 路由层
│   │   │   ├── market.py             # 市场数据 API
│   │   │   ├── training.py           # 训练 API + WebSocket
│   │   │   ├── prediction.py         # 预测 API
│   │   │   ├── backtest.py           # 回测 API
│   │   │   └── dataset.py            # 数据集 API
│   │   ├── models/                   # SQLAlchemy ORM 模型
│   │   ├── schemas/                  # Pydantic 请求/响应模型
│   │   └── services/                 # 业务逻辑层
│   ├── .env                          # 数据库配置
│   ├── init_db.py                    # 数据库初始化脚本
│   ├── docs/prediction-api.md        # API 文档
│   └── requirements.txt              # 后端依赖
├── core/                             # LSTM 核心模块
│   ├── data_processor.py             # 数据加载、滑窗、归一化
│   ├── model.py                      # LSTM 模型构建/训练/预测
│   └── utils.py                      # 工具函数
├── data/                             # CSV 数据集
│   ├── nasdaq.csv                    # 纳斯达克指数日线
│   ├── sinewave.csv                  # 正弦波合成数据
│   ├── sp500.csv                     # 标普 500 指数
│   └── dataset_1_600519.csv          # 贵州茅台（600519）A 股数据
├── saved_models/                     # 训练好的 .h5 模型文件
├── config.json                       # 训练配置
├── run.py                            # 原始 CLI 训练入口
├── trading_strategy.py               # 交易策略 + 回测引擎
├── fetching_data.py                  # 数据获取脚本
├── fetch_nasdaq.py                   # 纳斯达克数据获取
└── requirements.txt                  # 核心依赖
```

---

## 快速开始

### 环境要求
- Python 3.10+
- MySQL 服务器
- 虚拟环境（推荐）

### 1. 安装依赖

```bash
# 核心依赖（LSTM 训练 + 回测）
pip install -r requirements.txt

# 后端依赖（FastAPI + 数据库）
pip install -r backend/requirements.txt
```

### 2. 配置数据库

编辑 `backend/.env`，填入你的 MySQL 连接信息：

```env
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=123456
MYSQL_DATABASE=lstm_stock
```

### 3. 初始化数据库

```bash
cd backend
python init_db.py
```

此脚本会自动创建数据库、所有表，并创建一个测试用户（`test` / `test123`）。

### 4. 启动后端服务

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

访问 `http://localhost:8000/docs` 查看交互式 API 文档。

### 5. CLI 方式训练

```bash
# 激活虚拟环境
source .venv/Scripts/activate  # Windows
# 或 source .venv/bin/activate  # Linux/Mac

# 使用 config.json 配置训练
python run.py
```

### 6. 运行独立回测

```bash
python trading_strategy.py
```

自动加载 `saved_models/` 中的最新模型，遍历 500+ 参数组合，将最优结果图表保存为 `trading_results.png`。

### 7. 获取 NASDAQ 数据

```bash
python fetch_nasdaq.py
```

---

## API 概览

所有 API 端点前缀为 `/api`。

### 市场数据

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/market/search` | 模糊搜索 A 股股票 |
| GET | `/api/market/features` | 可用特征列列表 |
| GET | `/api/market/kline` | 预览 K 线数据 |

### 数据集管理

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/dataset/create` | 创建数据集（延迟加载） |
| GET | `/api/dataset/list` | 数据集列表 |
| POST | `/api/dataset/{id}/download` | 触发数据下载 |
| DELETE | `/api/dataset/{id}` | 删除数据集 |

### 训练

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/training/run` | 启动异步训练任务 |
| GET | `/api/training/list` | 训练历史列表 |
| GET | `/api/training/{id}` | 训练详情 + Epoch 日志 |
| DELETE | `/api/training/{id}` | 删除训练记录 |
| WebSocket | `/api/training/{id}/ws` | 实时训练进度推送 |

### 预测

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/prediction/run` | 执行预测 |
| GET | `/api/prediction/list` | 预测结果列表 |
| GET | `/api/prediction/{id}` | 预测详情 |
| PUT | `/api/prediction/{id}/match` | 填入真实收盘价评估准确率 |
| DELETE | `/api/prediction/{id}` | 删除预测记录 |

### 回测

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/backtest/run` | 执行回测 |
| GET | `/api/backtest/list` | 回测结果列表 |
| GET | `/api/backtest/{id}` | 回测详情（净值快照 + 交易记录） |
| DELETE | `/api/backtest/{id}` | 删除回测记录 |

---

## 配置说明

`config.json` 主要配置项：

```json
{
  "data": {
    "filename": "data/nasdaq.csv",
    "columns": ["close"],
    "sequence_length": 10,
    "train_test_split": 0.95
  },
  "training": {
    "epochs": 100,
    "batch_size": 32,
    "loss": "mse",
    "optimizer": "adam"
  },
  "model": {
    "layers": [
      {"type": "lstm", "neurons": 100, "activation": "tanh", "return_seq": true},
      {"type": "dropout", "rate": 0.2},
      {"type": "lstm", "neurons": 100, "activation": "tanh", "return_seq": true},
      {"type": "lstm", "neurons": 100, "activation": "tanh", "return_seq": false},
      {"type": "dropout", "rate": 0.2},
      {"type": "dense", "neurons": 1, "activation": "linear"}
    ]
  }
}
```

---

## 许可证

本项目基于 [MIT License](LICENSE) 开源。

原始项目来源于 [Altum Intelligence 文章](https://www.altumintelligence.com/articles/a/Time-Series-Prediction-Using-LSTM-Deep-Neural-Networks) 及配套代码。
