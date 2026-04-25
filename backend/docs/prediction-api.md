# 预测功能 API 接口文档

> 基础路径：`/api/prediction`  
> 所有接口需要 Bearer Token 鉴权（Authorization: Bearer \<token\>）

---

## 1. 执行预测 `POST /api/prediction/run`

选择股票 + 训练好的模型 + 交易策略，生成次日预测结果并返回完整详情（含图表数据点）。

### 请求体

```json
{
  "training_job_id": 1,                    // [必填] 训练任务ID
  "strategy_type": "trend_rider",          // [可选] 策略类型，默认 trend_rider
  "strategy_params": { ... },              // [可选] 策略参数覆盖
  "dataset_id": 3,                         // [可选] 数据集ID（优先级最高）
  "data_filename": "nasdaq.csv",           // [可选] 数据文件名
  "stock_code": "600519",                  // [可选] 股票代码
  "stock_name": "贵州茅台"                  // [可选] 股票名称
}
```

### 字段说明

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `training_job_id` | int | **是** | 指定已完成的训练任务，用其模型做预测 |
| `strategy_type` | string | 否 | 策略类型：`trend_rider` / `ema_cross` / `macd`，默认 `trend_rider` |
| `strategy_params` | object | 否 | 覆盖策略默认参数，见下方策略参数表 |
| `dataset_id` | int | 否 | 数据集ID，首次使用自动下载（**优先级最高**） |
| `data_filename` | string | 否 | data 目录下的 CSV 文件名 |
| `stock_code` | string | 否 | 股票代码，如 `600519`；不传则从 dataset 推断 |
| `stock_name` | string | 否 | 股票名称，如 `贵州茅台`；不传则从 dataset 推断 |

> **数据源优先级**：`dataset_id` > `data_filename` > `config.json` 默认配置

### 三种策略默认参数

```js
// trend_rider（趋势骑手）
{
  "trend_ema": 50, "confirm_ema": 20,
  "bull_trail": 0.15, "bear_trail": 0.03,
  "stop_loss_pct": 0.12, "rsi_period": 14
}

// ema_cross（EMA交叉）
{
  "fast_ema": 8, "slow_ema": 21,
  "bull_trail": 0.15, "bear_trail": 0.03,
  "stop_loss_pct": 0.12, "rsi_period": 14
}

// macd（MACD策略）
{
  "trend_ema": 50,
  "bull_trail": 0.15, "bear_trail": 0.03,
  "stop_loss_pct": 0.12, "rsi_period": 14
}
```

### 请求示例

**示例 A：使用数据集 + 指定模型**
```json
{
  "training_job_id": 5,
  "strategy_type": "ema_cross",
  "dataset_id": 3
}
```

**示例 B：使用本地 CSV 文件**
```json
{
  "training_job_id": 2,
  "strategy_type": "macd",
  "data_filename": "nasdaq.csv"
}
```

**示例 C：自定义策略参数**
```json
{
  "training_job_id": 1,
  "strategy_type": "trend_rider",
  "dataset_id": 3,
  "strategy_params": {
    "trend_ema": 40,
    "stop_loss_pct": 0.10
  }
}
```

### 响应体 (PredictionDetail)

```json
{
  "id": 1,
  "stock_code": "600519",
  "stock_name": "贵州茅台",
  "data_source": "dataset:贵州茅台(3)",
  "training_job_id": 5,
  "saved_model_path": "saved_models/24042026-220000-e50.h5",
  "sequence_length": 50,
  "strategy_type": "ema_cross",
  "strategy_params": { "fast_ema": 8, "slow_ema": 21, "bull_trail": 0.15, "bear_trail": 0.03, "stop_loss_pct": 0.12, "rsi_period": 14 },
  "predicted_date": "2026-04-24",
  "predicted_price": 1685.32,
  "predicted_direction": 1,
  "predicted_signal": 1,
  "last_close": 1680.50,
  "actual_price": null,
  "actual_direction": null,
  "is_matched": 0,
  "mae": 12.35,
  "rmse": 18.67,
  "direction_accuracy": 0.62,
  "n_points": 450,
  "status": "completed",
  "error_message": null,
  "created_at": "2026-04-24T22:10:00",
  "updated_at": "2026-04-24T22:10:05",
  "points": [
    {
      "date_index": 49,
      "date_label": "2024-01-15",
      "actual_close": 1650.00,
      "predicted_close": 1648.20,
      "predicted_norm": -0.00109,
      "signal": 0
    },
    {
      "date_index": 50,
      "date_label": "2024-01-16",
      "actual_close": 1655.50,
      "predicted_close": 1653.80,
      "predicted_norm": 0.00103,
      "signal": 1
    }
    // ... 最多 500 个数据点
  ]
}
```

### 响应字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | int | 预测记录 ID |
| `stock_code` | string? | 股票代码 |
| `stock_name` | string? | 股票名称 |
| `data_source` | string? | 数据来源标识 |
| `training_job_id` | int? | 关联的训练任务 ID |
| `sequence_length` | int? | 模型序列长度 |
| `strategy_type` | string | 使用的策略类型 |
| `strategy_params` | object? | 实际使用的策略参数 |
| **`predicted_date`** | **string?** | **预测的目标日期（次日）** |
| **`predicted_price`** | **float?** | **次日预测价格（已反标准化为真实价格）** |
| **`predicted_direction`** | **int?** | **预测方向：1=涨, -1=跌, 0=平** |
| **`predicted_signal`** | **int?** | **策略信号：1=买入, -1=卖出, 0=持有** |
| **`last_close`** | **float?** | **预测基准价（最后一日收盘价）** |
| `actual_price` | float? | 实际次日收盘价（匹配后填充） |
| `actual_direction` | int? | 实际次日方向（匹配后填充） |
| `is_matched` | int | 是否已匹配：0=未匹配, 1=已匹配 |
| `mae` | float? | 平均绝对误差 |
| `rmse` | float? | 均方根误差 |
| `direction_accuracy` | float? | 方向预测准确率（0~1） |
| `n_points` | int? | 预测数据点总数 |
| `status` | string | 状态：completed / failed |
| `points` | array | 图表数据点列表（降采样，最多 500 个） |

### points 数组字段

| 字段 | 类型 | 图表用途 |
|------|------|----------|
| `date_label` | string | X 轴日期 |
| `actual_close` | float | 真实收盘价折线 |
| `predicted_close` | float | 预测收盘价折线 |
| `signal` | int | 买卖标记：1=买点标记, -1=卖点标记, 0=无操作 |

---

## 2. 预测结果列表 `GET /api/prediction/list`

### 请求参数（Query）

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `skip` | int | 0 | 跳过条数 |
| `limit` | int | 20 | 每页条数（1~100） |

### 请求示例

```
GET /api/prediction/list?skip=0&limit=10
```

### 响应体

```json
{
  "total": 15,
  "items": [
    {
      "id": 3,
      "stock_code": "600519",
      "stock_name": "贵州茅台",
      "data_source": "dataset:贵州茅台(3)",
      "training_job_id": 5,
      "strategy_type": "ema_cross",
      "predicted_date": "2026-04-25",
      "predicted_price": 1690.00,
      "predicted_direction": 1,
      "predicted_signal": 1,
      "last_close": 1685.32,
      "actual_price": null,
      "actual_direction": null,
      "is_matched": 0,
      "mae": 10.5,
      "rmse": 15.2,
      "direction_accuracy": 0.65,
      "n_points": 450,
      "created_at": "2026-04-24T22:10:00"
    }
    // ...
  ]
}
```

---

## 3. 预测详情 `GET /api/prediction/{id}`

获取预测详情，包含完整的图表数据点。**用于渲染预测 vs 实际对比图。**

### 请求示例

```
GET /api/prediction/3
```

### 响应体

与 `POST /run` 的响应结构相同（PredictionDetail），此处不再重复。

---

## 4. 匹配实际数据 `PUT /api/prediction/{id}/match`

当次日收盘价出来后，用实际价格更新预测记录，计算预测准确度。

### 请求体

```json
{
  "actual_price": 1688.50
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `actual_price` | float | **是** | 次日实际收盘价 |

### 请求示例

```
PUT /api/prediction/3/match
Content-Type: application/json

{
  "actual_price": 1688.50
}
```

### 响应体

返回更新后的 PredictionDetail，其中 `actual_price`、`actual_direction`、`is_matched` 已填充：

```json
{
  "id": 3,
  "stock_code": "600519",
  "stock_name": "贵州茅台",
  "predicted_price": 1690.00,
  "predicted_direction": 1,
  "last_close": 1685.32,
  "actual_price": 1688.50,
  "actual_direction": 1,
  "is_matched": 1,
  // ... 其他字段同上
}
```

> 后端自动计算 `actual_direction`：`actual_price > last_close ? 1 : -1`

---

## 5. 删除预测记录 `DELETE /api/prediction/{id}`

### 请求示例

```
DELETE /api/prediction/3
```

### 响应体

```json
{
  "message": "删除成功"
}
```

---

## 前端接入指南

### 典型页面流程

```
[预测页面]
  │
  ├─ Step 1: 用户选择股票
  │   └─ 调用 GET /api/market/search 搜索
  │   └─ 选择后创建 Dataset（或用已有 Dataset）
  │
  ├─ Step 2: 用户选择训练好的模型
  │   └─ 调用 GET /api/training/list 获取已完成的训练任务
  │
  ├─ Step 3: 用户选择交易策略
  │   └─ 三选一：trend_rider / ema_cross / macd
  │
  ├─ Step 4: 点击"执行预测"
  │   └─ 调用 POST /api/prediction/run
  │   └─ 返回 PredictionDetail（含图表数据）
  │
  └─ Step 5: 展示结果
      ├─ 次日预测价格 + 涨跌方向 + 策略信号
      ├─ 预测 vs 实际 对比折线图（用 points 数据）
      ├─ 精度指标卡片（MAE / RMSE / 方向准确率）
      └─ 买卖信号标记图
```

### 图表数据渲染

`points` 数组直接用于前端图表库（ECharts / Recharts 等）：

```jsx
// React + Recharts 示例
<LineChart data={detail.points}>
  <XAxis dataKey="date_label" />
  <YAxis />
  
  {/* 真实收盘价折线 */}
  <Line type="monotone" dataKey="actual_close" stroke="#666" name="实际价格" />
  
  {/* 预测收盘价折线 */}
  <Line type="monotone" dataKey="predicted_close" stroke="#3b82f6" name="预测价格" />
  
  {/* 买卖信号标记 - 用 Scatter 叠加 */}
</LineChart>
```

### 信号标记渲染

`signal` 字段可以用来在图表上标记买卖点：

```js
const buyPoints = detail.points.filter(p => p.signal === 1);
const sellPoints = detail.points.filter(p => p.signal === -1);
```

### 预测结果卡片

```jsx
// 次日预测摘要卡片
<Card>
  <div>预测股票：{detail.stock_name} ({detail.stock_code})</div>
  <div>次日预测价：¥{detail.predicted_price?.toFixed(2)}</div>
  <div>
    预测方向：
    {detail.predicted_direction === 1 ? '📈 上涨' : 
     detail.predicted_direction === -1 ? '📉 下跌' : '➡️ 持平'}
  </div>
  <div>
    策略信号：
    {detail.predicted_signal === 1 ? '🟢 建议买入' : 
     detail.predicted_signal === -1 ? '🔴 建议卖出' : '⚪ 持有观望'}
  </div>
  <div>基准收盘价：¥{detail.last_close?.toFixed(2)}</div>
  
  {/* 匹配后显示 */}
  {detail.is_matched === 1 && (
    <div>
      <div>实际收盘价：¥{detail.actual_price?.toFixed(2)}</div>
      <div>
        预测{detail.predicted_direction === detail.actual_direction ? '✅ 正确' : '❌ 错误'}
      </div>
    </div>
  )}
</Card>

// 精度指标卡片
<Card>
  <div>MAE（平均绝对误差）：{detail.mae?.toFixed(2)}</div>
  <div>RMSE（均方根误差）：{detail.rmse?.toFixed(2)}</div>
  <div>方向准确率：{(detail.direction_accuracy * 100).toFixed(1)}%</div>
</Card>
```

### 匹配实际数据

次日收盘后，用户可以填入实际价格来验证预测：

```js
// 前端调用
const matchResult = await fetch(`/api/prediction/${predictionId}/match`, {
  method: 'PUT',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${token}`
  },
  body: JSON.stringify({ actual_price: 1688.50 })
});
```

### 错误码

| HTTP 状态码 | 场景 |
|-------------|------|
| 400 | 策略类型不支持 / 训练任务未完成 / 模型文件缺失 |
| 404 | 训练任务不存在 / 数据集不存在 / 预测记录不存在 |
| 500 | 模型加载失败 / 策略计算失败 / 数据读取失败 |
