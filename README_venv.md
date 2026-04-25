# LSTM股票预测项目 - 虚拟环境使用说明

## 项目结构

```
LSTM-Neural-Network-for-Time-Series-Prediction/
├── venv/                    # 虚拟环境（已移动至此）
├── core/                    # 核心代码
├── data/                    # 数据文件
├── saved_models/            # 训练好的模型
├── activate_venv.bat        # Windows批处理启动脚本
├── activate_venv.ps1        # PowerShell启动脚本
├── run.py                   # 主程序
├── config.json              # 配置文件
└── requirements.txt         # 依赖包列表
```

## 快速启动方式

### 方式一：使用启动脚本（推荐）

**Windows批处理文件（双击运行）:**
```
双击 activate_venv.bat
```

**PowerShell脚本（右键选择"使用PowerShell运行"）:**
```
双击 activate_venv.ps1
```

### 方式二：手动启动

```bash
# 激活虚拟环境
venv\Scripts\activate.bat

# 运行项目
python run.py
```

## 虚拟环境管理

### 创建新的虚拟环境（如果需要）

```bash
# 使用Python 3.10创建虚拟环境
py -3.10 -m venv venv

# 激活虚拟环境
venv\Scripts\activate.bat

# 安装依赖
pip install -r requirements.txt
```

### 更新依赖

```bash
# 激活虚拟环境后
pip install --upgrade -r requirements.txt
```

## Git忽略规则

虚拟环境相关文件已被添加到 `.gitignore`，不会被Git跟踪：
- `venv/` - 虚拟环境目录
- `*.pyc` - Python编译文件
- `__pycache__/` - 缓存目录
- 模型文件、日志文件等

## 注意事项

1. **虚拟环境已包含所有依赖**，无需重新安装
2. **启动脚本会自动激活虚拟环境**并运行项目
3. **项目配置**可通过 `config.json` 文件修改
4. **训练好的模型**保存在 `saved_models/` 目录

## 故障排除

如果启动失败，请检查：
1. 虚拟环境是否存在（`venv/` 目录）
2. Python版本是否为3.10
3. 依赖包是否完整安装

如需重新创建虚拟环境，请参考上面的"创建新的虚拟环境"部分。