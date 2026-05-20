import os
import asyncio
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')
load_dotenv(env_path)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.backtest import router as backtest_router
from app.api.training import router as training_router
from app.api.market import router as market_router
from app.api.dataset import router as dataset_router
from app.api.prediction import router as prediction_router
from app.services.training import TrainingEventBus

app = FastAPI(
    title="LSTM Stock Prediction Backend",
    description="LSTM股票预测系统后端API - 包含交易策略回测与模型训练功能",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(backtest_router, prefix="/api")
app.include_router(training_router, prefix="/api")
app.include_router(market_router, prefix="/api")
app.include_router(dataset_router, prefix="/api")
app.include_router(prediction_router, prefix="/api")


@app.on_event("startup")
async def on_startup():
    """保存 asyncio 事件循环引用，供训练线程跨线程推送事件"""
    TrainingEventBus.set_loop(asyncio.get_event_loop())


@app.get("/", tags=["健康检查"])
def root():
    return {"message": "LSTM Stock Prediction Backend is running", "version": "1.0.0"}


@app.get("/health", tags=["健康检查"])
def health_check():
    return {"status": "ok"}
