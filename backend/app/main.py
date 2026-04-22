import os
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')
load_dotenv(env_path)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.auth import router as auth_router
from app.api.backtest import router as backtest_router

app = FastAPI(
    title="LSTM Stock Prediction Backend",
    description="LSTM股票预测系统后端API - 包含用户鉴权与交易策略回测功能",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix="/api")
app.include_router(backtest_router, prefix="/api")


@app.get("/", tags=["健康检查"])
def root():
    return {"message": "LSTM Stock Prediction Backend is running", "version": "1.0.0"}


@app.get("/health", tags=["健康检查"])
def health_check():
    return {"status": "ok"}
