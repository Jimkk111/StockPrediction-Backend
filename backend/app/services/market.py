"""
Market Service: akshare 封装 + 特征列表定义

职责：
- 搜索股票（模糊匹配代码/名称）
- 获取 K 线行情数据（预览用，不落盘）
- 提供内置特征列表
"""
import os
import time
import logging
from typing import List, Optional

import akshare as ak
import pandas as pd

from app.schemas.market import (
    FeatureInfo, FeatureListResponse, StockSearchResult, StockSearchResponse,
    KlineDataItem, KlineResponse,
)

logger = logging.getLogger(__name__)


# ==================== akshare 通用调用包装 ====================

_AK_MAX_RETRIES = 3
_AK_RETRY_DELAY = 2  # 重试间隔（秒）


def _call_akshare(func, *args, **kwargs):
    """
    通用 akshare API 调用包装，带重试和超时。
    akshare 的网络请求容易因东财服务器断连而失败，
    统一加重试逻辑避免单次网络抖动导致 500。
    """
    last_err = None
    for attempt in range(1, _AK_MAX_RETRIES + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_err = e
            logger.warning(f"akshare 调用失败（第 {attempt}/{_AK_MAX_RETRIES} 次）"
                           f" func={func.__name__}: {e}")
            if attempt < _AK_MAX_RETRIES:
                time.sleep(_AK_RETRY_DELAY)
    raise RuntimeError(f"akshare 调用失败，已重试 {_AK_MAX_RETRIES} 次: {last_err}")


# 本地 CSV 缓存路径（持久化，避免每次启动都重新拉取）
_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "cache")
_STOCK_LIST_CSV = os.path.join(_CACHE_DIR, "stock_list.csv")


# ==================== 列名映射 ====================

# akshare 中文列名 → 项目标准英文列名（东财源 stock_zh_a_hist）
COLUMN_MAP = {
    "日期": "Date",
    "股票代码": "Symbol",
    "开盘": "Open",
    "收盘": "Close",
    "最高": "High",
    "最低": "Low",
    "成交量": "Volume",
    "成交额": "Amount",
    "振幅": "Amplitude",
    "涨跌幅": "PctChange",
    "涨跌额": "Change",
    "换手率": "Turnover",
}

# 新浪源 stock_zh_a_daily 的列名映射（已是英文，只做小写→首字母大写对齐）
COLUMN_MAP_SINA = {
    "date": "Date",
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "volume": "Volume",
    "amount": "Amount",
    "turnover": "Turnover",
    # 新浪源没有 Amplitude/PctChange/Change，缺失列不映射
}

# 新浪源 symbol 格式：6开头加 sh，0/3开头加 sz
def _to_sina_symbol(symbol: str) -> str:
    """将纯数字代码转为新浪源格式：000001 → sz000001，600519 → sh600519"""
    if symbol.startswith(("sh", "sz", "SH", "SZ")):
        return symbol.lower()
    if symbol.startswith(("6", "9")):
        return f"sh{symbol}"
    return f"sz{symbol}"

# 英文列名 → KlineDataItem 字段名（全小写）
COLUMN_TO_FIELD = {
    "Date": "date",
    "Open": "open",
    "Close": "close",
    "High": "high",
    "Low": "low",
    "Volume": "volume",
    "Amount": "amount",
    "Amplitude": "amplitude",
    "PctChange": "pct_change",
    "Change": "change",
    "Turnover": "turnover",
}


# ==================== 内置特征列表 ====================

def get_feature_list(data_type: str = "stock") -> FeatureListResponse:
    """
    获取内置特征列表。
    data_type 预留扩展：stock / fund
    """
    stock_features = [
        FeatureInfo(key="Date",      name="日期",   category="基础", selectable=False),
        FeatureInfo(key="Open",      name="开盘价", category="价格", selectable=True),
        FeatureInfo(key="High",      name="最高价", category="价格", selectable=True),
        FeatureInfo(key="Low",       name="最低价", category="价格", selectable=True),
        FeatureInfo(key="Close",     name="收盘价", category="价格", selectable=True, default_selected=True),
        FeatureInfo(key="Volume",    name="成交量", category="量价", selectable=True, default_selected=True),
        FeatureInfo(key="Amount",    name="成交额", category="量价", selectable=True),
        FeatureInfo(key="Turnover",  name="换手率", category="量价", selectable=True),
        FeatureInfo(key="PctChange", name="涨跌幅", category="变动", selectable=True),
        FeatureInfo(key="Change",    name="涨跌额", category="变动", selectable=True),
        FeatureInfo(key="Amplitude", name="振幅",   category="变动", selectable=True),
    ]

    if data_type == "fund":
        # TODO: 基金特征列表，后续扩展
        fund_features = [
            FeatureInfo(key="Date",      name="日期",   category="基础", selectable=False),
            FeatureInfo(key="Open",      name="开盘价", category="价格", selectable=True),
            FeatureInfo(key="High",      name="最高价", category="价格", selectable=True),
            FeatureInfo(key="Low",       name="最低价", category="价格", selectable=True),
            FeatureInfo(key="Close",     name="收盘价", category="价格", selectable=True, default_selected=True),
            FeatureInfo(key="Volume",    name="成交量", category="量价", selectable=True, default_selected=True),
            FeatureInfo(key="Amount",    name="成交额", category="量价", selectable=True),
            FeatureInfo(key="PctChange", name="涨跌幅", category="变动", selectable=True),
        ]
        return FeatureListResponse(type="fund", features=fund_features)

    return FeatureListResponse(type="stock", features=stock_features)


# ==================== A 股列表缓存 ====================

_stock_list_cache = {"df": None, "ts": 0}
_CACHE_TTL = 3600  # 1 小时内存缓存


def _save_stock_list_csv(df: pd.DataFrame):
    """持久化股票列表到本地 CSV"""
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        df.to_csv(_STOCK_LIST_CSV, index=False, encoding="utf-8-sig")
        logger.info(f"股票列表已缓存到 {_STOCK_LIST_CSV}，共 {len(df)} 条")
    except Exception as e:
        logger.warning(f"保存股票列表缓存失败: {e}")


def _load_stock_list_csv() -> Optional[pd.DataFrame]:
    """从本地 CSV 加载股票列表"""
    if not os.path.exists(_STOCK_LIST_CSV):
        return None
    try:
        df = pd.read_csv(_STOCK_LIST_CSV, dtype=str)
        if len(df) > 0 and "code" in df.columns and "name" in df.columns:
            logger.info(f"从本地缓存加载股票列表，共 {len(df)} 条")
            return df
    except Exception as e:
        logger.warning(f"读取本地股票列表缓存失败: {e}")
    return None


def _fetch_stock_list_from_api() -> pd.DataFrame:
    """从 akshare 拉取 A 股列表"""
    try:
        df = _call_akshare(ak.stock_info_a_code_name)
        if df is not None and not df.empty:
            # 统一列名为 code / name
            df = df.rename(columns={"code": "code", "name": "name"})
            df = df.dropna(subset=["code", "name"]).drop_duplicates(subset=["code"])
            return df
    except Exception as e:
        raise RuntimeError(f"获取 A 股列表失败: {e}")
    raise RuntimeError("获取 A 股列表失败: 返回数据为空")


def _get_stock_list() -> pd.DataFrame:
    """
    获取 A 股列表，三级降级策略：
    1. 内存缓存（1 小时内有效）
    2. 本地 CSV 持久化缓存
    3. akshare API 实时拉取 + 写入缓存
    """
    now = time.time()

    # 第一级：内存缓存
    if _stock_list_cache["df"] is not None and now - _stock_list_cache["ts"] < _CACHE_TTL:
        return _stock_list_cache["df"]

    # 第二级：本地 CSV 缓存（如果 CSV 在 6 小时内更新过，直接用）
    if os.path.exists(_STOCK_LIST_CSV):
        csv_age = now - os.path.getmtime(_STOCK_LIST_CSV)
        if csv_age < 6 * 3600:
            df = _load_stock_list_csv()
            if df is not None:
                _stock_list_cache["df"] = df
                _stock_list_cache["ts"] = now
                return df

    # 第三级：API 拉取
    try:
        df = _fetch_stock_list_from_api()
        _stock_list_cache["df"] = df
        _stock_list_cache["ts"] = now
        _save_stock_list_csv(df)
        return df
    except Exception as e:
        logger.error(f"API 拉取 A 股列表失败: {e}")

        # 降级：即使 CSV 过期也用
        df = _load_stock_list_csv()
        if df is not None:
            _stock_list_cache["df"] = df
            _stock_list_cache["ts"] = now
            return df

        # 再降级：用过期的内存缓存
        if _stock_list_cache["df"] is not None:
            return _stock_list_cache["df"]

        raise RuntimeError(f"无法获取 A 股列表（API 失败 + 无缓存）: {e}")


def search_stocks(keyword: str, limit: int = 20) -> StockSearchResponse:
    """模糊搜索股票代码或名称"""
    if not keyword or len(keyword.strip()) == 0:
        return StockSearchResponse(items=[])

    df = _get_stock_list()
    keyword = keyword.strip()

    mask = df["code"].str.contains(keyword, na=False) | df["name"].str.contains(keyword, na=False)
    matched = df[mask].head(limit)

    items = [
        StockSearchResult(code=str(row["code"]), name=str(row["name"]))
        for _, row in matched.iterrows()
    ]
    return StockSearchResponse(items=items)


# ==================== K 线数据获取 ====================

def _fetch_stock_hist(
    symbol: str,
    period: str = "daily",
    start_date: str = "",
    end_date: str = "",
    adjust: str = "qfq",
) -> pd.DataFrame:
    """
    获取 A 股历史行情，双源降级策略：
    1. 东财源（stock_zh_a_hist）—— 数据全（含振幅/涨跌幅等），但易断连
    2. 新浪源（stock_zh_a_daily）—— 数据较少但更稳定
    """
    # 第一选择：东财源
    try:
        df = _call_akshare(
            ak.stock_zh_a_hist,
            symbol=symbol, period=period,
            start_date=start_date, end_date=end_date,
            adjust=adjust,
        )
        if df is not None and not df.empty:
            df = df.rename(columns=COLUMN_MAP)
            logger.info(f"东财源获取行情成功: {symbol}, {len(df)} 条")
            return df
    except Exception as e:
        logger.warning(f"东财源获取行情失败，降级到新浪源: {symbol}, error={e}")

    # 第二选择：新浪源（只支持日线，不复权，避免 qfq 因子请求也走东财被封域名）
    try:
        sina_symbol = _to_sina_symbol(symbol)
        # 新浪源 adjust="" 不复权，数据仍可用；qfq 会额外请求东财接口导致超时
        df = _call_akshare(
            ak.stock_zh_a_daily,
            symbol=sina_symbol,
            start_date=start_date,
            end_date=end_date,
            adjust="",  # 不复权，保稳定性
        )
        if df is not None and not df.empty:
            df = df.rename(columns=COLUMN_MAP_SINA)
            logger.info(f"新浪源获取行情成功（不复权）: {symbol}, {len(df)} 条")
            return df
    except Exception as e:
        logger.warning(f"新浪源获取行情也失败: {symbol}, error={e}")

    raise RuntimeError(f"无法获取行情数据（东财源+新浪源均失败）: symbol={symbol}")


def get_kline(
    symbol: str,
    period: str = "daily",
    start_date: str = "",
    end_date: str = "",
    adjust: str = "qfq",
    columns: Optional[List[str]] = None,
) -> KlineResponse:
    """
    获取股票 K 线数据。

    Parameters:
        symbol: 股票代码，如 "600519"
        period: daily / weekly / monthly
        start_date: YYYYMMDD
        end_date: YYYYMMDD
        adjust: qfq / hfq / ""
        columns: 需要返回的英文列名列表，不传则返回全部可选列
    """
    # 1. 获取原始数据（双源降级）
    try:
        df = _fetch_stock_hist(
            symbol=symbol, period=period,
            start_date=start_date, end_date=end_date,
            adjust=adjust,
        )
    except RuntimeError as e:
        logger.error(f"获取行情失败（双源均不可用）: symbol={symbol}, error={e}")
        raise

    if df.empty:
        return KlineResponse(
            symbol=symbol, name="", period=period,
            count=0, columns=[], data=[],
        )

    # 2. 列名映射（中文 → 英文）
    df = df.rename(columns=COLUMN_MAP)

    # 3. 获取股票名称（从第一行的 akshare 数据中取，或用 A 股列表查）
    symbol_name = _lookup_symbol_name(symbol)

    # 4. 确定要返回的列
    available_cols = [c for c in df.columns if c != "Symbol"]  # 去掉股票代码列
    if columns:
        # Date 始终包含，Symbol 不返回
        selected_cols = ["Date"] + [c for c in columns if c in available_cols and c != "Date"]
    else:
        selected_cols = available_cols

    # 5. 只保留选中列
    output_cols = [c for c in selected_cols if c in df.columns]
    df_out = df[output_cols].copy()

    # 6. 转为响应格式
    data_items = []
    for _, row in df_out.iterrows():
        item = {}
        for col in output_cols:
            field_name = COLUMN_TO_FIELD.get(col, col.lower())
            val = row[col]
            # 日期保持字符串，数值转 float
            if col == "Date":
                item[field_name] = str(val)
            else:
                try:
                    item[field_name] = float(val) if pd.notna(val) else None
                except (ValueError, TypeError):
                    item[field_name] = None
        data_items.append(KlineDataItem(**item))

    return KlineResponse(
        symbol=symbol,
        name=symbol_name,
        period=period,
        count=len(data_items),
        columns=output_cols,
        data=data_items,
    )


def _lookup_symbol_name(symbol: str) -> str:
    """从 A 股列表中查找股票名称"""
    try:
        df = _get_stock_list()
        match = df[df["code"] == symbol]
        if not match.empty:
            return str(match.iloc[0]["name"])
    except Exception:
        pass
    return ""


# ==================== 下载数据（供 dataset service 调用） ====================

def download_stock_data(
    symbol: str,
    period: str = "daily",
    start_date: str = "",
    end_date: str = "",
    adjust: str = "qfq",
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    从 akshare 下载股票数据并做列名映射。
    返回映射后的 DataFrame，由调用方决定如何保存。
    支持双源降级（东财源 → 新浪源）。
    """
    try:
        df = _fetch_stock_hist(
            symbol=symbol, period=period,
            start_date=start_date, end_date=end_date,
            adjust=adjust,
        )
    except RuntimeError as e:
        logger.error(f"下载数据失败（双源均不可用）: symbol={symbol}, error={e}")
        raise

    if df.empty:
        raise ValueError(f"未获取到数据: symbol={symbol}, period={period}, "
                         f"start_date={start_date}, end_date={end_date}")

    # 列名映射
    df = df.rename(columns=COLUMN_MAP)

    # 去掉 Symbol 列（项目不需要）
    if "Symbol" in df.columns:
        df = df.drop(columns=["Symbol"])

    # 按用户选择的列过滤
    if columns:
        keep = ["Date"] + [c for c in columns if c in df.columns and c != "Date"]
        df = df[[c for c in keep if c in df.columns]]

    return df
