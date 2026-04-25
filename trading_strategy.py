import os
import sys
import json
import shutil
import tempfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_all_data():
    configs = json.load(open('config.json', 'r'))
    filepath = os.path.join('data', configs['data']['filename'])
    dataframe = pd.read_csv(filepath)
    return dataframe


# 计算RSI指标(相对强弱指数)
def compute_rsi(price_series, period=14):
    delta = price_series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)
    return rsi

# 计算ATR指标(平均真实波幅)
def compute_atr(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, min_periods=period).mean()
    return atr


def load_lstm_full_predictions(dataframe):
    print("\n加载LSTM模型并生成全量预测...")
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_models')
    if not os.path.exists(model_dir):
        print("  模型目录不存在")
        return None

    model_files = [f for f in os.listdir(model_dir) if f.endswith('.h5')]
    if not model_files:
        print("  没有找到模型文件")
        return None

    model_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)

    configs = json.load(open('config.json', 'r'))
    seq_len = configs['data']['sequence_length']

    temp_dir = tempfile.mkdtemp(prefix='lstm_model_')

    for mf in model_files:
        model_path = os.path.join(model_dir, mf)
        print(f"  尝试加载: {mf}")

        try:
            temp_model_path = os.path.join(temp_dir, 'model.h5')
            shutil.copy2(model_path, temp_model_path)

            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from core.model import Model

            model = Model()
            model.load_model(temp_model_path)

            input_shape = model.model.input_shape
            n_features = input_shape[2] if len(input_shape) == 3 else 2
            print(f"  模型输入形状: {input_shape}")

            all_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            available_cols = [c for c in all_cols if c in dataframe.columns]
            use_cols = available_cols[:n_features]
            print(f"  使用特征列: {use_cols}")

            data_raw = dataframe[use_cols].values.astype(float)

            normalised_data = []
            for i in range(len(data_raw) - seq_len + 1):
                window = data_raw[i:i + seq_len]
                norm_window = np.zeros_like(window)
                for col in range(window.shape[1]):
                    base = window[0, col]
                    if base != 0:
                        norm_window[:, col] = (window[:, col] / base) - 1
                    else:
                        norm_window[:, col] = 0
                normalised_data.append(norm_window)
            normalised_data = np.array(normalised_data)

            x_all = normalised_data[:, :-1, :]

            print(f"  全量数据: {x_all.shape}, 生成预测中...")

            predictions = model.predict_point_by_point(x_all)
            print(f"  生成 {len(predictions)} 个预测点")

            pred_series = pd.Series(predictions)
            smooth_fast = pred_series.ewm(span=3, adjust=False).mean()
            smooth_slow = pred_series.ewm(span=10, adjust=False).mean()

            lstm_signal = np.zeros(len(dataframe))
            for i in range(len(predictions)):
                idx = i + seq_len - 1
                if idx < len(lstm_signal):
                    if smooth_fast.iloc[i] > smooth_slow.iloc[i]:
                        lstm_signal[idx] = 1
                    else:
                        lstm_signal[idx] = -1

            return {
                'predictions': predictions,
                'seq_len': seq_len,
                'lstm_signal': lstm_signal,
            }

        except Exception as e:
            print(f"  加载 {mf} 失败: {str(e)[:150]}")
            continue

    try:
        shutil.rmtree(temp_dir)
    except:
        pass
    print("  所有模型加载失败")
    return None


def load_lstm_predictions_from_model(dataframe, model_path, seq_len):
    """
    从指定模型路径加载LSTM模型并生成全量预测。

    与 load_lstm_full_predictions 不同，此函数：
    - 接受明确的模型文件路径，而非自动搜索 saved_models 目录
    - 接受明确的 seq_len，而非从 config.json 读取
    - 用于回测时选择特定训练任务的模型
    """
    print(f"\n加载指定LSTM模型并生成全量预测: {model_path}")

    if not os.path.isfile(model_path):
        print(f"  模型文件不存在: {model_path}")
        return None

    temp_dir = tempfile.mkdtemp(prefix='lstm_model_')

    try:
        # 复制到临时目录避免文件锁问题
        temp_model_path = os.path.join(temp_dir, 'model.h5')
        shutil.copy2(model_path, temp_model_path)

        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from core.model import Model

        model = Model()
        model.load_model(temp_model_path)

        input_shape = model.model.input_shape
        n_features = input_shape[2] if len(input_shape) == 3 else 2
        print(f"  模型输入形状: {input_shape}, 序列长度: {seq_len}")

        # 根据模型输入维度选择特征列
        all_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        available_cols = [c for c in all_cols if c in dataframe.columns]
        use_cols = available_cols[:n_features]
        print(f"  使用特征列: {use_cols}")

        data_raw = dataframe[use_cols].values.astype(float)

        # 滑动窗口标准化
        normalised_data = []
        for i in range(len(data_raw) - seq_len + 1):
            window = data_raw[i:i + seq_len]
            norm_window = np.zeros_like(window)
            for col in range(window.shape[1]):
                base = window[0, col]
                if base != 0:
                    norm_window[:, col] = (window[:, col] / base) - 1
                else:
                    norm_window[:, col] = 0
            normalised_data.append(norm_window)
        normalised_data = np.array(normalised_data)

        x_all = normalised_data[:, :-1, :]

        print(f"  全量数据: {x_all.shape}, 生成预测中...")
        predictions = model.predict_point_by_point(x_all)
        print(f"  生成 {len(predictions)} 个预测点")

        # 生成LSTM信号
        pred_series = pd.Series(predictions)
        smooth_fast = pred_series.ewm(span=3, adjust=False).mean()
        smooth_slow = pred_series.ewm(span=10, adjust=False).mean()

        lstm_signal = np.zeros(len(dataframe))
        for i in range(len(predictions)):
            idx = i + seq_len - 1
            if idx < len(lstm_signal):
                if smooth_fast.iloc[i] > smooth_slow.iloc[i]:
                    lstm_signal[idx] = 1
                else:
                    lstm_signal[idx] = -1

        return {
            'predictions': predictions,
            'seq_len': seq_len,
            'lstm_signal': lstm_signal,
        }

    except Exception as e:
        print(f"  加载模型失败: {str(e)[:200]}")
        return None
    finally:
        try:
            shutil.rmtree(temp_dir)
        except:
            pass

# 趋势跟踪策略
def generate_signals_trend_rider(prices, high, low, close, lstm_data=None,
                                   trend_ema=50, confirm_ema=20,
                                   bull_trail=0.15, bear_trail=0.03,
                                   stop_loss_pct=0.12, rsi_period=14):
    n = len(prices)
    signals = pd.DataFrame(index=range(n))
    signals['price'] = prices
    price_series = pd.Series(prices)

    signals['trend_ema'] = price_series.ewm(span=trend_ema, adjust=False).mean()
    signals['confirm_ema'] = price_series.ewm(span=confirm_ema, adjust=False).mean()
    signals['rsi'] = compute_rsi(price_series, rsi_period)

    prev_trend = signals['trend_ema'].diff(5)
    signals['trend_rising'] = prev_trend > 0

    has_lstm = lstm_data is not None
    if has_lstm:
        lstm_signal = lstm_data['lstm_signal']

    signals['signal'] = 0
    in_position = False
    entry_price = 0
    highest_since_entry = 0

    for i in range(max(trend_ema, confirm_ema, rsi_period) + 6, n):
        price = prices[i]
        trend_val = signals['trend_ema'].iloc[i]
        confirm_val = signals['confirm_ema'].iloc[i]
        rsi_val = signals['rsi'].iloc[i]
        trend_rising = signals['trend_rising'].iloc[i]

        above_trend = price > trend_val
        above_confirm = price > confirm_val

        if not in_position:
            buy_cond = above_trend and above_confirm and trend_rising and rsi_val < 72

            if buy_cond:
                signals.iloc[i, signals.columns.get_loc('signal')] = 1
                in_position = True
                entry_price = price
                highest_since_entry = price
        else:
            if price > highest_since_entry:
                highest_since_entry = price

            profit_pct = (price - entry_price) / entry_price

            if above_trend and trend_rising:
                current_trail = bull_trail
            elif above_trend:
                current_trail = (bull_trail + bear_trail) / 2
            else:
                current_trail = bear_trail

            fixed_stop = entry_price * (1 - stop_loss_pct)
            trailing_stop = highest_since_entry * (1 - current_trail)
            stop_price = max(fixed_stop, trailing_stop)

            should_sell = False

            if price < stop_price:
                should_sell = True
            elif not above_trend and not above_confirm:
                should_sell = True
            elif not above_trend and rsi_val > 60 and profit_pct > 0.03:
                should_sell = True

            if has_lstm and not should_sell:
                if lstm_signal[i] < 0 and not above_confirm and profit_pct > 0.05:
                    should_sell = True

            if should_sell:
                signals.iloc[i, signals.columns.get_loc('signal')] = -1
                in_position = False

    signals['position'] = signals['signal'].diff().fillna(0)
    return signals

# EMA交叉策略
def generate_signals_ema_cross_wide(prices, high, low, close, lstm_data=None,
                                      fast_ema=8, slow_ema=21,
                                      bull_trail=0.15, bear_trail=0.03,
                                      stop_loss_pct=0.12, rsi_period=14):
    n = len(prices)
    signals = pd.DataFrame(index=range(n))
    signals['price'] = prices
    price_series = pd.Series(prices)

    signals['fast_ema'] = price_series.ewm(span=fast_ema, adjust=False).mean()
    signals['slow_ema'] = price_series.ewm(span=slow_ema, adjust=False).mean()
    signals['rsi'] = compute_rsi(price_series, rsi_period)

    has_lstm = lstm_data is not None
    if has_lstm:
        lstm_signal = lstm_data['lstm_signal']

    signals['signal'] = 0
    in_position = False
    entry_price = 0
    highest_since_entry = 0

    for i in range(max(fast_ema, slow_ema, rsi_period) + 1, n):
        price = prices[i]
        fast_val = signals['fast_ema'].iloc[i]
        slow_val = signals['slow_ema'].iloc[i]
        prev_fast = signals['fast_ema'].iloc[i - 1]
        prev_slow = signals['slow_ema'].iloc[i - 1]
        rsi_val = signals['rsi'].iloc[i]

        fast_above_slow = fast_val > slow_val
        prev_fast_above_slow = prev_fast > prev_slow
        golden_cross = fast_above_slow and not prev_fast_above_slow
        death_cross = not fast_above_slow and prev_fast_above_slow

        if not in_position:
            buy_cond = fast_above_slow and rsi_val < 72

            if buy_cond:
                signals.iloc[i, signals.columns.get_loc('signal')] = 1
                in_position = True
                entry_price = price
                highest_since_entry = price
        else:
            if price > highest_since_entry:
                highest_since_entry = price

            profit_pct = (price - entry_price) / entry_price

            if fast_above_slow:
                current_trail = bull_trail
            else:
                current_trail = bear_trail

            fixed_stop = entry_price * (1 - stop_loss_pct)
            trailing_stop = highest_since_entry * (1 - current_trail)
            stop_price = max(fixed_stop, trailing_stop)

            should_sell = False

            if price < stop_price:
                should_sell = True
            elif death_cross and profit_pct > 0.02:
                should_sell = True
            elif not fast_above_slow and rsi_val > 60 and profit_pct > 0.03:
                should_sell = True

            if has_lstm and not should_sell:
                if lstm_signal[i] < 0 and not fast_above_slow and profit_pct > 0.05:
                    should_sell = True

            if should_sell:
                signals.iloc[i, signals.columns.get_loc('signal')] = -1
                in_position = False

    signals['position'] = signals['signal'].diff().fillna(0)
    return signals

# MACD策略
def generate_signals_macd_wide(prices, high, low, close, lstm_data=None,
                                 trend_ema=50, bull_trail=0.15, bear_trail=0.03,
                                 stop_loss_pct=0.12, rsi_period=14):
    n = len(prices)
    signals = pd.DataFrame(index=range(n))
    signals['price'] = prices
    price_series = pd.Series(prices)

    signals['trend_ema'] = price_series.ewm(span=trend_ema, adjust=False).mean()
    signals['rsi'] = compute_rsi(price_series, rsi_period)

    ema_fast = price_series.ewm(span=12, adjust=False).mean()
    ema_slow = price_series.ewm(span=26, adjust=False).mean()
    signals['macd_line'] = ema_fast - ema_slow
    signals['macd_signal'] = signals['macd_line'].ewm(span=9, adjust=False).mean()

    has_lstm = lstm_data is not None
    if has_lstm:
        lstm_signal = lstm_data['lstm_signal']

    signals['signal'] = 0
    in_position = False
    entry_price = 0
    highest_since_entry = 0

    for i in range(trend_ema + 26, n):
        price = prices[i]
        trend_val = signals['trend_ema'].iloc[i]
        rsi_val = signals['rsi'].iloc[i]
        macd_line = signals['macd_line'].iloc[i]
        macd_sig = signals['macd_signal'].iloc[i]
        prev_macd_line = signals['macd_line'].iloc[i - 1]
        prev_macd_sig = signals['macd_signal'].iloc[i - 1]

        above_trend = price > trend_val
        macd_above = macd_line > macd_sig
        prev_macd_above = prev_macd_line > prev_macd_sig
        macd_cross_up = macd_above and not prev_macd_above
        macd_cross_down = not macd_above and prev_macd_above

        if not in_position:
            buy_cond = above_trend and macd_above and rsi_val < 70

            if macd_cross_up and above_trend:
                buy_cond = True

            if buy_cond:
                signals.iloc[i, signals.columns.get_loc('signal')] = 1
                in_position = True
                entry_price = price
                highest_since_entry = price
        else:
            if price > highest_since_entry:
                highest_since_entry = price

            profit_pct = (price - entry_price) / entry_price

            if above_trend and macd_above:
                current_trail = bull_trail
            elif above_trend:
                current_trail = (bull_trail + bear_trail) / 2
            else:
                current_trail = bear_trail

            fixed_stop = entry_price * (1 - stop_loss_pct)
            trailing_stop = highest_since_entry * (1 - current_trail)
            stop_price = max(fixed_stop, trailing_stop)

            should_sell = False

            if price < stop_price:
                should_sell = True
            elif macd_cross_down and not above_trend:
                should_sell = True
            elif not above_trend and rsi_val > 60 and profit_pct > 0.03:
                should_sell = True

            if has_lstm and not should_sell:
                if lstm_signal[i] < 0 and not above_trend and profit_pct > 0.05:
                    should_sell = True

            if should_sell:
                signals.iloc[i, signals.columns.get_loc('signal')] = -1
                in_position = False

    signals['position'] = signals['signal'].diff().fillna(0)
    return signals


def backtest(signals, initial_capital=100000.0, commission_rate=0.001):
    prices = signals['price'].values
    n = len(prices)
    capital = initial_capital
    shares = 0
    portfolio_values = np.zeros(n)
    trade_log = []

    for i in range(n):
        if signals['signal'].iloc[i] == 1 and shares == 0:
            shares = capital / prices[i]
            cost = capital * commission_rate
            capital = 0
            capital -= cost
            trade_log.append({'type': 'BUY', 'idx': i, 'price': prices[i]})
        elif signals['signal'].iloc[i] == -1 and shares > 0:
            capital = shares * prices[i]
            cost = capital * commission_rate
            capital -= cost
            shares = 0
            trade_log.append({'type': 'SELL', 'idx': i, 'price': prices[i]})

        portfolio_values[i] = capital + shares * prices[i]

    if shares > 0:
        capital = shares * prices[-1]
        shares = 0
        portfolio_values[-1] = capital

    total_return = (portfolio_values[-1] / initial_capital) - 1
    n_days = len(prices)
    n_years = n_days / 252
    annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

    running_max = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - running_max) / running_max
    max_drawdown = drawdown.min()

    benchmark_values = prices / prices[0] * initial_capital
    benchmark_total_return = (prices[-1] / prices[0]) - 1
    benchmark_annual = (1 + benchmark_total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

    win_trades = 0
    loss_trades = 0
    for j in range(0, len(trade_log) - 1, 2):
        if j + 1 < len(trade_log):
            buy_price = trade_log[j]['price']
            sell_price = trade_log[j + 1]['price']
            if sell_price > buy_price * (1 + 2 * commission_rate):
                win_trades += 1
            else:
                loss_trades += 1
    total_trades_pair = win_trades + loss_trades
    win_rate = win_trades / total_trades_pair if total_trades_pair > 0 else 0

    results = {
        'total_return': total_return,
        'annual_return': annual_return,
        'max_drawdown': max_drawdown,
        'n_trades': len(trade_log),
        'win_rate': win_rate,
        'portfolio_values': portfolio_values,
        'benchmark_values': benchmark_values,
        'benchmark_total_return': benchmark_total_return,
        'benchmark_annual': benchmark_annual,
        'trade_log': trade_log,
        'drawdown': drawdown,
    }
    return results


def plot_trading_results(signals, results, dates_raw, strategy_name="Trading Strategy"):
    prices = signals['price'].values
    portfolio_values = results['portfolio_values']
    benchmark_values = results['benchmark_values']
    trade_log = results['trade_log']
    drawdown = results['drawdown']

    try:
        date_objs = pd.to_datetime(dates_raw, format='%d-%m-%y')
    except:
        date_objs = pd.to_datetime(dates_raw)

    fig, axes = plt.subplots(3, 1, figsize=(18, 15), gridspec_kw={'height_ratios': [3, 2, 1]})
    fig.suptitle(f'{strategy_name} - 回测结果', fontsize=16, fontweight='bold')

    ax1 = axes[0]
    ax1.plot(date_objs, prices, color='black', linewidth=1.0, label='收盘价', alpha=0.8)

    for col, color, label in [
        ('fast_ema', 'blue', '快线EMA'),
        ('slow_ema', 'orange', '慢线EMA'),
        ('trend_ema', 'purple', '趋势EMA'),
        ('confirm_ema', 'cyan', '确认EMA'),
    ]:
        if col in signals.columns:
            ax1.plot(date_objs, signals[col].values, color=color, linewidth=0.8, alpha=0.5, label=label)

    buy_trades = [t for t in trade_log if t['type'] == 'BUY']
    sell_trades = [t for t in trade_log if t['type'] == 'SELL']

    if buy_trades:
        buy_dates = [date_objs[t['idx']] for t in buy_trades]
        buy_prices = [t['price'] for t in buy_trades]
        ax1.scatter(buy_dates, buy_prices, marker='^', color='red', s=120, zorder=5, label='买入')

    if sell_trades:
        sell_dates = [date_objs[t['idx']] for t in sell_trades]
        sell_prices = [t['price'] for t in sell_trades]
        ax1.scatter(sell_dates, sell_prices, marker='v', color='green', s=120, zorder=5, label='卖出')

    ax1.set_ylabel('价格', fontsize=12)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    ax2 = axes[1]
    ax2.plot(date_objs, portfolio_values, color='blue', linewidth=1.2, label='策略净值')
    ax2.plot(date_objs, benchmark_values, color='gray', linewidth=1.0, alpha=0.7, label='基准(买入持有)')
    ax2.set_ylabel('组合价值', fontsize=12)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    ax3 = axes[2]
    ax3.fill_between(date_objs, drawdown * 100, 0, color='red', alpha=0.4)
    ax3.plot(date_objs, drawdown * 100, color='red', linewidth=0.8)
    ax3.set_ylabel('回撤 (%)', fontsize=12)
    ax3.set_xlabel('日期', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

    stats_text = (
        f"年化收益率: {results['annual_return'] * 100:.2f}%\n"
        f"最大回撤: {results['max_drawdown'] * 100:.2f}%\n"
        f"总收益率: {results['total_return'] * 100:.2f}%\n"
        f"交易次数: {results['n_trades']}\n"
        f"胜率: {results['win_rate'] * 100:.1f}%\n"
        f"基准年化: {results['benchmark_annual'] * 100:.2f}%"
    )
    ax1.text(0.98, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig('trading_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n图表已保存至 trading_results.png")


def run_strategy():
    print("=" * 60)
    print("LSTM股票预测 - 交易策略回测 (宽幅追踪版)")
    print("=" * 60)

    dataframe = load_all_data()
    prices = dataframe['Close'].values
    high = dataframe['High'].values
    low = dataframe['Low'].values
    close = prices
    dates = dataframe['Date'].values

    print(f"\n数据量: {len(prices)} 个交易日 ({len(prices)/252:.1f} 年)")

    lstm_data = load_lstm_full_predictions(dataframe)

    strategies = {}

    for trend in [40, 50, 60]:
        for confirm in [15, 20, 25]:
            if confirm >= trend:
                continue
            for bull_tr in [0.12, 0.15, 0.18, 0.20]:
                for bear_tr in [0.02, 0.03, 0.04]:
                    for sl in [0.10, 0.12, 0.15]:
                        name = f'趋势骑手(T{trend}/C{confirm},BT{int(bull_tr*100)}%/KT{int(bear_tr*100)}%/SL{int(sl*100)}%)'
                        strategies[name] = lambda trend=trend, confirm=confirm, bull_tr=bull_tr, bear_tr=bear_tr, sl=sl: generate_signals_trend_rider(
                            prices, high, low, close, lstm_data,
                            trend_ema=trend, confirm_ema=confirm,
                            bull_trail=bull_tr, bear_trail=bear_tr,
                            stop_loss_pct=sl)

    for fast in [5, 8, 10]:
        for slow in [20, 21, 25]:
            if fast >= slow:
                continue
            for bull_tr in [0.12, 0.15, 0.18, 0.20]:
                for bear_tr in [0.02, 0.03, 0.04]:
                    for sl in [0.10, 0.12, 0.15]:
                        name = f'EMA交叉({fast}/{slow},BT{int(bull_tr*100)}%/KT{int(bear_tr*100)}%/SL{int(sl*100)}%)'
                        strategies[name] = lambda fast=fast, slow=slow, bull_tr=bull_tr, bear_tr=bear_tr, sl=sl: generate_signals_ema_cross_wide(
                            prices, high, low, close, lstm_data,
                            fast_ema=fast, slow_ema=slow,
                            bull_trail=bull_tr, bear_trail=bear_tr,
                            stop_loss_pct=sl)

    for trend in [40, 50, 60]:
        for bull_tr in [0.12, 0.15, 0.18, 0.20]:
            for bear_tr in [0.02, 0.03, 0.04]:
                for sl in [0.10, 0.12, 0.15]:
                    name = f'MACD趋势(T{trend},BT{int(bull_tr*100)}%/KT{int(bear_tr*100)}%/SL{int(sl*100)}%)'
                    strategies[name] = lambda trend=trend, bull_tr=bull_tr, bear_tr=bear_tr, sl=sl: generate_signals_macd_wide(
                        prices, high, low, close, lstm_data,
                        trend_ema=trend,
                        bull_trail=bull_tr, bear_trail=bear_tr,
                        stop_loss_pct=sl)

    all_results = {}
    best_name = None
    best_results = None
    best_signals = None
    best_score = -np.inf
    qualified = []

    print(f"\n正在测试 {len(strategies)} 种策略组合...")
    print("=" * 60)

    for idx, (name, strategy_fn) in enumerate(strategies.items()):
        signals = strategy_fn()
        results = backtest(signals)
        all_results[name] = (signals, results)

        meets_target = results['annual_return'] > 0.20 and abs(results['max_drawdown']) < 0.20
        score = results['annual_return'] - abs(results['max_drawdown'])

        if meets_target:
            qualified.append((name, signals, results, score))
            print(f"  [{len(qualified)}.达标] {name}: 年化{results['annual_return']*100:.2f}%, 回撤{results['max_drawdown']*100:.2f}%, 交易{results['n_trades']}次")

        if meets_target and score > best_score:
            best_score = score
            best_name = name
            best_results = results
            best_signals = signals

        if (idx + 1) % 500 == 0:
            print(f"  已测试 {idx + 1}/{len(strategies)}...")

    if best_name is None:
        print("\n没有策略达标，选择综合评分最高的策略")
        best_score = -np.inf
        for name, (signals, results) in all_results.items():
            score = results['annual_return'] - abs(results['max_drawdown'])
            if score > best_score:
                best_score = score
                best_name = name
                best_results = results
                best_signals = signals

    print("\n" + "=" * 60)
    print(f"最优策略: {best_name}")
    print(f"  年化收益率: {best_results['annual_return'] * 100:.2f}%")
    print(f"  最大回撤:   {best_results['max_drawdown'] * 100:.2f}%")
    print(f"  总收益率:   {best_results['total_return'] * 100:.2f}%")
    print(f"  交易次数:   {best_results['n_trades']}")
    print(f"  胜率:       {best_results['win_rate'] * 100:.1f}%")
    print(f"  基准年化:   {best_results['benchmark_annual'] * 100:.2f}%")
    if qualified:
        print(f"\n达标策略数: {len(qualified)}")
    print("=" * 60)

    plot_trading_results(best_signals, best_results, dates, strategy_name=best_name)

    return best_signals, best_results


if __name__ == '__main__':
    run_strategy()
