import pandas as pd
import numpy as np

def calculate_ema(prices, window):
    return prices.ewm(span=window, adjust=False).mean()

def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(high, low, close, window=14):
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    return atr

def advanced_ema_strategy(data, short_ema=12, long_ema=26, rsi_period=14, rsi_threshold=50, atr_period=14, risk_per_trade=0.01):
    df = data.copy()
    df['EMA_short'] = calculate_ema(df['Close'], short_ema)
    df['EMA_long'] = calculate_ema(df['Close'], long_ema)
    df['RSI'] = calculate_rsi(df['Close'], rsi_period)
    df['ATR'] = calculate_atr(df['High'], df['Low'], df['Close'], atr_period)

    df['Position'] = 0
    position = 0
    entry_price = 0
    stop_loss = 0

    trades = []

    for i in range(1, len(df)):
        if position == 0:
            # Entry conditions
            if (df['EMA_short'].iloc[i] > df['EMA_long'].iloc[i]) and \
               (df['EMA_short'].iloc[i-1] <= df['EMA_long'].iloc[i-1]) and \
               (df['RSI'].iloc[i] > rsi_threshold):
                
                position = 1
                entry_price = df['Close'].iloc[i]
                stop_loss = entry_price - 1.5 * df['ATR'].iloc[i]
                trades.append({'Type': 'Buy', 'Date': df.index[i], 'Price': entry_price})
                df.at[df.index[i], 'Position'] = 1

        elif position == 1:
            # Update trailing stop loss
            new_stop = df['Close'].iloc[i] - 1.5 * df['ATR'].iloc[i]
            stop_loss = max(stop_loss, new_stop)

            # Exit conditions
            if (df['EMA_short'].iloc[i] < df['EMA_long'].iloc[i]) or (df['Close'].iloc[i] < stop_loss):
                position = 0
                exit_price = df['Close'].iloc[i]
                trades.append({'Type': 'Sell', 'Date': df.index[i], 'Price': exit_price})
                df.at[df.index[i], 'Position'] = -1

            else:
                df.at[df.index[i], 'Position'] = 1

    # Backtest performance metrics
    returns = []
    for j in range(1, len(trades), 2):
        buy = trades[j-1]['Price']
        sell = trades[j]['Price']
        returns.append((sell - buy) / buy)

    total_return = np.prod([1 + r for r in returns]) - 1 if returns else 0
    win_rate = np.mean([r > 0 for r in returns]) if returns else 0
    max_drawdown = 0
    peak = 1
    equity_curve = np.cumprod([1 + r for r in returns]) if returns else np.array([])

    for x in equity_curve:
        if x > peak:
            peak = x
        dd = (peak - x) / peak
        if dd > max_drawdown:
            max_drawdown = dd

    performance = {
        'Total Return': total_return,
        'Win Rate': win_rate,
        'Max Drawdown': max_drawdown,
        'Number of Trades': len(returns)
    }

    return df, trades, performance
