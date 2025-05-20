import ccxt
import pandas as pd
import time
from ta.trend import EMAIndicator

exchange = ccxt.binance({
    'apiKey': 'YOUR_API_KEY',
    'secret': 'YOUR_SECRET',
    'enableRateLimit': True,
})

symbol = 'BTC/USDT'
timeframe = '1h'
amount = 0.001
trailing_stop_pct = 0.005  
def fetch_ohlcv():
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def calculate_ema(df):
    df['ema10'] = EMAIndicator(df['close'], window=10).ema_indicator()
    df['ema50'] = EMAIndicator(df['close'], window=50).ema_indicator()
    return df

def get_position():
    try:
        positions = exchange.fapiPrivate_get_positionrisk()  
        for pos in positions:
            if pos['symbol'] == symbol.replace('/', '') and float(pos['positionAmt']) != 0:
                return pos
    except Exception:
        return None
    return None

def place_order(side, amount):
    if side == 'buy':
        print("Placing market buy order")
        exchange.create_market_buy_order(symbol, amount)
    else:
        print("Placing market sell order")
        exchange.create_market_sell_order(symbol, amount)

def trailing_stop(position, current_price):
    side = 'buy' if float(position['positionAmt']) > 0 else 'sell'
    entry_price = float(position['entryPrice'])
    stop_loss_price = None

    if side == 'buy':
        new_stop = current_price * (1 - trailing_stop_pct)
        old_stop = float(position.get('stopLoss', 0)) or (entry_price * (1 - trailing_stop_pct))
        stop_loss_price = max(new_stop, old_stop)
        if stop_loss_price < current_price:
            print(f"Updating stop loss to {stop_loss_price:.2f} for LONG position")

    else:
        new_stop = current_price * (1 + trailing_stop_pct)
        old_stop = float(position.get('stopLoss', 0)) or (entry_price * (1 + trailing_stop_pct))
        stop_loss_price = min(new_stop, old_stop)
        if stop_loss_price > current_price:
            print(f"Updating stop loss to {stop_loss_price:.2f} for SHORT position")


def main():
    df = fetch_ohlcv()
    df = calculate_ema(df)

    if len(df) < 2:
        print("Not enough data")
        return

    last = df.iloc[-1]
    prev = df.iloc[-2]

    position = get_position()

    if prev['ema10'] < prev['ema50'] and last['ema10'] > last['ema50']:
        if not position or float(position['positionAmt']) <= 0:
            place_order('buy', amount)

    elif prev['ema10'] > prev['ema50'] and last['ema10'] < last['ema50']:
        if not position or float(position['positionAmt']) >= 0:
            place_order('sell', amount)

    if position:
        current_price = df['close'].iloc[-1]
        trailing_stop(position, current_price)

if __name__ == "__main__":
    while True:
        try:
            main()
        except Exception as e:
            print("Error:", e)
        time.sleep(60 * 60) 
