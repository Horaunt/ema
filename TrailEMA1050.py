import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import ta
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

class EMACrossoverStrategy:
    def __init__(self, symbol, timeframe, lot, atr_period=14, atr_mult=3.0):
        self.symbol = symbol
        self.timeframe = timeframe
        self.lot = lot
        self.atr_period = atr_period
        self.atr_mult = atr_mult
        self.magic = 234000
        if not mt5.initialize():
            logging.error("MT5 initialize() failed")
            return
        if not mt5.symbol_select(self.symbol, True):
            logging.error(f"Symbol {self.symbol} not available")
            return

    def get_historical_data(self, count):
        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, count)
        df = pd.DataFrame(rates)
        if df.empty:
            return df
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        return df

    def compute_indicators(self, df):
        df['ema10'] = df['close'].ewm(span=10, adjust=False).mean()
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=self.atr_period)
        return df

    def generate_signals(self, df):
        df['signal'] = 0
        df.loc[(df['ema10'] > df['ema50']) & (df['ema10'].shift(1) <= df['ema50'].shift(1)), 'signal'] = 1
        df.loc[(df['ema10'] < df['ema50']) & (df['ema10'].shift(1) >= df['ema50'].shift(1)), 'signal'] = -1
        return df

    def place_order(self, order_type, price, sl, tp, volume):
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            logging.error(f"{self.symbol} not found")
            return
        deviation = 20
        if order_type == mt5.ORDER_TYPE_BUY:
            price = mt5.symbol_info_tick(self.symbol).ask
        else:
            price = mt5.symbol_info_tick(self.symbol).bid
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": deviation,
            "magic": self.magic,
            "comment": "EMA Crossover",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result and result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.error(f"Order send failed, retcode={result.retcode}")
        else:
            logging.info(f"Order placed: {order_type} {volume} {self.symbol} at {price}")
        return result

    def update_trailing_stops(self):
        positions = mt5.positions_get(symbol=self.symbol)
        if positions is None:
            return
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            return
        data = self.get_historical_data(50)
        if data.empty:
            return
        data = self.compute_indicators(data)
        atr_val = data['atr'].iloc[-1]
        for pos in positions:
            if pos.magic != self.magic:
                continue
            if pos.type == mt5.ORDER_TYPE_BUY:
                new_sl = tick.bid - atr_val * self.atr_mult
                if new_sl > pos.sl:
                    req = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "symbol": self.symbol,
                        "position": pos.ticket,
                        "volume": float(pos.volume),
                        "type": mt5.ORDER_TYPE_SELL,
                        "sl": new_sl,
                        "price": tick.bid,
                        "magic": self.magic,
                        "comment": "Trailing Stop",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_IOC,
                    }
                    mt5.order_send(req)
            elif pos.type == mt5.ORDER_TYPE_SELL:
                new_sl = tick.ask + atr_val * self.atr_mult
                if new_sl < pos.sl or pos.sl == 0.0:
                    req = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "symbol": self.symbol,
                        "position": pos.ticket,
                        "volume": float(pos.volume),
                        "type": mt5.ORDER_TYPE_BUY,
                        "sl": new_sl,
                        "price": tick.ask,
                        "magic": self.magic,
                        "comment": "Trailing Stop",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_IOC,
                    }
                    mt5.order_send(req)

    def backtest_strategy(self):
        df = self.get_historical_data(1000)
        if df.empty:
            return
        df = self.compute_indicators(df)
        df = self.generate_signals(df)
        balance = 0.0
        in_position = False
        position_type = None
        entry_price = 0.0
        stop_price = 0.0
        for i in range(1, len(df)):
            row = df.iloc[i]
            price = row['close']
            atr_val = row['atr']
            if not in_position:
                if row['signal'] == 1 and not np.isnan(atr_val):
                    in_position = True
                    position_type = 'buy'
                    entry_price = price
                    stop_price = price - atr_val * self.atr_mult
                elif row['signal'] == -1 and not np.isnan(atr_val):
                    in_position = True
                    position_type = 'sell'
                    entry_price = price
                    stop_price = price + atr_val * self.atr_mult
            else:
                if position_type == 'buy':
                    if not np.isnan(atr_val):
                        stop_price = max(stop_price, price - atr_val * self.atr_mult)
                    if price <= stop_price:
                        balance += price - entry_price
                        in_position = False
                elif position_type == 'sell':
                    if not np.isnan(atr_val):
                        stop_price = min(stop_price, price + atr_val * self.atr_mult)
                    if price >= stop_price:
                        balance += entry_price - price
                        in_position = False
        logging.info(f"Backtest completed, Net P/L: {balance}")

    def run(self):
        while True:
            df = self.get_historical_data(100)
            if df.empty:
                time.sleep(60)
                continue
            df = self.compute_indicators(df)
            df = self.generate_signals(df)
            last_signal = df['signal'].iloc[-1]
            tick = mt5.symbol_info_tick(self.symbol)
            if last_signal == 1:
                price = tick.ask
                atr_val = df['atr'].iloc[-1]
                sl = price - atr_val * self.atr_mult
                tp = price + atr_val * self.atr_mult
                self.place_order(mt5.ORDER_TYPE_BUY, price, sl, tp, self.lot)
            elif last_signal == -1:
                price = tick.bid
                atr_val = df['atr'].iloc[-1]
                sl = price + atr_val * self.atr_mult
                tp = price - atr_val * self.atr_mult
                self.place_order(mt5.ORDER_TYPE_SELL, price, sl, tp, self.lot)
            self.update_trailing_stops()
            time.sleep(60)

if __name__ == "__main__":
    strategy = EMACrossoverStrategy("EURUSD", mt5.TIMEFRAME_M15, 0.1)
    strategy.backtest_strategy()
