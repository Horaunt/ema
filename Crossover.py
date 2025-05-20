import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import yfinance as yf
import pandas_ta as ta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class EMACrossoverStrategy:
    """
    Advanced EMA Crossover Trading Strategy with intelligent position sizing, adaptive trailing stops,
    and comprehensive performance metrics.
    
    Strategy rules:
    1. Buy when EMA(10) crosses above EMA(50)
    2. Sell when EMA(10) crosses below EMA(50)
    3. Implement trailing stop-loss for both long and short positions
    4. Use dynamic position sizing based on volatility
    5. Apply market regime filters to avoid trading in unfavorable conditions
    """
    
    def __init__(self, 
                 symbol, 
                 timeframe='1d',
                 fast_ema=10, 
                 slow_ema=50, 
                 trailing_stop_atr_multiplier=2.0,
                 initial_capital=100000,
                 position_size_pct=0.02,
                 max_position_size_pct=0.05,
                 risk_free_rate=0.03,
                 atr_period=14,
                 regime_period=200,
                 volatility_lookback=20,
                 start_date=None,
                 end_date=None):
        """
        Initialize the EMA Crossover Strategy with customizable parameters.
        
        Parameters:
        -----------
        symbol : str
            Ticker symbol to trade
        timeframe : str
            Time interval for data ('1d', '1h', etc.)
        fast_ema : int
            Fast EMA period
        slow_ema : int
            Slow EMA period
        trailing_stop_atr_multiplier : float
            ATR multiplier for trailing stop calculation
        initial_capital : float
            Starting capital for backtesting
        position_size_pct : float
            Base position size as percentage of capital
        max_position_size_pct : float
            Maximum position size as percentage of capital
        risk_free_rate : float
            Annual risk-free rate for performance calculations
        atr_period : int
            Period for ATR calculation
        regime_period : int
            Period for market regime identification
        volatility_lookback : int
            Period for volatility calculation
        start_date : str, optional
            Start date for data (YYYY-MM-DD)
        end_date : str, optional
            End date for data (YYYY-MM-DD)
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.trailing_stop_atr_multiplier = trailing_stop_atr_multiplier
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.max_position_size_pct = max_position_size_pct
        self.risk_free_rate = risk_free_rate
        self.atr_period = atr_period
        self.regime_period = regime_period
        self.volatility_lookback = volatility_lookback
        
        # Set date range
        self.start_date = start_date if start_date else (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
        self.end_date = end_date if end_date else datetime.now().strftime('%Y-%m-%d')
        
        # Initialize data structures
        self.data = None
        self.trades = []
        self.equity_curve = []
        self.position = 0
        self.cash = initial_capital
        self.equity = initial_capital
        self.entry_price = 0
        self.trailing_stop = 0
        self.trade_count = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
    def fetch_data(self):
        """
        Fetch historical price data and perform initial preprocessing.
        """
        print(f"Fetching data for {self.symbol} from {self.start_date} to {self.end_date}...")
        
        # Get data from yfinance
        self.data = yf.download(self.symbol, start=self.start_date, end=self.end_date, interval=self.timeframe)
        
        if self.data.empty:
            raise ValueError(f"No data found for {self.symbol} with the specified parameters.")
            
        print(f"Data loaded: {len(self.data)} periods")
        return self.data
    
    def prepare_indicators(self):
        """
        Calculate technical indicators used in the strategy.
        """
        if self.data is None:
            self.fetch_data()
            
        # Calculate EMAs
        self.data[f'EMA_{self.fast_ema}'] = self.data['Close'].ewm(span=self.fast_ema, adjust=False).mean()
        self.data[f'EMA_{self.slow_ema}'] = self.data['Close'].ewm(span=self.slow_ema, adjust=False).mean()
        
        # Calculate ATR for trailing stops
        self.data['ATR'] = ta.atr(self.data['High'], self.data['Low'], self.data['Close'], length=self.atr_period)
        
        # Calculate crossover signals
        self.data['Signal'] = 0
        self.data.loc[self.data[f'EMA_{self.fast_ema}'] > self.data[f'EMA_{self.slow_ema}'], 'Signal'] = 1
        self.data.loc[self.data[f'EMA_{self.fast_ema}'] < self.data[f'EMA_{self.slow_ema}'], 'Signal'] = -1
        
        # Calculate signal changes (crossovers)
        self.data['Signal_Change'] = self.data['Signal'].diff()
        
        # Calculate volatility for position sizing
        self.data['Daily_Return'] = self.data['Close'].pct_change()
        self.data['Volatility'] = self.data['Daily_Return'].rolling(window=self.volatility_lookback).std()
        
        # Market regime filter
        self.data['SMA_Regime'] = self.data['Close'].rolling(window=self.regime_period).mean()
        self.data['Regime'] = 0
        self.data.loc[self.data['Close'] > self.data['SMA_Regime'], 'Regime'] = 1
        self.data.loc[self.data['Close'] < self.data['SMA_Regime'], 'Regime'] = -1
        
        # Clean up NaN values
        self.data = self.data.dropna()
        
        return self.data
    
    def calculate_position_size(self, index):
        """
        Calculate position size based on account equity and market volatility.
        
        Parameters:
        -----------
        index : int
            Current index in the data
            
        Returns:
        --------
        float
            Position size in units
        """
        # Base position size based on equity
        base_size = self.equity * self.position_size_pct
        
        # Adjust for volatility (less volatile = larger position)
        volatility = self.data['Volatility'].iloc[index]
        if not np.isnan(volatility) and volatility > 0:
            vol_adjustment = 0.02 / max(volatility, 0.005)  # Normalize by target volatility of 2%
            vol_adjustment = min(max(vol_adjustment, 0.5), 2.0)  # Cap adjustments between 0.5x and 2x
        else:
            vol_adjustment = 1.0
            
        # Calculate final position size with constraints
        position_value = base_size * vol_adjustment
        position_value = min(position_value, self.equity * self.max_position_size_pct)
        
        # Convert to units based on current price
        price = self.data['Close'].iloc[index]
        position_size = position_value / price
        
        return position_size
    
    def update_trailing_stop(self, index, position_type):
        """
        Update trailing stop price based on ATR and recent price action.
        
        Parameters:
        -----------
        index : int
            Current index in the data
        position_type : int
            1 for long positions, -1 for short positions
            
        Returns:
        --------
        float
            Updated trailing stop price
        """
        current_price = self.data['Close'].iloc[index]
        atr_value = self.data['ATR'].iloc[index]
        
        # For long positions
        if position_type == 1:
            stop_distance = atr_value * self.trailing_stop_atr_multiplier
            new_stop = current_price - stop_distance
            
            # Only move the stop if it would move up (tighten)
            if new_stop > self.trailing_stop:
                return new_stop
            return self.trailing_stop
            
        # For short positions
        elif position_type == -1:
            stop_distance = atr_value * self.trailing_stop_atr_multiplier
            new_stop = current_price + stop_distance
            
            # Only move the stop if it would move down (tighten)
            if self.trailing_stop == 0 or new_stop < self.trailing_stop:
                return new_stop
            return self.trailing_stop
            
        return 0
    
    def is_stop_triggered(self, index, position_type):
        """
        Check if trailing stop is triggered based on current price.
        
        Parameters:
        -----------
        index : int
            Current index in the data
        position_type : int
            1 for long positions, -1 for short positions
            
        Returns:
        --------
        bool
            True if stop is triggered, False otherwise
        """
        if self.trailing_stop == 0:
            return False
            
        high = self.data['High'].iloc[index]
        low = self.data['Low'].iloc[index]
        
        # For long positions, check if price dropped below stop
        if position_type == 1:
            return low <= self.trailing_stop
            
        # For short positions, check if price rose above stop
        elif position_type == -1:
            return high >= self.trailing_stop
            
        return False
    
    def backtest(self):
        """
        Run the backtest using the prepared data and strategy rules.
        """
        if self.data is None or 'Signal' not in self.data.columns:
            self.prepare_indicators()
            
        print("Running backtest...")
        
        # Initialize tracking variables
        self.equity_curve = []
        self.position = 0
        self.cash = self.initial_capital
        self.equity = self.initial_capital
        self.trades = []
        self.trade_count = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        # Set up results tracking
        results = pd.DataFrame(index=self.data.index)
        results['Position'] = 0
        results['Cash'] = self.cash
        results['Equity'] = self.equity
        results['Returns'] = 0.0
        
        # Loop through data
        for i in range(1, len(self.data)):
            current_date = self.data.index[i]
            
            # Get current bar data
            open_price = self.data['Open'].iloc[i]
            high_price = self.data['High'].iloc[i]
            low_price = self.data['Low'].iloc[i]
            close_price = self.data['Close'].iloc[i]
            
            # Previous state
            prev_equity = self.equity
            prev_position = self.position
            signal = self.data['Signal'].iloc[i]
            signal_change = self.data['Signal_Change'].iloc[i]
            regime = self.data['Regime'].iloc[i]
            
            # Check for stop loss first
            if self.position != 0 and self.is_stop_triggered(i, self.position):
                exit_price = self.trailing_stop  # Use stop price
                trade_pnl = self.position * (exit_price - self.entry_price)
                
                # Record trade
                trade = {
                    'entry_date': self.entry_date,
                    'exit_date': current_date,
                    'entry_price': self.entry_price,
                    'exit_price': exit_price,
                    'position': self.position,
                    'pnl': trade_pnl,
                    'exit_reason': 'stop_loss'
                }
                self.trades.append(trade)
                
                # Update stats
                self.trade_count += 1
                if trade_pnl > 0:
                    self.winning_trades += 1
                else:
                    self.losing_trades += 1
                
                # Close position
                self.cash += self.position * exit_price
                self.position = 0
                self.trailing_stop = 0
            
            # Check for signal changes (crossovers)
            if signal_change != 0 and signal_change is not np.nan:
                # Buy signal (EMA 10 crosses above EMA 50) - only if regime is favorable
                if signal_change > 0 and regime >= 0:
                    # Close any existing short position
                    if self.position < 0:
                        trade_pnl = self.position * (open_price - self.entry_price)
                        
                        # Record trade
                        trade = {
                            'entry_date': self.entry_date,
                            'exit_date': current_date,
                            'entry_price': self.entry_price,
                            'exit_price': open_price,
                            'position': self.position,
                            'pnl': trade_pnl,
                            'exit_reason': 'crossover'
                        }
                        self.trades.append(trade)
                        
                        # Update stats
                        self.trade_count += 1
                        if trade_pnl > 0:
                            self.winning_trades += 1
                        else:
                            self.losing_trades += 1
                        
                        # Close position
                        self.cash += self.position * open_price
                        self.position = 0
                    
                    # Open new long position with position sizing
                    position_size = self.calculate_position_size(i)
                    affordable_size = self.cash / open_price
                    position_size = min(position_size, affordable_size)
                    
                    if position_size > 0:
                        self.position = position_size
                        self.cash -= self.position * open_price
                        self.entry_price = open_price
                        self.entry_date = current_date
                        
                        # Set initial trailing stop
                        self.trailing_stop = self.update_trailing_stop(i, 1)
                
                # Sell signal (EMA 10 crosses below EMA 50) or negative regime
                elif signal_change < 0 or (regime < 0 and self.position > 0):
                    # Close any existing long position
                    if self.position > 0:
                        trade_pnl = self.position * (open_price - self.entry_price)
                        
                        # Record trade
                        trade = {
                            'entry_date': self.entry_date,
                            'exit_date': current_date,
                            'entry_price': self.entry_price,
                            'exit_price': open_price,
                            'position': self.position,
                            'pnl': trade_pnl,
                            'exit_reason': 'crossover' if signal_change < 0 else 'regime_change'
                        }
                        self.trades.append(trade)
                        
                        # Update stats
                        self.trade_count += 1
                        if trade_pnl > 0:
                            self.winning_trades += 1
                        else:
                            self.losing_trades += 1
                        
                        # Close position
                        self.cash += self.position * open_price
                        self.position = 0
                        self.trailing_stop = 0
                    
                    # Open new short position with position sizing if signal change (not just regime)
                    if signal_change < 0:
                        position_size = self.calculate_position_size(i)
                        
                        if position_size > 0:
                            self.position = -position_size
                            self.cash += abs(self.position) * open_price
                            self.entry_price = open_price
                            self.entry_date = current_date
                            
                            # Set initial trailing stop
                            self.trailing_stop = self.update_trailing_stop(i, -1)
            
            # Update trailing stop if position exists
            if self.position != 0:
                self.trailing_stop = self.update_trailing_stop(i, 1 if self.position > 0 else -1)
            
            # Calculate current equity
            self.equity = self.cash
            if self.position != 0:
                self.equity += self.position * close_price
            
            # Store daily results
            results.loc[current_date, 'Position'] = self.position
            results.loc[current_date, 'Cash'] = self.cash
            results.loc[current_date, 'Equity'] = self.equity
            results.loc[current_date, 'Returns'] = (self.equity / prev_equity) - 1 if prev_equity > 0 else 0
        
        # Store final backtest results
        self.results = results
        
        # Calculate performance metrics
        self.calculate_performance_metrics()
        
        print("Backtest completed successfully.")
        return self.results
    
    def calculate_performance_metrics(self):
        """
        Calculate key performance metrics for the strategy.
        """
        # Basic metrics
        self.results['Cumulative_Returns'] = (1 + self.results['Returns']).cumprod()
        total_return = self.results['Cumulative_Returns'].iloc[-1] - 1
        
        # Annualized return
        days = (self.results.index[-1] - self.results.index[0]).days
        annual_return = (1 + total_return) ** (365 / days) - 1
        
        # Risk metrics
        daily_returns = self.results['Returns']
        volatility = daily_returns.std() * np.sqrt(252)  # Annualized volatility
        downside_returns = daily_returns[daily_returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252)
        max_drawdown = (self.results['Equity'] / self.results['Equity'].cummax() - 1).min()
        
        # Calculate Sharpe and Sortino ratios
        sharpe_ratio = (annual_return - self.risk_free_rate) / volatility if volatility > 0 else 0
        sortino_ratio = (annual_return - self.risk_free_rate) / downside_volatility if downside_volatility > 0 else 0
        
        # Win rate
        win_rate = self.winning_trades / self.trade_count if self.trade_count > 0 else 0
        
        # Profit factor
        if self.trades:
            winning_trades_sum = sum([t['pnl'] for t in self.trades if t['pnl'] > 0])
            losing_trades_sum = sum([abs(t['pnl']) for t in self.trades if t['pnl'] < 0])
            profit_factor = winning_trades_sum / losing_trades_sum if losing_trades_sum > 0 else float('inf')
        else:
            profit_factor = 0
        
        # Average trade metrics
        if self.trades:
            avg_trade = sum([t['pnl'] for t in self.trades]) / len(self.trades)
            avg_win = sum([t['pnl'] for t in self.trades if t['pnl'] > 0]) / self.winning_trades if self.winning_trades > 0 else 0
            avg_loss = sum([t['pnl'] for t in self.trades if t['pnl'] < 0]) / self.losing_trades if self.losing_trades > 0 else 0
            win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        else:
            avg_trade = 0
            avg_win = 0
            avg_loss = 0
            win_loss_ratio = 0
        
        # Store metrics in a dictionary
        self.metrics = {
            'initial_capital': self.initial_capital,
            'final_equity': self.results['Equity'].iloc[-1],
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_trade': avg_trade,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'win_loss_ratio': win_loss_ratio,
            'trade_count': self.trade_count
        }
        
        return self.metrics
    
    def print_performance_summary(self):
        """
        Print a summary of the strategy's performance metrics.
        """
        if not hasattr(self, 'metrics'):
            print("No performance metrics available. Run backtest first.")
            return
        
        print("\n" + "="*50)
        print(f"PERFORMANCE SUMMARY: {self.symbol}")
        print("="*50)
        print(f"Period: {self.results.index[0].strftime('%Y-%m-%d')} to {self.results.index[-1].strftime('%Y-%m-%d')}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Equity: ${self.metrics['final_equity']:,.2f}")
        print(f"Total Return: {self.metrics['total_return']*100:.2f}%")
        print(f"Annualized Return: {self.metrics['annual_return']*100:.2f}%")
        print(f"Annualized Volatility: {self.metrics['volatility']*100:.2f}%")
        print(f"Sharpe Ratio: {self.metrics['sharpe_ratio']:.2f}")
        print(f"Sortino Ratio: {self.metrics['sortino_ratio']:.2f}")
        print(f"Maximum Drawdown: {self.metrics['max_drawdown']*100:.2f}%")
        print(f"Win Rate: {self.metrics['win_rate']*100:.2f}%")
        print(f"Profit Factor: {self.metrics['profit_factor']:.2f}")
        print(f"Average Trade: ${self.metrics['avg_trade']:,.2f}")
        print(f"Average Win: ${self.metrics['avg_win']:,.2f}")
        print(f"Average Loss: ${self.metrics['avg_loss']:,.2f}")
        print(f"Win/Loss Ratio: {self.metrics['win_loss_ratio']:.2f}")
        print(f"Total Trades: {self.metrics['trade_count']}")
        print("="*50)
    
    def plot_results(self, include_trades=True):
        """
        Plot the equity curve and trading signals.
        
        Parameters:
        -----------
        include_trades : bool
            Whether to include trade markers on the chart
        """
        if not hasattr(self, 'results'):
            print("No results available. Run backtest first.")
            return
        
        # Set up figure and axes
        fig, axes = plt.subplots(3, 1, figsize=(14, 16), gridspec_kw={'height_ratios': [2, 1, 1]})
        
        # Price and EMAs
        axes[0].set_title(f"{self.symbol} Price Chart with EMA Crossover Signals", fontsize=14)
        axes[0].plot(self.data.index, self.data['Close'], label='Close Price', color='black', linewidth=1.5)
        axes[0].plot(self.data.index, self.data[f'EMA_{self.fast_ema}'], label=f'EMA {self.fast_ema}', color='blue', linewidth=1.2)
        axes[0].plot(self.data.index, self.data[f'EMA_{self.slow_ema}'], label=f'EMA {self.slow_ema}', color='red', linewidth=1.2)
        
        # Mark regime changes
        regime_changes = self.data['Regime'].diff().fillna(0)
        regime_up = self.data[regime_changes > 0].index
        regime_down = self.data[regime_changes < 0].index
        axes[0].scatter(regime_up, self.data.loc[regime_up, 'Close'], marker='^', color='green', alpha=0.5, s=100, label='Bullish Regime')
        axes[0].scatter(regime_down, self.data.loc[regime_down, 'Close'], marker='v', color='red', alpha=0.5, s=100, label='Bearish Regime')
        
        # Mark trades
        if include_trades and self.trades:
            for trade in self.trades:
                if trade['position'] > 0:  # Long trades
                    entry_color = 'green'
                    exit_color = 'red' if trade['pnl'] < 0 else 'blue'
                else:  # Short trades
                    entry_color = 'red'
                    exit_color = 'green' if trade['pnl'] < 0 else 'blue'
                
                # Entry markers
                axes[0].scatter(trade['entry_date'], self.data.loc[trade['entry_date'], 'Close'], 
                             marker='^' if trade['position'] > 0 else 'v', 
                             color=entry_color, s=120)
                
                # Exit markers
                axes[0].scatter(trade['exit_date'], trade['exit_price'], 
                             marker='x', color=exit_color, s=120)
        
        axes[0].set_ylabel('Price', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc='upper left')
        
        # Format x-axis dates
        date_format = mdates.DateFormatter('%Y-%m-%d')
        axes[0].xaxis.set_major_formatter(date_format)
        axes[0].tick_params(axis='x', rotation=45)
        
        # Equity Curve
        axes[1].set_title('Equity Curve', fontsize=14)
        axes[1].plot(self.results.index, self.results['Equity'], label='Strategy Equity', color='blue', linewidth=1.5)
        
        # Benchmark comparison
        benchmark_returns = self.data['Close'].pct_change().fillna(0)
        benchmark_equity = (1 + benchmark_returns).cumprod() * self.initial_capital
        axes[1].plot(self.data.index, benchmark_equity, label=f'{self.symbol} Buy & Hold', color='gray', linewidth=1, alpha=0.7)
        
        axes[1].set_ylabel('Equity ($)', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc='upper left')
        
        # Format x-axis dates
        axes[1].xaxis.set_major_formatter(date_format)
        axes[1].tick_params(axis='x', rotation=45)
        
        # Drawdown chart
        self.results['Drawdown'] = 1 - self.results['Equity'] / self.results['Equity'].cummax()
        axes[2].set_title('Drawdown', fontsize=14)
        axes[2].fill_between(self.results.index, 0, self.results['Drawdown'] * 100, color='red', alpha=0.3)
        axes[2].set_ylabel('Drawdown (%)', fontsize=12)
        axes[2].set_ylim(bottom=0)
        axes[2].grid(True, alpha=0.3)
        
        # Format x-axis dates
        axes[2].xaxis.set_major_formatter(date_format)
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
    def monte_carlo_analysis(self, iterations=1000):
        """
        Perform Monte Carlo simulation to assess strategy robustness.
        
        Parameters:
        -----------
        iterations : int
            Number of Monte Carlo iterations
            
        Returns:
        --------
        dict
            Dictionary with Monte Carlo simulation results
        """
        if not self.trades:
            print("No trades available. Run backtest first.")
            return
        
        # Extract trade returns
        trade_returns = [t['pnl'] / self.initial_capital for t in self.trades]
        
        print(f"Running Monte Carlo simulation with {iterations} iterations...")
        
        # Initialize results containers
        final_equities = []
        max_drawdowns = []
        sharpe_ratios = []
        annual_returns = []
        
        for i in range(iterations):
            # Shuffle trade returns
            np.random.shuffle(trade_returns)
            
            # Simulate equity curve
            equity = [self.initial_capital]
            for ret in trade_returns:
                equity.append(equity[-1] * (1 + ret))
            
            equity_series = pd.Series(equity)
            
            # Calculate metrics
            final_equity = equity[-1]
            returns = equity_series.pct_change().dropna()
            
            # Max drawdown
            drawdown = 1 - equity_series / equity_series.cummax()
            max_dd = drawdown.max()
            
            # Annual return (assuming trades occur over same timeframe)
            days = (self.results.index[-1] - self.results.index[0]).days
            annual_ret = (final_equity / self.initial_capital) ** (365 / days) - 1
            
            # Sharpe (simplified)
            sharpe = (annual_ret - self.risk_free_rate) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
            
            # Store results
            final_equities.append(final_equity)
            max_drawdowns.append(max_dd)
            sharpe_ratios.append(sharpe)
            annual_returns.append(annual_ret)
        
        # Calculate statistics
        final_equity_mean = np.mean(final_equities)
        final_equity_std = np.std(final_equities)
        final_equity_percentiles = {
            '5%': np.percentile(final_equities, 5),
            '25%': np.percentile(final_equities, 25),
            '50%': np.percentile(final_equities, 50),
            '75%': np.percentile(final_equities, 75),
            '95%': np.percentile(final_equities, 95)
        }
        
        drawdown_mean = np.mean(max_drawdowns)
        drawdown_percentiles = {
            '5%': np.percentile(max_drawdowns, 5),
            '50%': np.percentile(max_drawdowns, 50),
            '95%': np.percentile(max_drawdowns, 95)
        }
        
        sharpe_mean = np.mean(sharpe_ratios)
        sharpe_percentiles = {
            '5%': np.percentile(sharpe_ratios, 5),
            '50%': np.percentile(sharpe_ratios, 50),
            '95%': np.percentile(sharpe_ratios, 95)
        }
        
        annual_return_mean = np.mean(annual_returns)
        annual_return_percentiles = {
            '5%': np.percentile(annual_returns, 5),
            '50%': np.percentile(annual_returns, 50),
            '95%': np.percentile(annual_returns, 95)
        }
        
        # Plot Monte Carlo results
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Final equity distribution
        axes[0, 0].hist(final_equities, bins=50, alpha=0.7, color='blue')
        axes[0, 0].axvline(final_equity_mean, color='red', linestyle='--', label=f'Mean: ${final_equity_mean:,.2f}')
        axes[0, 0].axvline(final_equity_percentiles['5%'], color='green', linestyle='--', label=f'5% Worst: ${final_equity_percentiles["5%"]:,.2f}')
        axes[0, 0].set_title('Final Equity Distribution', fontsize=12)
        axes[0, 0].set_xlabel('Final Equity ($)', fontsize=10)
        axes[0, 0].set_ylabel('Frequency', fontsize=10)
        axes[0, 0].legend()
        
        # Max drawdown distribution
        axes[0, 1].hist(max_drawdowns, bins=50, alpha=0.7, color='red')
        axes[0, 1].axvline(drawdown_mean, color='black', linestyle='--', label=f'Mean: {drawdown_mean*100:.2f}%')
        axes[0, 1].axvline(drawdown_percentiles['95%'], color='purple', linestyle='--', label=f'5% Worst: {drawdown_percentiles["95%"]*100:.2f}%')
        axes[0, 1].set_title('Maximum Drawdown Distribution', fontsize=12)
        axes[0, 1].set_xlabel('Maximum Drawdown', fontsize=10)
        axes[0, 1].set_ylabel('Frequency', fontsize=10)
        axes[0, 1].legend()
        
        # Sharpe ratio distribution
        axes[1, 0].hist(sharpe_ratios, bins=50, alpha=0.7, color='green')
        axes[1, 0].axvline(sharpe_mean, color='black', linestyle='--', label=f'Mean: {sharpe_mean:.2f}')
        axes[1, 0].set_title('Sharpe Ratio Distribution', fontsize=12)
        axes[1, 0].set_xlabel('Sharpe Ratio', fontsize=10)
        axes[1, 0].set_ylabel('Frequency', fontsize=10)
        axes[1, 0].legend()
        
        # Annual return distribution
        axes[1, 1].hist(annual_returns, bins=50, alpha=0.7, color='purple')
        axes[1, 1].axvline(annual_return_mean, color='black', linestyle='--', label=f'Mean: {annual_return_mean*100:.2f}%')
        axes[1, 1].axvline(annual_return_percentiles['5%'], color='red', linestyle='--', label=f'5% Worst: {annual_return_percentiles["5%"]*100:.2f}%')
        axes[1, 1].set_title('Annual Return Distribution', fontsize=12)
        axes[1, 1].set_xlabel('Annual Return', fontsize=10)
        axes[1, 1].set_ylabel('Frequency', fontsize=10)
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()
        
        # Compile results
        mc_results = {
            'final_equity': {
                'mean': final_equity_mean,
                'std': final_equity_std,
                'percentiles': final_equity_percentiles
            },
            'max_drawdown': {
                'mean': drawdown_mean,
                'percentiles': drawdown_percentiles
            },
            'sharpe_ratio': {
                'mean': sharpe_mean,
                'percentiles': sharpe_percentiles
            },
            'annual_return': {
                'mean': annual_return_mean,
                'percentiles': annual_return_percentiles
            }
        }
        
        # Print summary
        print("\n" + "="*50)
        print("MONTE CARLO SIMULATION RESULTS")
        print("="*50)
        print(f"Iterations: {iterations}")
        print(f"\nFinal Equity:")
        print(f"  Mean: ${final_equity_mean:,.2f}")
        print(f"  Std Dev: ${final_equity_std:,.2f}")
        print(f"  5% Worst Case: ${final_equity_percentiles['5%']:,.2f}")
        print(f"  Median: ${final_equity_percentiles['50%']:,.2f}")
        
        print(f"\nMaximum Drawdown:")
        print(f"  Mean: {drawdown_mean*100:.2f}%")
        print(f"  5% Worst Case: {drawdown_percentiles['95%']*100:.2f}%")
        print(f"  Median: {drawdown_percentiles['50%']*100:.2f}%")
        
        print(f"\nSharpe Ratio:")
        print(f"  Mean: {sharpe_mean:.2f}")
        print(f"  5% Worst Case: {sharpe_percentiles['5%']:.2f}")
        print(f"  Median: {sharpe_percentiles['50%']:.2f}")
        
        print(f"\nAnnual Return:")
        print(f"  Mean: {annual_return_mean*100:.2f}%")
        print(f"  5% Worst Case: {annual_return_percentiles['5%']*100:.2f}%")
        print(f"  Median: {annual_return_percentiles['50%']*100:.2f}%")
        print("="*50)
        
        return mc_results
    
    def walk_forward_analysis(self, window_size=252, step_size=63, parameter_ranges=None):
        """
        Perform walk-forward optimization to test strategy robustness.
        
        Parameters:
        -----------
        window_size : int
            Size of in-sample window in trading days
        step_size : int
            Number of days to move forward for each iteration
        parameter_ranges : dict, optional
            Dictionary of parameter ranges to test
            
        Returns:
        --------
        dict
            Walk-forward analysis results
        """
        if self.data is None:
            self.fetch_data()
            
        print("Running walk-forward optimization...")
        
        # Default parameter ranges if none provided
        if parameter_ranges is None:
            parameter_ranges = {
                'fast_ema': [5, 8, 10, 12, 15],
                'slow_ema': [30, 40, 50, 60, 70],
                'trailing_stop_atr_multiplier': [1.5, 2.0, 2.5, 3.0]
            }
        
        # Store results
        wf_results = []
        out_of_sample_equity_curves = []
        
        # Ensure data has enough history
        if len(self.data) < window_size + step_size:
            print("Error: Not enough data for walk-forward analysis.")
            return None
        
        # Loop through time windows
        for start_idx in range(0, len(self.data) - window_size - step_size, step_size):
            end_idx = start_idx + window_size
            test_end_idx = end_idx + step_size
            
            # Split data into in-sample and out-of-sample
            train_data = self.data.iloc[start_idx:end_idx].copy()
            test_data = self.data.iloc[end_idx:test_end_idx].copy()
            
            if len(train_data) < 100 or len(test_data) < 20:
                continue
                
            print(f"\nWalk-Forward Window: {train_data.index[0].strftime('%Y-%m-%d')} to {train_data.index[-1].strftime('%Y-%m-%d')}")
            print(f"Test Window: {test_data.index[0].strftime('%Y-%m-%d')} to {test_data.index[-1].strftime('%Y-%m-%d')}")
            
            # Find optimal parameters on in-sample data
            best_params = {}
            best_sharpe = -float('inf')
            
            # Grid search for optimal parameters
            param_combinations = []
            for fast_ema in parameter_ranges['fast_ema']:
                for slow_ema in parameter_ranges['slow_ema']:
                    if fast_ema >= slow_ema:  # Skip invalid combinations
                        continue
                    for atr_mult in parameter_ranges['trailing_stop_atr_multiplier']:
                        param_combinations.append({
                            'fast_ema': fast_ema,
                            'slow_ema': slow_ema,
                            'trailing_stop_atr_multiplier': atr_mult
                        })
            
            print(f"Testing {len(param_combinations)} parameter combinations...")
            
            # Test each parameter combination
            for params in param_combinations:
                # Create a temporary strategy with these parameters
                temp_strategy = EMACrossoverStrategy(
                    symbol=self.symbol,
                    fast_ema=params['fast_ema'],
                    slow_ema=params['slow_ema'],
                    trailing_stop_atr_multiplier=params['trailing_stop_atr_multiplier'],
                    initial_capital=self.initial_capital
                )
                
                # Run backtest on in-sample data
                temp_strategy.data = train_data.copy()
                temp_strategy.prepare_indicators()
                temp_strategy.backtest()
                
                # Evaluate performance
                sharpe = temp_strategy.metrics['sharpe_ratio']
                
                # Update best params if better
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = params.copy()
            
            print(f"Best In-Sample Parameters: {best_params}")
            print(f"Best In-Sample Sharpe Ratio: {best_sharpe:.2f}")
            
            # Test best parameters on out-of-sample data
            oos_strategy = EMACrossoverStrategy(
                symbol=self.symbol,
                fast_ema=best_params['fast_ema'],
                slow_ema=best_params['slow_ema'],
                trailing_stop_atr_multiplier=best_params['trailing_stop_atr_multiplier'],
                initial_capital=self.initial_capital
            )
            
            oos_strategy.data = test_data.copy()
            oos_strategy.prepare_indicators()
            oos_strategy.backtest()
            
            oos_sharpe = oos_strategy.metrics['sharpe_ratio']
            oos_return = oos_strategy.metrics['total_return']
            
            print(f"Out-of-Sample Sharpe Ratio: {oos_sharpe:.2f}")
            print(f"Out-of-Sample Return: {oos_return*100:.2f}%")
            
            # Store results
            window_result = {
                'window_start': train_data.index[0],
                'window_end': train_data.index[-1],
                'test_start': test_data.index[0],
                'test_end': test_data.index[-1],
                'best_params': best_params,
                'in_sample_sharpe': best_sharpe,
                'out_of_sample_sharpe': oos_sharpe,
                'out_of_sample_return': oos_return
            }
            
            wf_results.append(window_result)
            out_of_sample_equity_curves.append(oos_strategy.results['Equity'])
        
        if not wf_results:
            print("Error: No valid results from walk-forward analysis.")
            return None
        
        # Combine results
        wf_stats = {
            'windows': wf_results,
            'avg_is_sharpe': np.mean([r['in_sample_sharpe'] for r in wf_results]),
            'avg_oos_sharpe': np.mean([r['out_of_sample_sharpe'] for r in wf_results]),
            'avg_oos_return': np.mean([r['out_of_sample_return'] for r in wf_results]),
            'robustness_ratio': np.mean([r['out_of_sample_sharpe'] / r['in_sample_sharpe'] if r['in_sample_sharpe'] > 0 else 0 for r in wf_results])
        }
        
        # Plot walk-forward results
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Parameters over time
        window_dates = [r['window_start'] for r in wf_results]
        fast_emas = [r['best_params']['fast_ema'] for r in wf_results]
        slow_emas = [r['best_params']['slow_ema'] for r in wf_results]
        atr_mults = [r['best_params']['trailing_stop_atr_multiplier'] for r in wf_results]
        
        ax1 = axes[0]
        ax1.plot(window_dates, fast_emas, 'b-o', label='Fast EMA')
        ax1.plot(window_dates, slow_emas, 'r-o', label='Slow EMA')
        ax1.set_title('Parameter Stability Over Time', fontsize=14)
        ax1.set_ylabel('EMA Periods', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        
        ax2 = ax1.twinx()
        ax2.plot(window_dates, atr_mults, 'g-o', label='ATR Multiplier')
        ax2.set_ylabel('ATR Multiplier', fontsize=12, color='g')
        ax2.tick_params(axis='y', labelcolor='g')
        ax2.legend(loc='upper right')
        
        # Performance comparison
        is_sharpes = [r['in_sample_sharpe'] for r in wf_results]
        oos_sharpes = [r['out_of_sample_sharpe'] for r in wf_results]
        oos_returns = [r['out_of_sample_return'] * 100 for r in wf_results]
        
        ax3 = axes[1]
        width = 0.35
        x = np.arange(len(window_dates))
        ax3.bar(x - width/2, is_sharpes, width, label='In-Sample Sharpe', color='blue', alpha=0.6)
        ax3.bar(x + width/2, oos_sharpes, width, label='Out-of-Sample Sharpe', color='green', alpha=0.6)
        ax3.set_title('In-Sample vs Out-of-Sample Performance', fontsize=14)
        ax3.set_ylabel('Sharpe Ratio', fontsize=12)
        ax3.set_xticks(x)
        ax3.set_xticklabels([d.strftime('%Y-%m-%d') for d in window_dates], rotation=45)
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper left')
        
        ax4 = ax3.twinx()
        ax4.plot(x, oos_returns, 'r-o', label='OOS Return')
        ax4.set_ylabel('Out-of-Sample Return (%)', fontsize=12, color='r')
        ax4.tick_params(axis='y', labelcolor='r')
        ax4.legend(loc='upper right')
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        print("\n" + "="*50)
        print("WALK-FORWARD ANALYSIS SUMMARY")
        print("="*50)
        print(f"Number of windows: {len(wf_results)}")
        print(f"Average In-Sample Sharpe: {wf_stats['avg_is_sharpe']:.2f}")
        print(f"Average Out-of-Sample Sharpe: {wf_stats['avg_oos_sharpe']:.2f}")
        print(f"Average Out-of-Sample Return: {wf_stats['avg_oos_return']*100:.2f}%")
        print(f"Robustness Ratio (OOS/IS Sharpe): {wf_stats['robustness_ratio']:.2f}")
        print("="*50)
        
        return wf_stats
    
    def optimize_parameters(self, parameter_ranges=None):
        """
        Perform grid search optimization to find optimal parameter values.
        
        Parameters:
        -----------
        parameter_ranges : dict, optional
            Dictionary of parameter ranges to test
            
        Returns:
        --------
        dict
            Optimization results
        """
        if self.data is None:
            self.prepare_indicators()
            
        print("Running parameter optimization...")
        
        # Default parameter ranges if none provided
        if parameter_ranges is None:
            parameter_ranges = {
                'fast_ema': [5, 8, 10, 12, 15, 20],
                'slow_ema': [30, 40, 50, 60, 70, 80],
                'trailing_stop_atr_multiplier': [1.5, 2.0, 2.5, 3.0, 3.5]
            }
        
        # Generate all valid parameter combinations
        param_combinations = []
        for fast_ema in parameter_ranges['fast_ema']:
            for slow_ema in parameter_ranges['slow_ema']:
                if fast_ema >= slow_ema:  # Skip invalid combinations
                    continue
                for atr_mult in parameter_ranges['trailing_stop_atr_multiplier']:
                    param_combinations.append({
                        'fast_ema': fast_ema,
                        'slow_ema': slow_ema,
                        'trailing_stop_atr_multiplier': atr_mult
                    })
        
        total_combinations = len(param_combinations)
        print(f"Testing {total_combinations} parameter combinations...")
        
        # Track performance for each combination
        results = []
        
        # Test each parameter combination
        for i, params in enumerate(param_combinations):
            if (i + 1) % 10 == 0:
                print(f"Progress: {i+1}/{total_combinations} ({(i+1)/total_combinations*100:.1f}%)")
                
            # Create a temporary strategy with these parameters
            temp_strategy = EMACrossoverStrategy(
                symbol=self.symbol,
                fast_ema=params['fast_ema'],
                slow_ema=params['slow_ema'],
                trailing_stop_atr_multiplier=params['trailing_stop_atr_multiplier'],
                initial_capital=self.initial_capital
            )
            
            # Run backtest
            temp_strategy.data = self.data.copy()
            temp_strategy.prepare_indicators()
            temp_strategy.backtest()
            
            # Store results
            params_result = params.copy()
            params_result.update({
                'sharpe_ratio': temp_strategy.metrics['sharpe_ratio'],
                'sortino_ratio': temp_strategy.metrics['sortino_ratio'],
                'annual_return': temp_strategy.metrics['annual_return'],
                'max_drawdown': temp_strategy.metrics['max_drawdown'],
                'win_rate': temp_strategy.metrics['win_rate'],
                'profit_factor': temp_strategy.metrics['profit_factor'],
                'trade_count': temp_strategy.metrics['trade_count']
            })
            
            results.append(params_result)
        
        # Convert to DataFrame for easier analysis
        results_df = pd.DataFrame(results)
        
        # Find optimal parameters based on different metrics
        best_sharpe = results_df.loc[results_df['sharpe_ratio'].idxmax()]
        best_sortino = results_df.loc[results_df['sortino_ratio'].idxmax()]
        best_return = results_df.loc[results_df['annual_return'].idxmax()]
        best_drawdown = results_df.loc[results_df['max_drawdown'].idxmin()]
        
        # Print summary of top performers
        print("\n" + "="*50)
        print("PARAMETER OPTIMIZATION RESULTS")
        print("="*50)
        
        print("\nBest Sharpe Ratio Parameters:")
        print(f"  Fast EMA: {best_sharpe['fast_ema']}")
        print(f"  Slow EMA: {best_sharpe['slow_ema']}")
        print(f"  ATR Multiplier: {best_sharpe['trailing_stop_atr_multiplier']}")
        print(f"  Sharpe: {best_sharpe['sharpe_ratio']:.2f}")
        print(f"  Annual Return: {best_sharpe['annual_return']*100:.2f}%")
        print(f"  Max Drawdown: {best_sharpe['max_drawdown']*100:.2f}%")
        
        print("\nBest Sortino Ratio Parameters:")
        print(f"  Fast EMA: {best_sortino['fast_ema']}")
        print(f"  Slow EMA: {best_sortino['slow_ema']}")
        print(f"  ATR Multiplier: {best_sortino['trailing_stop_atr_multiplier']}")
        print(f"  Sortino: {best_sortino['sortino_ratio']:.2f}")
        print(f"  Annual Return: {best_sortino['annual_return']*100:.2f}%")
        print(f"  Max Drawdown: {best_sortino['max_drawdown']*100:.2f}%")
        
        print("\nBest Annual Return Parameters:")
        print(f"  Fast EMA: {best_return['fast_ema']}")
        print(f"  Slow EMA: {best_return['slow_ema']}")
        print(f"  ATR Multiplier: {best_return['trailing_stop_atr_multiplier']}")
        print(f"  Annual Return: {best_return['annual_return']*100:.2f}%")
        print(f"  Sharpe: {best_return['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {best_return['max_drawdown']*100:.2f}%")
        
        print("\nBest Max Drawdown Parameters:")
        print(f"  Fast EMA: {best_drawdown['fast_ema']}")
        print(f"  Slow EMA: {best_drawdown['slow_ema']}")
        print(f"  ATR Multiplier: {best_drawdown['trailing_stop_atr_multiplier']}")
        print(f"  Max Drawdown: {best_drawdown['max_drawdown']*100:.2f}%")
        print(f"  Sharpe: {best_drawdown['sharpe_ratio']:.2f}")
        print(f"  Annual Return: {best_drawdown['annual_return']*100:.2f}%")
        
        # Plot parameter performance heatmaps
        self._plot_parameter_heatmaps(results_df)
        
        return {
            'results_df': results_df,
            'best_sharpe': best_sharpe.to_dict(),
            'best_sortino': best_sortino.to_dict(),
            'best_return': best_return.to_dict(),
            'best_drawdown': best_drawdown.to_dict()
        }
    
    def _plot_parameter_heatmaps(self, results_df):
        """
        Plot heatmaps for parameter optimization results.
        
        Parameters:
        -----------
        results_df : pandas.DataFrame
            DataFrame with optimization results
        """
        # Get unique parameter values
        fast_ema_values = sorted(results_df['fast_ema'].unique())
        slow_ema_values = sorted(results_df['slow_ema'].unique())
        atr_values = sorted(results_df['trailing_stop_atr_multiplier'].unique())
        
        # Create heatmap data structures
        if len(atr_values) > 1:
            # For each ATR value, create a heatmap of EMA combinations
            for atr_mult in atr_values:
                # Filter data for this ATR multiplier
                atr_data = results_df[results_df['trailing_stop_atr_multiplier'] == atr_mult]
                
                # Prepare matrices for heatmaps
                sharpe_matrix = np.zeros((len(fast_ema_values), len(slow_ema_values)))
                return_matrix = np.zeros((len(fast_ema_values), len(slow_ema_values)))
                
                # Fill matrices
                for i, fast in enumerate(fast_ema_values):
                    for j, slow in enumerate(slow_ema_values):
                        # Find matching row
                        mask = (atr_data['fast_ema'] == fast) & (atr_data['slow_ema'] == slow)
                        if mask.any():
                            row = atr_data[mask]
                            sharpe_matrix[i, j] = row['sharpe_ratio'].values[0]
                            return_matrix[i, j] = row['annual_return'].values[0] * 100  # Convert to percentage
                
                # Create heatmaps
                fig, axes = plt.subplots(1, 2, figsize=(16, 6))
                
                # Sharpe ratio heatmap
                im1 = axes[0].imshow(sharpe_matrix, cmap='viridis')
                axes[0].set_title(f'Sharpe Ratio (ATR Mult = {atr_mult})', fontsize=14)
                axes[0].set_xlabel('Slow EMA Period', fontsize=12)
                axes[0].set_ylabel('Fast EMA Period', fontsize=12)
                axes[0].set_xticks(np.arange(len(slow_ema_values)))
                axes[0].set_yticks(np.arange(len(fast_ema_values)))
                axes[0].set_xticklabels(slow_ema_values)
                axes[0].set_yticklabels(fast_ema_values)
                plt.colorbar(im1, ax=axes[0])
                
                # Annual return heatmap
                im2 = axes[1].imshow(return_matrix, cmap='plasma')
                axes[1].set_title(f'Annual Return % (ATR Mult = {atr_mult})', fontsize=14)
                axes[1].set_xlabel('Slow EMA Period', fontsize=12)
                axes[1].set_ylabel('Fast EMA Period', fontsize=12)
                axes[1].set_xticks(np.arange(len(slow_ema_values)))
                axes[1].set_yticks(np.arange(len(fast_ema_values)))
                axes[1].set_xticklabels(slow_ema_values)
                axes[1].set_yticklabels(fast_ema_values)
                plt.colorbar(im2, ax=axes[1])
                
                plt.tight_layout()
                plt.show()
        
        # Create 3D surface plots for all parameters
        from mpl_toolkits.mplot3d import Axes3D
        
        # Group by fast and slow EMA, taking the best ATR multiplier for each combination
        grouped = results_df.groupby(['fast_ema', 'slow_ema'])['sharpe_ratio'].max().reset_index()
        
        # Create mesh grid
        fast_emas = sorted(grouped['fast_ema'].unique())
        slow_emas = sorted(grouped['slow_ema'].unique())
        X, Y = np.meshgrid(fast_emas, slow_emas)
        
        # Create Z matrix (sharpe values)
        Z = np.zeros((len(slow_emas), len(fast_emas)))
        for i, slow in enumerate(slow_emas):
            for j, fast in enumerate(fast_emas):
                mask = (grouped['fast_ema'] == fast) & (grouped['slow_ema'] == slow)
                if mask.any():
                    Z[i, j] = grouped[mask]['sharpe_ratio'].values[0]
        
        # Plot 3D surface
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
        
        ax.set_title('Parameter Optimization Surface (Best Sharpe Ratio)', fontsize=14)
        ax.set_xlabel('Fast EMA Period', fontsize=12)
        ax.set_ylabel('Slow EMA Period', fontsize=12)
        ax.set_zlabel('Sharpe Ratio', fontsize=12)
        
        # Add color bar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        plt.tight_layout()
        plt.show()
    
    def run_with_mt5(self, live_trading=False, paper_trading=False):
        """
        Run the strategy with MetaTrader 5 integration for live or paper trading.
        
        Parameters:
        -----------
        live_trading : bool
            Whether to execute real trades
        paper_trading : bool
            Whether to simulate trades without real execution
            
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        # This is a placeholder for MT5 integration
        # In a real implementation, you would connect to MT5 using its API
        
        try:
            import MetaTrader5 as mt5
        except ImportError:
            print("Error: MetaTrader5 package is not installed. Use 'pip install MetaTrader5' to install it.")
            return False
        
        if not mt5.initialize():
            print(f"MT5 initialization failed. Error code: {mt5.last_error()}")
            return False
        
        print("MetaTrader 5 initialized successfully.")
        print(f"Terminal info: {mt5.terminal_info()}")

        # Check if live or paper trading
        if live_trading:
            print("Running live trading...")
            # Implement live trading logic here
        elif paper_trading:
            print("Running paper trading...")
            # Implement paper trading logic here
        else:
            print("No trading mode selected. Please choose either live or paper trading.")
            return False
        
        # Close MT5 connection
        mt5.shutdown()
        
        return True
        print(f"Version: {mt5.version()}")
#not complete strategy, will complete it later