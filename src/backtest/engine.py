"""
Backtest engine module.
Event-driven backtesting engine.
"""

import pandas as pd
from typing import Dict, Any, List
from src.execution.paper_broker import PaperBroker
from src.risk.risk_manager import RiskManager
from src.signals.base import SignalBase
from src.utils.logger import logger

class BacktestEngine:
    """
    Event-driven engine for running historical backtests.
    """
    def __init__(self, data: pd.DataFrame, broker: PaperBroker, risk_manager: RiskManager, signal_generator: SignalBase) -> None:
        self.data = data
        self.broker = broker
        self.risk_manager = risk_manager
        self.signal_generator = signal_generator
        self.equity_curve: List[Dict[str, Any]] = []
        self.trades: List[Dict[str, Any]] = []
        
    def run(self) -> pd.DataFrame:
        """
        Executes the backtest row by row to simulate real-time feed.
        Returns the equity curve dataframe.
        """
        logger.info("Starting backtest...")
        
        symbol = 'BTC/USDT'
        
        # Need ATR for risk management
        from src.features.indicators import calculate_atr
        atr_series = calculate_atr(self.data['high'], self.data['low'], self.data['close'], window=14)
        
        for i in range(50, len(self.data)):
            # Simulate real-time data availability
            current_data = self.data.iloc[:i+1]
            current_row = current_data.iloc[-1]
            current_price = float(current_row['close'])
            current_time = current_data.index[-1]
            current_atr = float(atr_series.iloc[i])
            
            # Evaluate equity
            pos_qty = self.broker.get_positions().get(symbol, 0.0)
            current_equity = self.broker.get_balance() + (pos_qty * current_price)
            self.risk_manager.update_equity(current_equity)
            
            self.equity_curve.append({'timestamp': current_time, 'equity': current_equity})
            
            if self.risk_manager.halted:
                # Liquidate if halted
                if pos_qty > 0:
                    self.broker.submit_order(symbol, pos_qty, 'sell', price=current_price)
                continue
                
            signal = self.signal_generator.generate_signal(current_data)
            
            if signal == 1 and pos_qty == 0:
                # Buy
                qty = self.risk_manager.calculate_position_size(self.broker.get_balance(), current_price, current_atr)
                if qty > 0:
                    res = self.broker.submit_order(symbol, qty, 'buy', price=current_price)
                    if res.get('status') == 'filled':
                        self.trades.append({
                            'timestamp': current_time,
                            'side': 'buy',
                            'price': res['price'],
                            'qty': qty
                        })
            elif signal == -1 and pos_qty > 0:
                # Sell
                res = self.broker.submit_order(symbol, pos_qty, 'sell', price=current_price)
                if res.get('status') == 'filled':
                    self.trades.append({
                        'timestamp': current_time,
                        'side': 'sell',
                        'price': res['price'],
                        'qty': pos_qty,
                        'pnl': (res['price'] - self.trades[-1]['price']) * pos_qty # Simplified PnL
                    })
                    
        logger.info("Backtest completed.")
        return pd.DataFrame(self.equity_curve).set_index('timestamp')
