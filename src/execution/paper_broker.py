"""
Paper broker module.
Simulates trade execution with slippage and fees.
"""

from typing import Dict, Any, Optional
from src.execution.broker_base import BrokerBase
from src.utils.logger import logger

class PaperBroker(BrokerBase):
    """
    Simulated broker for backtesting and paper trading.
    Applies fees and slippage.
    """
    def __init__(self, initial_capital: float, fee_rate: float, slippage_pct: float = 0.0005) -> None:
        self.capital = initial_capital
        self.fee_rate = fee_rate
        self.slippage_pct = slippage_pct
        self.positions: Dict[str, float] = {}
        
    def get_balance(self) -> float:
        return self.capital
        
    def get_positions(self) -> Dict[str, float]:
        return self.positions

    def submit_order(self, symbol: str, qty: float, side: str, order_type: str = 'market', price: Optional[float] = None) -> Dict[str, Any]:
        if price is None:
            logger.error("PaperBroker requires a price for execution simulation.")
            return {"status": "error", "message": "Price required"}
            
        # Apply slippage
        exec_price = price * (1 + self.slippage_pct) if side == 'buy' else price * (1 - self.slippage_pct)
        
        cost = exec_price * qty
        fee = cost * self.fee_rate
        
        if side == 'buy':
            if self.capital < (cost + fee):
                logger.warning(f"Insufficient capital for buy. Needed {cost+fee}, have {self.capital}")
                return {"status": "rejected", "reason": "insufficient_funds"}
            self.capital -= (cost + fee)
            self.positions[symbol] = self.positions.get(symbol, 0.0) + qty
        elif side == 'sell':
            current_qty = self.positions.get(symbol, 0.0)
            if current_qty < qty:
                logger.warning(f"Insufficient position for sell. Needed {qty}, have {current_qty}")
                return {"status": "rejected", "reason": "insufficient_position"}
            self.capital += (cost - fee)
            self.positions[symbol] -= qty
            
        logger.info(f"Paper Executed: {side.upper()} {qty} {symbol} @ {exec_price:.2f} (Fee: {fee:.2f})")
        
        return {
            "status": "filled",
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "price": exec_price,
            "fee": fee
        }
