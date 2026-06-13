"""
Base broker module.
Defines interface for trade execution.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class BrokerBase(ABC):
    """Abstract base class for all brokers (Live and Paper)."""
    
    @abstractmethod
    def get_balance(self) -> float:
        """Returns available capital."""
        pass
        
    @abstractmethod
    def submit_order(self, symbol: str, qty: float, side: str, order_type: str = 'market', price: Optional[float] = None) -> Dict[str, Any]:
        """Submits an order."""
        pass
        
    @abstractmethod
    def get_positions(self) -> Dict[str, float]:
        """Returns current positions."""
        pass
