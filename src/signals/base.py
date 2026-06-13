"""
Base signal module.
Defines the interface for all trading signals.
"""

from abc import ABC, abstractmethod
import pandas as pd

class SignalBase(ABC):
    """Abstract base class for all trading signal generators."""
    
    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> int:
        """
        Analyzes data and generates a trading signal.
        
        Args:
            data (pd.DataFrame): Historical data.
            
        Returns:
            int: 1 for Buy, -1 for Sell, 0 for Hold.
        """
        pass
