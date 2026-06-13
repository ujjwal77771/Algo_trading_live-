"""
Structured logger for the algo trading system.
Provides console and file logging.
"""

import logging
import sys
from typing import Optional

def setup_logger(name: str, level: int = logging.INFO, log_file: str = "trading.log") -> logging.Logger:
    """
    Sets up a logger with both console and file handlers.
    
    Args:
        name (str): Name of the logger.
        level (int): Logging level.
        log_file (str): Path to the log file.
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if logger.handlers:
        return logger

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    return logger

# Default logger instance
logger = setup_logger("AlgoTrading", level=logging.INFO)
