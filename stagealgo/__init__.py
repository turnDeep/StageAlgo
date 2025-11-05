"""
StageAlgo - Technical Analysis and Stock Screening System

レイヤードアーキテクチャに基づく株式分析・スクリーニングシステム
"""

__version__ = "2.0.0"

# Core exports
from .core.infrastructure.fmp_client import FMPDataFetcher, fetch_stock_data
from .core.infrastructure.data_fetcher import fetch_stock_data as fetch_data

# Analysis exports
from .analysis.stage.detector import StageDetector
from .screeners.base_screener import BaseScreener

__all__ = [
    "FMPDataFetcher",
    "fetch_stock_data",
    "fetch_data",
    "StageDetector",
    "BaseScreener",
]
