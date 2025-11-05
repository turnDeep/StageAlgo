"""
Core Layer - Infrastructure and Basic Indicators
"""

from .infrastructure.fmp_client import FMPDataFetcher, fetch_stock_data
from .infrastructure.data_fetcher import fetch_stock_data as fetch_data

__all__ = [
    "FMPDataFetcher",
    "fetch_stock_data",
    "fetch_data",
]
