"""
Base Screener Class

全てのスクリーナーはこのクラスを継承して実装します。
並列処理とプログレス表示を提供します。
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


class BaseScreener(ABC):
    """スクリーナーの基底クラス"""

    def __init__(self, tickers: List[str], workers: Optional[int] = None):
        """
        Initialize screener

        Args:
            tickers: List of ticker symbols to screen
            workers: Number of parallel workers (defaults to CPU count)
        """
        self.tickers = tickers
        self.workers = workers or cpu_count()
        self.results = []

    @abstractmethod
    def screen_single_ticker(self, ticker: str) -> Optional[Dict]:
        """
        単一銘柄のスクリーニングロジック

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dict with screening results, or None if ticker doesn't pass
        """
        pass

    def run(self, show_progress: bool = True) -> pd.DataFrame:
        """
        全銘柄をスクリーニング

        Args:
            show_progress: Show progress bar

        Returns:
            DataFrame with screening results
        """
        self.results = []

        if self.workers == 1:
            # Single-threaded execution
            iterator = tqdm(self.tickers, desc=f"Screening") if show_progress else self.tickers
            for ticker in iterator:
                result = self.screen_single_ticker(ticker)
                if result:
                    self.results.append(result)
        else:
            # Multi-threaded execution
            with Pool(self.workers) as pool:
                iterator = pool.imap_unordered(self.screen_single_ticker, self.tickers)

                if show_progress:
                    iterator = tqdm(
                        iterator,
                        total=len(self.tickers),
                        desc=f"Screening ({self.workers} workers)"
                    )

                for result in iterator:
                    if result:
                        self.results.append(result)

        return pd.DataFrame(self.results)

    @classmethod
    def from_csv(cls, csv_path: str, ticker_column: str = 'Ticker', **kwargs):
        """
        CSVファイルからティッカーリストを読み込んでスクリーナーを作成

        Args:
            csv_path: Path to CSV file
            ticker_column: Name of ticker column
            **kwargs: Additional arguments passed to __init__

        Returns:
            Screener instance
        """
        df = pd.read_csv(csv_path)
        tickers = df[ticker_column].tolist()
        return cls(tickers, **kwargs)

    def save_results(self, output_path: str):
        """
        結果をCSVに保存

        Args:
            output_path: Output CSV file path
        """
        if not self.results:
            print("No results to save")
            return

        df = pd.DataFrame(self.results)
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
