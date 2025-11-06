# dashboard_visualizer.py
import pandas as pd
from typing import Dict, Any
from datetime import datetime
import numpy as np

class DashboardVisualizer:
    """
    Generates the HTML dashboard from calculated market data.
    """

    def _format_percentage(self, value: float, decimals: int = 2) -> str:
        if pd.isna(value):
            return "N/A"
        color = '#28a745' if value > 0 else '#dc3545'
        return f'<span style="color: {color}; font-weight: 500;">{value:.{decimals}f}%</span>'

    def _get_trend_arrow(self, value: bool) -> str:
        if pd.isna(value):
            return " "
        return '<span style="color: #28a745; font-size: 1.2em;">▲</span>' if value else '<span style="color: #dc3545; font-size: 1.2em;">▼</span>'

    def _generate_bar(self, value: float, color: str = '#007bff', max_val: int = 100) -> str:
        if pd.isna(value):
            return ""
        # Normalize value to be between 0 and 100
        width = min(max(abs(value), 0), max_val)
        return f"""
        <div style="position: relative; height: 20px; background-color: #e9ecef; border-radius: 4px; width: 100px;">
            <div style="position: absolute; left: 0; top: 0; height: 100%; width: {width}%; background-color: {color}; border-radius: 4px;"></div>
            <span style="position: absolute; left: 5px; top: 50%; transform: translateY(-50%); font-size: 12px; color: #fff; text-shadow: 1px 1px 1px #000;">{value:.1f}</span>
        </div>
        """

    def _generate_main_table(self, df: pd.DataFrame, title: str) -> str:
        if df.empty:
            return f'<h2>{title}</h2><p>No data available.</p>'

        headers = [
            'ticker', 'index', 'price', '% 1D', 'Relative Strength', 'RS STS %',
            '% YTD', '% 1W', '% 1M', '% 1Y', '% From 52W High',
            '10MA', '20MA', '50MA', '200MA', '20>50MA', '50>200MA'
        ]

        # Ensure all columns exist, fill with NaN if they don't
        for col in headers:
            if col not in df.columns:
                df[col] = np.nan

        # Reorder df to match headers
        df = df[headers]

        table_rows = ""
        for _, row in df.iterrows():
            table_rows += f"""
            <tr>
                <td><strong>{row['ticker']}</strong><br><span style="font-size: 11px; color: #6c757d;">{row['index']}</span></td>
                <td>${row['price']:.2f}</td>
                <td>{self._format_percentage(row['% 1D'])}</td>
                <td>{self._generate_bar(row['Relative Strength'], '#6c757d')}</td>
                <td>{self._generate_bar(row['RS STS %'], '#007bff')}</td>
                <td>{self._format_percentage(row['% YTD'])}</td>
                <td>{self._format_percentage(row['% 1W'])}</td>
                <td>{self._format_percentage(row['% 1M'])}</td>
                <td>{self._format_percentage(row['% 1Y'])}</td>
                <td>{self._generate_bar(row['% From 52W High'], '#dc3545', 50)}</td>
                <td style="text-align: center;">{self._get_trend_arrow(row['10MA'])}</td>
                <td style="text-align: center;">{self._get_trend_arrow(row['20MA'])}</td>
                <td style="text-align: center;">{self._get_trend_arrow(row['50MA'])}</td>
                <td style="text-align: center;">{self._get_trend_arrow(row['200MA'])}</td>
                <td style="text-align: center;">{self._get_trend_arrow(row['20>50MA'])}</td>
                <td style="text-align: center;">{self._get_trend_arrow(row['50>200MA'])}</td>
            </tr>
            """

        html = f"""
        <div class="table-container">
            <h3>{title}</h3>
            <table>
                <thead>
                    <tr>
                        <th rowspan="2">Ticker</th>
                        <th rowspan="2">Price</th>
                        <th rowspan="2">% 1D</th>
                        <th rowspan="2">Relative Strength</th>
                        <th rowspan="2">RS STS %</th>
                        <th colspan="4">Performance</th>
                        <th colspan="1">Highs</th>
                        <th colspan="6">Trend Indicators (MAs)</th>
                    </tr>
                    <tr>
                        <th>% YTD</th>
                        <th>% 1W</th>
                        <th>% 1M</th>
                        <th>% 1Y</th>
                        <th>% From 52W High</th>
                        <th>10MA</th>
                        <th>20MA</th>
                        <th>50MA</th>
                        <th>200MA</th>
                        <th>20>50MA</th>
                        <th>50>200MA</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
        </div>
        """
        return html

    def _get_exposure_arrow(self, score: float) -> str:
        if 80 <= score <= 100: return "1"
        if 60 <= score < 80:  return "2"
        if 40 <= score < 60:  return "3"
        if 20 <= score < 40:  return "4"
        if 0 <= score < 20:   return "5"
        return "0"

    def _generate_exposure_section(self, exposure: Dict) -> str:
        score = exposure.get('score', 0)
        level = exposure.get('level', 'N/A')
        arrow_pos = self._get_exposure_arrow(score)

        return f"""
        <div class="grid-item">
            <h3>Market Exposure</h3>
            <div class="exposure-grid">
                <table class="exposure-table">
                    <tbody>
                        <tr class="ex-bullish"><td>Bullish</td><td>100% - 80%</td><td class="arrow-cell">{'<span class="arrow">◀</span>' if arrow_pos == '1' else ''}</td></tr>
                        <tr class="ex-positive"><td>Positive</td><td>80% - 60%</td><td class="arrow-cell">{'<span class="arrow">◀</span>' if arrow_pos == '2' else ''}</td></tr>
                        <tr class="ex-neutral"><td>Neutral</td><td>60% - 40%</td><td class="arrow-cell">{'<span class="arrow">◀</span>' if arrow_pos == '3' else ''}</td></tr>
                        <tr class="ex-negative"><td>Negative</td><td>40% - 20%</td><td class="arrow-cell">{'<span class="arrow">◀</span>' if arrow_pos == '4' else ''}</td></tr>
                        <tr class="ex-bearish"><td>Bearish</td><td>20% - 0%</td><td class="arrow-cell">{'<span class="arrow">◀</span>' if arrow_pos == '5' else ''}</td></tr>
                    </tbody>
                </table>
                <div class="exposure-score">
                    <div>{score:.2f}%</div>
                    <div style="font-size: 16px;">{level}</div>
                </div>
            </div>
            <canvas id="exposureChart" style="margin-top: 20px;"></canvas>
        </div>
        """

    def _generate_screener_section(self, screener_results: Dict) -> str:
        if not screener_results:
            return ""

        screener_names = {
            'momentum_97': 'Momentum 97',
            'explosive_eps': 'Explosive Estimated EPS Growth Stocks (RS STS % ≧ 80)',
            'healthy_chart': 'Healthy Chart Stocks',
            'up_on_volume': 'Up on Volume List (RS STS % ≧ 80)',
            'top_2_rs': 'Top 2% Rs Rating List',
            'bullish_4pct': '4% Bullish Yesterday (RS STS % ≧ 80)'
        }

        html = ""
        for key, name in screener_names.items():
            tickers = screener_results.get(key, pd.DataFrame())
            ticker_list = tickers['Ticker'].tolist() if not tickers.empty else []

            html += f"""
            <div class="grid-item screener-item">
                <h5>{name}</h5>
                <div class="ticker-list">
                    {' '.join([f'<span>{t}</span>' for t in ticker_list]) if ticker_list else 'No stocks found.'}
                </div>
            </div>
            """
        return html

    def generate_html_dashboard(self,
                                exposure: Dict,
                                market_performance: pd.DataFrame,
                                sectors_performance: pd.DataFrame,
                                macro_performance: pd.DataFrame,
                                screener_results: Dict[str, pd.DataFrame]
                               ) -> str:

        # Load exposure history
        try:
            exposure_history_df = pd.read_csv('market_exposure_history.csv')
            exposure_dates = exposure_history_df['date'].tolist()
            exposure_scores = exposure_history_df['score'].tolist()
        except FileNotFoundError:
            exposure_dates = []
            exposure_scores = []

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Market Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; background-color: #f8f9fa; color: #212529; margin: 0; padding: 20px; }}
        .container {{ max-width: 1800px; margin: 0 auto; }}
        h1, h3, h5 {{ margin: 0; }}
        h1 {{ margin-bottom: 5px; }}
        .header {{ margin-bottom: 20px; }}
        .grid-container {{ display: grid; grid-template-columns: 2.5fr 1fr; gap: 20px; }}
        .main-content {{ display: flex; flex-direction: column; gap: 20px; }}
        .right-sidebar {{ display: flex; flex-direction: column; gap: 20px; }}
        .grid-item {{ background-color: #ffffff; border: 1px solid #dee2e6; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }}
        .table-container {{ overflow-x: auto; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
        th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #dee2e6; }}
        thead th {{ background-color: #f8f9fa; font-weight: 600; text-align: center; vertical-align: middle; }}
        thead th[rowspan="2"] {{ border-right: 1px solid #dee2e6; }}
        tbody tr:hover {{ background-color: #f1f3f5; }}
        .exposure-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; align-items: center; }}
        .exposure-table {{ font-size: 14px; }}
        .exposure-table td {{ border: none; padding: 6px; }}
        .exposure-table .arrow-cell {{ width: 30px; }}
        .exposure-table .arrow {{ color: #dc3545; font-size: 24px; }}
        .exposure-score {{ text-align: center; font-size: 28px; font-weight: bold; }}
        .ex-bullish {{ background-color: rgba(40, 167, 69, 0.2); }}
        .ex-positive {{ background-color: rgba(40, 167, 69, 0.1); }}
        .ex-neutral {{ background-color: rgba(255, 193, 7, 0.1); }}
        .ex-negative {{ background-color: rgba(220, 53, 69, 0.1); }}
        .ex-bearish {{ background-color: rgba(220, 53, 69, 0.2); }}
        .screener-item h5 {{ margin-bottom: 10px; }}
        .ticker-list {{ display: flex; flex-wrap: wrap; gap: 5px; }}
        .ticker-list span {{ background-color: #e9ecef; border-radius: 4px; padding: 3px 8px; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Market Dashboard</h1>
            <span>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</span>
        </div>
        <div class="grid-container">
            <div class="main-content">
                {self._generate_main_table(market_performance, "Market")}
                {self._generate_main_table(sectors_performance, "Sectors")}
                {self._generate_main_table(macro_performance, "Macro")}
            </div>
            <div class="right-sidebar">
                {self._generate_exposure_section(exposure)}
                {self._generate_screener_section(screener_results)}
            </div>
        </div>
    </div>
    <script>
        const ctx = document.getElementById('exposureChart').getContext('2d');
        const exposureChart = new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: {exposure_dates},
                datasets: [{{
                    label: 'Market Exposure Score',
                    data: {exposure_scores},
                    borderColor: 'rgba(0, 123, 255, 1)',
                    backgroundColor: 'rgba(0, 123, 255, 0.1)',
                    borderWidth: 2,
                    pointRadius: 3,
                    fill: true,
                    tension: 0.1
                }}]
            }},
            options: {{
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 100
                    }}
                }},
                plugins: {{
                    legend: {{
                        display: false
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
        """
        return html

    def save_html(self, html_content: str, filename: str = 'market_dashboard.html'):
        """Saves the HTML content to a file."""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"Dashboard saved to: {filename}")
