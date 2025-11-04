# dashboard_visualizer.py
"""
Dashboard Visualization Module
HTMLダッシュボードの生成
"""

import pandas as pd
from typing import Dict
from datetime import datetime


class DashboardVisualizer:
    """
    HTMLダッシュボードの生成
    """

    def __init__(self):
        self.html_content = ""

    def _format_percentage(self, value: float, decimals: int = 2) -> str:
        """パーセンテージを色付きフォーマット"""
        color = 'green' if value > 0 else 'red'
        return f'<span style="color: {color};">{value:.{decimals}f}%</span>'

    def _format_value(self, value: float, decimals: int = 2) -> str:
        """数値を色付きフォーマット"""
        color = 'green' if value > 0 else 'red'
        return f'<span style="color: {color};">{value:.{decimals}f}</span>'

    def generate_html_dashboard(self,
                                exposure: Dict,
                                performance: pd.DataFrame,
                                vix: Dict,
                                sectors: pd.DataFrame,
                                power_law: Dict) -> str:
        """
        HTMLダッシュボードを生成
        """
        # パーセンテージ列を事前にフォーマット
        performance_html = performance.copy()
        if not performance_html.empty:
            for col in ['YTD %', '1W %', '1M %', '1Y %', 'From 52W High %']:
                if col in performance_html.columns:
                    performance_html[col] = performance_html[col].apply(
                        lambda x: self._format_percentage(x)
                    )
            performance_html['Current Price'] = performance_html['Current Price'].apply(
                lambda x: f'${x:.2f}'
            )

        sectors_html = sectors.copy()
        if not sectors_html.empty:
            if '1D %' in sectors_html.columns:
                sectors_html['1D %'] = sectors_html['1D %'].apply(
                    lambda x: self._format_percentage(x)
                )
            if 'Relative Strength' in sectors_html.columns:
                sectors_html['Relative Strength'] = sectors_html['Relative Strength'].apply(
                    lambda x: self._format_value(x)
                )
            if 'RS Rating' in sectors_html.columns:
                sectors_html['RS Rating'] = sectors_html['RS Rating'].apply(
                    lambda x: f'{x:.1f}'
                )
            if 'Price' in sectors_html.columns:
                sectors_html['Price'] = sectors_html['Price'].apply(
                    lambda x: f'${x:.2f}'
                )

        # マーカーの位置を計算 (-60から100の範囲を0-100%に変換)
        marker_position = (exposure['score'] + 60) / 1.6

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Market Dashboard</title>
    <meta charset="utf-8">
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            text-align: center;
            color: #333;
        }}
        .section {{
            margin: 30px 0;
        }}
        .section-title {{
            font-size: 18px;
            font-weight: bold;
            color: #444;
            margin-bottom: 10px;
            border-bottom: 2px solid #007bff;
            padding-bottom: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #007bff;
            color: white;
        }}
        .positive {{
            color: green;
        }}
        .negative {{
            color: red;
        }}
        .exposure-gauge {{
            width: 100%;
            height: 40px;
            background: linear-gradient(to right, #dc3545, #ffc107, #28a745);
            border-radius: 20px;
            position: relative;
            margin: 20px 0;
        }}
        .exposure-marker {{
            position: absolute;
            width: 4px;
            height: 50px;
            background-color: black;
            top: -5px;
            left: {marker_position}%;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .stat-card {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #007bff;
        }}
        .stat-label {{
            font-size: 12px;
            color: #666;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Market Dashboard</h1>
        <p style="text-align: center; color: #666;">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <!-- Market Exposure -->
        <div class="section">
            <div class="section-title">Market Exposure</div>
            <div class="exposure-gauge">
                <div class="exposure-marker"></div>
            </div>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-label">Exposure Score</div>
                    <div class="stat-value {'positive' if exposure['score'] > 0 else 'negative'}">{exposure['score']:.1f}%</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Market Level</div>
                    <div class="stat-value">{exposure['level']}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">VIX Level</div>
                    <div class="stat-value">{exposure.get('vix_level', 'N/A') if exposure.get('vix_level') else 'N/A'}</div>
                </div>
            </div>
        </div>

        <!-- Market Performance -->
        <div class="section">
            <div class="section-title">Market Performance Overview</div>
            {performance_html.to_html(index=False, escape=False) if not performance_html.empty else '<p>No data available</p>'}
        </div>

        <!-- VIX Analysis -->
        <div class="section">
            <div class="section-title">VIX Analysis</div>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-label">Current VIX</div>
                    <div class="stat-value">{vix.get('current', 'N/A') if vix.get('current') else 'N/A'}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Interpretation</div>
                    <div class="stat-value" style="font-size: 16px;">{vix.get('interpretation', 'N/A') if vix else 'N/A'}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">52W High</div>
                    <div class="stat-value">{vix.get('52w_high', 'N/A') if vix.get('52w_high') else 'N/A'}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">52W Low</div>
                    <div class="stat-value">{vix.get('52w_low', 'N/A') if vix.get('52w_low') else 'N/A'}</div>
                </div>
            </div>
        </div>

        <!-- Sector Performance -->
        <div class="section">
            <div class="section-title">Sector Performance</div>
            {sectors_html.to_html(index=False, escape=False) if not sectors_html.empty else '<p>No data available</p>'}
        </div>

        <!-- Power Law Indicators -->
        <div class="section">
            <div class="section-title">Power Law Indicators</div>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-label">5 Days Above 50MA</div>
                    <div class="stat-value">{power_law.get('5d_above_20ma_pct', 0):.1f}%</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">50MA Above 150MA</div>
                    <div class="stat-value">{power_law.get('20ma_above_50ma_pct', 0):.1f}%</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">150MA Above 200MA</div>
                    <div class="stat-value">{power_law.get('50ma_above_200ma_pct', 0):.1f}%</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Total Stocks Analyzed</div>
                    <div class="stat-value">{power_law.get('total', 0)}</div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""
        return html

    def save_html(self, html_content: str, filename: str = 'market_dashboard.html'):
        """
        HTMLをファイルに保存
        """
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"Dashboard saved to: {filename}")


if __name__ == '__main__':
    # テスト用のダミーデータ
    visualizer = DashboardVisualizer()

    exposure = {
        'score': 45.0,
        'level': 'Neutral',
        'vix_level': 18.5
    }

    performance = pd.DataFrame([
        {'Index': 'S&P 500', 'Ticker': 'SPY', 'YTD %': 12.5, '1W %': 2.3, '1M %': 5.1, '1Y %': 18.2, 'From 52W High %': -3.2, 'Current Price': 450.25}
    ])

    vix = {
        'current': 18.5,
        'interpretation': 'Low - Stable Market',
        '52w_high': 35.2,
        '52w_low': 12.1
    }

    sectors = pd.DataFrame([
        {'Sector': 'Technology', 'Ticker': 'XLK', 'Price': 150.25, '1D %': 1.2, 'Relative Strength': 5.3, 'RS Rating': 85.0}
    ])

    power_law = {
        '5d_above_20ma_pct': 65.0,
        '20ma_above_50ma_pct': 55.0,
        '50ma_above_200ma_pct': 45.0,
        'total': 8
    }

    html = visualizer.generate_html_dashboard(exposure, performance, vix, sectors, power_law)
    visualizer.save_html(html, 'test_dashboard.html')
