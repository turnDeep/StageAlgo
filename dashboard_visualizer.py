# dashboard_visualizer.py
import pandas as pd
from typing import Dict, Any
from datetime import datetime
import numpy as np
import json

class DashboardVisualizer:
    """Market Dashboard HTML Generator (Grid Layout)"""

    def generate_html_dashboard(self,
                                exposure: Dict,
                                market_performance: pd.DataFrame,
                                sectors_performance: pd.DataFrame,
                                macro_performance: pd.DataFrame,
                                screener_results: Dict[str, pd.DataFrame],
                                factors_vs_sp500: Dict[str, float] = None,
                                bond_yields: Dict[str, float] = None,
                                power_trend: Dict = None) -> str:
        """Generate complete HTML dashboard with 3-column grid layout"""

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
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: #f5f5f5;
            padding: 10px;
            font-size: 12px;
        }}
        .container {{ max-width: 1920px; margin: 0 auto; }}

        /* Header */
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 15px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .header h1 {{ font-size: 28px; margin: 0; }}
        .header .date {{ font-size: 14px; opacity: 0.9; }}

        /* 3-Column Grid Layout */
        .dashboard-grid {{
            display: grid;
            grid-template-columns: 300px 1fr 320px;
            gap: 12px;
            margin-bottom: 15px;
        }}

        /* Panel Styles */
        .panel {{
            background: white;
            border-radius: 6px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.08);
            padding: 12px;
        }}
        .panel-header {{
            font-weight: 700;
            font-size: 13px;
            color: #333;
            border-bottom: 2px solid #4a5568;
            padding-bottom: 6px;
            margin-bottom: 10px;
        }}

        /* Market Exposure Gauge */
        .exposure-gauge {{
            text-align: center;
            padding: 15px 0;
        }}
        .exposure-score {{
            font-size: 42px;
            font-weight: bold;
            color: #2d3748;
            margin: 10px 0;
        }}
        .exposure-level {{
            font-size: 16px;
            padding: 8px 16px;
            border-radius: 20px;
            display: inline-block;
            margin: 10px 0;
        }}
        .level-bullish {{ background: #48bb78; color: white; }}
        .level-positive {{ background: #68d391; color: white; }}
        .level-neutral {{ background: #ecc94b; color: #2d3748; }}
        .level-negative {{ background: #fc8181; color: white; }}
        .level-bearish {{ background: #f56565; color: white; }}

        /* Factor Cards */
        .factor-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
            margin-top: 10px;
        }}
        .factor-card {{
            background: #f7fafc;
            padding: 8px;
            border-radius: 4px;
            border-left: 3px solid #cbd5e0;
        }}
        .factor-card.positive {{ border-left-color: #48bb78; }}
        .factor-card.negative {{ border-left-color: #f56565; }}
        .factor-name {{ font-size: 11px; color: #718096; font-weight: 600; }}
        .factor-value {{
            font-size: 16px;
            font-weight: bold;
            margin-top: 4px;
        }}
        .factor-value.positive {{ color: #38a169; }}
        .factor-value.negative {{ color: #e53e3e; }}

        /* Screener Badges */
        .screener-section {{
            margin-bottom: 15px;
        }}
        .ticker-badge {{
            display: inline-block;
            padding: 4px 8px;
            margin: 3px;
            border-radius: 4px;
            font-weight: 600;
            font-size: 11px;
            border: 1px solid;
        }}
        .badge-green {{
            background: #c6f6d5;
            color: #22543d;
            border-color: #9ae6b4;
        }}
        .badge-blue {{
            background: #bee3f8;
            color: #2c5282;
            border-color: #90cdf4;
        }}
        .badge-yellow {{
            background: #feebc8;
            color: #744210;
            border-color: #fbd38d;
        }}

        /* Performance Table */
        .perf-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 11px;
        }}
        .perf-table th {{
            background: #edf2f7;
            padding: 6px 4px;
            text-align: left;
            font-weight: 600;
            border-bottom: 2px solid #cbd5e0;
            font-size: 10px;
        }}
        .perf-table td {{
            padding: 6px 4px;
            border-bottom: 1px solid #e2e8f0;
        }}
        .perf-table tr:hover {{ background: #f7fafc; }}
        .pct-positive {{ color: #38a169; font-weight: 600; }}
        .pct-negative {{ color: #e53e3e; font-weight: 600; }}

        /* RS History Sparkline */
        .rs-sparkline {{
            display: flex;
            align-items: flex-end;
            height: 20px;
            gap: 1px;
        }}
        .rs-bar {{
            flex: 1;
            background: #4299e1;
            opacity: 0.7;
        }}
        .rs-bar.negative {{
            background: #fc8181;
        }}

        /* Trend Arrows */
        .arrow {{ font-size: 14px; }}
        .arrow-up {{ color: #38a169; }}
        .arrow-down {{ color: #e53e3e; }}

        /* Center column spacing */
        .center-column {{
            display: flex;
            flex-direction: column;
            gap: 12px;
        }}

        /* Left/Right column spacing */
        .left-column, .right-column {{
            display: flex;
            flex-direction: column;
            gap: 12px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h1>Market Dashboard</h1>
                <div class="date">Generated: {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}</div>
            </div>
        </div>

        <div class="dashboard-grid">
            <!-- LEFT COLUMN -->
            <div class="left-column">
                {self._generate_exposure_panel(exposure, exposure_dates, exposure_scores)}
                {self._generate_factors_panel(factors_vs_sp500)}
                {self._generate_macro_panel(macro_performance, bond_yields)}
            </div>

            <!-- CENTER COLUMN -->
            <div class="center-column">
                {self._generate_market_table(market_performance, "Market Performance")}
                {self._generate_sector_table(sectors_performance, "Sectors Performance")}
            </div>

            <!-- RIGHT COLUMN -->
            <div class="right-column">
                {self._generate_screeners_column(screener_results)}
            </div>
        </div>
    </div>

    <script>
        // Exposure chart
        const ctx = document.getElementById('exposureChart');
        if (ctx) {{
            new Chart(ctx.getContext('2d'), {{
                type: 'line',
                data: {{
                    labels: {json.dumps(exposure_dates[-30:])},
                    datasets: [{{
                        label: 'Market Exposure',
                        data: {json.dumps(exposure_scores[-30:])},
                        borderColor: '#4299e1',
                        backgroundColor: 'rgba(66, 153, 225, 0.1)',
                        borderWidth: 2,
                        tension: 0.3
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{ y: {{ min: 0, max: 100 }} }},
                    plugins: {{ legend: {{ display: false }} }}
                }}
            }});
        }}
    </script>
</body>
</html>
        """
        return html

    def _generate_exposure_panel(self, exposure: Dict, exposure_dates: list, exposure_scores: list) -> str:
        score = exposure.get('score', 0)
        level = exposure.get('level', 'Neutral')
        level_class = f"level-{level.lower()}"

        return f"""
        <div class="panel">
            <div class="panel-header">Market Exposure</div>
            <div class="exposure-gauge">
                <div class="exposure-score">{score:.1f}%</div>
                <div class="exposure-level {level_class}">{level}</div>
                <canvas id="exposureChart" style="height: 120px; margin-top: 10px;"></canvas>
            </div>
        </div>
        """

    def _generate_factors_panel(self, factors: Dict[str, float]) -> str:
        if not factors:
            return ""

        cards_html = ""
        for name, value in factors.items():
            is_positive = value > 0
            card_class = "positive" if is_positive else "negative"
            value_class = "positive" if is_positive else "negative"
            sign = "+" if is_positive else ""

            cards_html += f"""
            <div class="factor-card {card_class}">
                <div class="factor-name">{name}</div>
                <div class="factor-value {value_class}">{sign}{value:.2f}%</div>
            </div>
            """

        return f"""
        <div class="panel">
            <div class="panel-header">Factors vs SP500 (Yesterday)</div>
            <div class="factor-grid">
                {cards_html}
            </div>
        </div>
        """

    def _generate_macro_panel(self, macro_df: pd.DataFrame, bonds: Dict[str, float]) -> str:
        macro_rows = ""
        if not macro_df.empty:
            for _, row in macro_df.head(3).iterrows():
                pct_1d = row.get('% 1D', 0)
                pct_class = "pct-positive" if pct_1d > 0 else "pct-negative"
                macro_rows += f"""
                <tr>
                    <td><strong>{row.get('ticker', '')}</strong></td>
                    <td style="text-align:right" class="{pct_class}">{pct_1d:+.2f}%</td>
                </tr>
                """

        bonds_rows = ""
        if bonds:
            for name, value in bonds.items():
                bonds_rows += f"""
                <tr>
                    <td><strong>{name}</strong></td>
                    <td style="text-align:right">{value:.2f}%</td>
                </tr>
                """

        return f"""
        <div class="panel">
            <div class="panel-header">Macro & Bonds</div>
            <table class="perf-table" style="margin-bottom:10px">
                {macro_rows}
                {bonds_rows}
            </table>
        </div>
        """

    def _generate_market_table(self, df: pd.DataFrame, title: str) -> str:
        if df.empty:
            return f'<div class="panel"><div class="panel-header">{title}</div><p>No data</p></div>'

        rows_html = ""
        for _, row in df.iterrows():
            rs_history = row.get('RS History', [0]*20)
            sparkline = self._create_sparkline(rs_history)

            rows_html += f"""
            <tr>
                <td><strong>{row.get('ticker', '')}</strong></td>
                <td>{self._format_pct(row.get('% 1D', 0))}</td>
                <td>{sparkline}</td>
                <td>{row.get('RS STS %', 0):.0f}</td>
                <td>{self._format_pct(row.get('% YTD', 0))}</td>
                <td>{self._format_pct(row.get('% 1W', 0))}</td>
                <td>{self._format_pct(row.get('% 1M', 0))}</td>
                <td>{self._arrow(row.get('10MA', False))}</td>
                <td>{self._arrow(row.get('50MA', False))}</td>
                <td>{self._arrow(row.get('200MA', False))}</td>
            </tr>
            """

        return f"""
        <div class="panel">
            <div class="panel-header">{title}</div>
            <table class="perf-table">
                <thead>
                    <tr>
                        <th>Ticker</th>
                        <th>1D%</th>
                        <th>RS History</th>
                        <th>RS%</th>
                        <th>YTD%</th>
                        <th>1W%</th>
                        <th>1M%</th>
                        <th>10MA</th>
                        <th>50MA</th>
                        <th>200MA</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
        """

    def _generate_sector_table(self, df: pd.DataFrame, title: str) -> str:
        # Similar to market table
        return self._generate_market_table(df, title)

    def _generate_screeners_column(self, screeners: Dict[str, pd.DataFrame]) -> str:
        screener_names = {
            'momentum_97': 'Momentum 97',
            'explosive_eps': 'Explosive EPS Growth (RS≥80)',
            'healthy_chart': 'Healthy Chart',
            'up_on_volume': 'Up on Volume (RS≥80)',
            'top_2_rs': 'Top 2% RS Rating',
            'bullish_4pct': '4% Bullish Yesterday (RS≥80)'
        }

        panels_html = ""
        for key, name in screener_names.items():
            df = screeners.get(key, pd.DataFrame())
            tickers = df['Ticker'].tolist() if 'Ticker' in df.columns else []

            badges = ""
            for ticker in tickers[:15]:  # Limit to 15
                badge_class = self._get_badge_class(key)
                badges += f'<span class="ticker-badge {badge_class}">{ticker}</span>'

            if not badges:
                badges = '<span style="color:#a0aec0">No stocks found</span>'

            panels_html += f"""
            <div class="panel screener-section">
                <div class="panel-header">{name}</div>
                <div>
                    {badges}
                </div>
            </div>
            """

        return panels_html

    def _create_sparkline(self, values: list) -> str:
        if not values or len(values) == 0:
            return '<div class="rs-sparkline"></div>'

        max_val = max(abs(v) for v in values)
        if max_val == 0: max_val = 1

        bars = ""
        for v in values[-20:]:
            height_pct = min(100, abs(v) / max_val * 100)
            bar_class = "rs-bar" if v >= 0 else "rs-bar negative"
            bars += f'<div class="{bar_class}" style="height:{height_pct}%"></div>'

        return f'<div class="rs-sparkline">{bars}</div>'

    def _format_pct(self, value: float) -> str:
        if pd.isna(value): return "N/A"
        pct_class = "pct-positive" if value > 0 else "pct-negative"
        return f'<span class="{pct_class}">{value:+.2f}%</span>'

    def _arrow(self, is_above: bool) -> str:
        if pd.isna(is_above): return ""
        if is_above:
            return '<span class="arrow-up">▲</span>'
        else:
            return '<span class="arrow-down">▼</span>'

    def _get_badge_class(self, screener_key: str) -> str:
        # Color-code badges by screener type
        if 'momentum' in screener_key: return 'badge-green'
        if 'eps' in screener_key or 'rs' in screener_key: return 'badge-blue'
        return 'badge-yellow'

    def save_html(self, html_content: str, filename: str = 'market_dashboard.html'):
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"✓ Dashboard saved to: {filename}")
