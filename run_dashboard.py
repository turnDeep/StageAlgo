# run_dashboard.py
"""
Market Dashboard Runner
ダッシュボードの実行とJSON/HTML生成
"""

import json
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Skipping .env file loading.")
    pass

from market_dashboard import MarketDashboard
from dashboard_visualizer import DashboardVisualizer


def main():
    """
    ダッシュボードを生成してJSON/HTMLに出力
    """
    print("Initializing Market Dashboard...")
    dashboard = MarketDashboard()

    print("\nCalculating market metrics...")

    # データ収集とダッシュボード生成
    exposure, market_performance, sectors_performance, macro_performance, screener_results = dashboard.generate_dashboard()

    # 追加データの取得
    print("\nCalculating additional metrics...")
    factors_vs_sp500 = dashboard.calculate_factors_vs_sp500()
    bond_yields = dashboard.get_bond_yields()
    power_trend = dashboard.calculate_power_trend()

    print("\nFactors vs SP500:")
    for name, value in factors_vs_sp500.items():
        print(f"  {name}: {value:+.2f}%")

    print("\nBond Yields:")
    for name, value in bond_yields.items():
        print(f"  {name}: {value:.2f}%")

    if power_trend:
        print("\nPower Trend:")
        print(f"  RSI: {power_trend.get('rsi', 0):.2f}")
        print(f"  MACD Histogram: {power_trend.get('macd_histogram', 0):.2f}")
        print(f"  Trend: {power_trend.get('trend', 'N/A')}")

    # すべての個別銘柄のパフォーマンスデータを計算
    print("\n" + "=" * 80)
    print("CALCULATING INDIVIDUAL STOCKS PERFORMANCE")
    print("=" * 80)
    individual_stocks = dashboard.calculate_all_stocks_performance()

    # JSONデータの準備
    dashboard_data = {
        'generated_at': dashboard.current_date.strftime('%Y-%m-%d %H:%M:%S'),
        'exposure': exposure,
        'factors_vs_sp500': factors_vs_sp500,
        'bond_yields': bond_yields,
        'power_trend': power_trend,
        'market_performance': {
            'data': market_performance.to_dict('records') if not market_performance.empty else []
        },
        'sectors_performance': {
            'data': sectors_performance.to_dict('records') if not sectors_performance.empty else []
        },
        'macro_performance': {
            'data': macro_performance.to_dict('records') if not macro_performance.empty else []
        },
        'individual_stocks': {
            'data': individual_stocks.to_dict('records') if not individual_stocks.empty else [],
            'count': len(individual_stocks) if not individual_stocks.empty else 0
        },
        'screener_results': {}
    }

    # スクリーナー結果を追加
    if screener_results:
        for name, df in screener_results.items():
            dashboard_data['screener_results'][name] = {
                'data': df.to_dict('records') if not df.empty else []
            }

    # 個別銘柄データのサマリーを表示
    if not individual_stocks.empty:
        print(f"\n✓ Individual stocks data: {len(individual_stocks)} stocks")
        print(f"  Top 10 by RS Rating:")
        top_10 = individual_stocks.head(10)[['Ticker', 'Price', 'RS Rating', '% 1D', '% 1M', '% YTD', 'Stage']]
        print(top_10.to_string(index=False))

    # JSONファイルに保存
    json_output = 'market_dashboard_data.json'
    print(f"\nSaving data to JSON: {json_output}")
    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump(dashboard_data, f, indent=2, ensure_ascii=False, default=str)
    print(f"✓ JSON data saved to: {json_output}")

    # HTML生成
    print("\nGenerating HTML dashboard...")
    visualizer = DashboardVisualizer()
    html = visualizer.generate_html_dashboard(
        exposure=exposure,
        market_performance=market_performance,
        sectors_performance=sectors_performance,
        macro_performance=macro_performance,
        screener_results=screener_results,
        factors_vs_sp500=factors_vs_sp500,
        bond_yields=bond_yields,
        power_trend=power_trend
    )
    visualizer.save_html(html, 'market_dashboard.html')

    print("\n✅ Dashboard generation complete!")
    print(f"📊 JSON data: {json_output}")
    print(f"🌐 HTML dashboard: market_dashboard.html")


if __name__ == '__main__':
    main()
