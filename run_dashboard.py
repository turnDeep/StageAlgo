# run_dashboard.py
"""
Market Dashboard Runner
ダッシュボードの実行とHTML生成
"""

from market_dashboard import MarketDashboard
from dashboard_visualizer import DashboardVisualizer


def main():
    """
    ダッシュボードを生成してHTMLに出力
    """
    print("Initializing Market Dashboard...")
    dashboard = MarketDashboard()

    print("\nCalculating market metrics...")

    # データ収集とダッシュボード生成
    exposure, performance, vix, sectors, power_law, screener_results = dashboard.generate_dashboard()

    # HTML生成
    print("\nGenerating HTML dashboard...")
    visualizer = DashboardVisualizer()
    html_content = visualizer.generate_html_dashboard(
        exposure, performance, vix, sectors, power_law, screener_results
    )

    visualizer.save_html(html_content, 'market_dashboard.html')

    print("\n✅ Dashboard generation complete!")
    print("📊 Open 'market_dashboard.html' in your browser to view the dashboard.")


if __name__ == '__main__':
    main()
