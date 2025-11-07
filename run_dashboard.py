# run_dashboard.py
"""
Market Dashboard Runner
ダッシュボードの実行とJSON/PNG生成
"""

import json
from dotenv import load_dotenv
from market_dashboard import MarketDashboard
from dashboard_image_generator import generate_dashboard_image_from_json

# Load environment variables from .env file
load_dotenv()


def main():
    """
    ダッシュボードを生成してJSON/PNGに出力
    """
    print("Initializing Market Dashboard...")
    dashboard = MarketDashboard()

    print("\nCalculating market metrics...")

    # データ収集とダッシュボード生成
    exposure, market_performance, sectors_performance, macro_performance, screener_results = dashboard.generate_dashboard()

    # JSONデータの準備
    dashboard_data = {
        'generated_at': dashboard.current_date.strftime('%Y-%m-%d %H:%M:%S'),
        'exposure': exposure,
        'market_performance': {
            'data': market_performance.to_dict('records') if not market_performance.empty else []
        },
        'sectors_performance': {
            'data': sectors_performance.to_dict('records') if not sectors_performance.empty else []
        },
        'macro_performance': {
            'data': macro_performance.to_dict('records') if not macro_performance.empty else []
        },
        'screener_results': {}
    }

    # スクリーナー結果を追加
    if screener_results:
        for name, df in screener_results.items():
            dashboard_data['screener_results'][name] = {
                'data': df.to_dict('records') if not df.empty else []
            }

    # JSONファイルに保存
    json_output = 'market_dashboard_data.json'
    print(f"\nSaving data to JSON: {json_output}")
    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump(dashboard_data, f, indent=2, ensure_ascii=False, default=str)
    print(f"✓ JSON data saved to: {json_output}")

    # PNG画像を生成
    print("\nGenerating PNG dashboard image...")
    png_output = 'market_dashboard.png'
    generate_dashboard_image_from_json(json_output, png_output)

    print("\n✅ Dashboard generation complete!")
    print(f"📊 JSON data: {json_output}")
    print(f"🖼️  PNG image: {png_output}")


if __name__ == '__main__':
    main()
