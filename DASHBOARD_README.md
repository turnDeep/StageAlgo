# Market Dashboard - マーケットダッシュボード

StageAlgoプロジェクトのマーケットダッシュボード機能です。市場の全体的な状況を分析し、HTMLダッシュボードとして出力します。

## 📊 機能

### 1. Market Exposure (市場エクスポージャー)
- 主要指数（SPY, QQQ, IWM）のステージ分析
- VIXレベルによる調整
- Market Breadth指標による調整
- エクスポージャースコア: -60 (Extreme Bearish) ~ 100 (Bullish)

### 2. Market Performance Overview (市場パフォーマンス)
- 主要指数: S&P 500, Nasdaq 100, Russell 2000, Dow Jones
- パフォーマンス指標:
  - YTD (年初来)
  - 1週間
  - 1ヶ月
  - 1年
  - 52週高値からの距離

### 3. VIX Analysis (VIX分析)
- 現在のVIXレベル
- VIXの解釈 (Very Low ~ Extreme)
- 52週高値/安値

### 4. Sector Performance (セクターパフォーマンス)
- 11セクターETFの分析
  - XLK (Technology)
  - XLF (Financials)
  - XLV (Healthcare)
  - XLE (Energy)
  - XLI (Industrials)
  - XLY (Consumer Discretionary)
  - XLP (Consumer Staples)
  - XLB (Materials)
  - XLU (Utilities)
  - XLRE (Real Estate)
  - XLC (Communication Services)
- RS Rating計算
- Relative Strength計算

### 5. Power Law Indicators
- 5日間50MA以上の銘柄割合
- 50MA > 150MA の銘柄割合
- 150MA > 200MA の銘柄割合

## 🚀 使用方法

### インストール

```bash
# 必要なパッケージをインストール
pip install yfinance curl-cffi pandas numpy
```

### 基本的な実行

```bash
# ダッシュボードを生成
python3 run_dashboard.py
```

実行後、以下のファイルが生成されます:
- `market_dashboard.html` - HTMLダッシュボード

### プログラムから使用

```python
from market_dashboard import MarketDashboard
from dashboard_visualizer import DashboardVisualizer

# ダッシュボードの初期化
dashboard = MarketDashboard()

# データ収集とダッシュボード生成
exposure, performance, vix, sectors, power_law = dashboard.generate_dashboard()

# HTMLダッシュボードの生成
visualizer = DashboardVisualizer()
html = visualizer.generate_html_dashboard(
    exposure, performance, vix, sectors, power_law
)
visualizer.save_html(html, 'market_dashboard.html')
```

## 📁 ファイル構成

```
StageAlgo/
├── market_dashboard.py          # メインダッシュボードクラス
├── market_breadth_analyzer.py   # 市場幅指標分析
├── dashboard_visualizer.py      # HTMLダッシュボード生成
├── run_dashboard.py             # 実行スクリプト
├── DASHBOARD_README.md          # このファイル
└── DASHBOARD_VALIDATION_REPORT.md  # 検証レポート
```

## 🔧 設定

### カスタムティッカーの追加

`market_dashboard.py`の`__init__`メソッドで指数やセクターを変更できます:

```python
# 主要指数のティッカー
self.major_indices = {
    'SPY': 'S&P 500',
    'QQQ': 'Nasdaq 100',
    'IWM': 'Russell 2000',
    'DIA': 'Dow Jones',
    # カスタム指数を追加
    'EEM': 'Emerging Markets',
}

# セクターETF
self.sectors = {
    'XLK': 'Technology',
    # ...
}
```

### Power Law銘柄の変更

`run_dashboard.py`または`market_dashboard.py`の`generate_dashboard`メソッドで、分析する銘柄を変更できます:

```python
sample_tickers = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
    'TSLA', 'META', 'NFLX',
    # カスタム銘柄を追加
    'AMD', 'INTC', 'CSCO'
]
power_law = dashboard.calculate_power_law_indicators(sample_tickers)
```

## 📊 出力例

### コンソール出力

```
================================================================================
MARKET DASHBOARD
Generated: 2025-11-04 10:30:00
================================================================================

### MARKET EXPOSURE ###
Score: 45.0%
Level: Neutral
VIX: 18.5
Stage Weights: {'SPY': 2, 'QQQ': 2, 'IWM': 1}

### MARKET PERFORMANCE OVERVIEW ###
Index          Ticker  YTD %  1W %  1M %  1Y %  From 52W High %  Current Price
S&P 500        SPY     12.50  2.30  5.10  18.20  -3.20            450.25
Nasdaq 100     QQQ     15.20  3.10  6.50  22.30  -2.50            380.50
...

### VIX ANALYSIS ###
Current VIX: 18.50
Interpretation: Low - Stable Market
52W High: 35.20
52W Low: 12.10

### SECTOR PERFORMANCE ###
Sector         Ticker  Price    1D %   Relative Strength  RS Rating
Technology     XLK     150.25   1.20   5.30               85.0
...

### POWER LAW INDICATORS ###
5 Days Above 50MA: 65.0%
50MA Above 150MA: 55.0%
150MA Above 200MA: 45.0%
Total stocks analyzed: 8

================================================================================
Dashboard generation complete!
================================================================================
```

### HTMLダッシュボード

`market_dashboard.html`をブラウザで開くと、以下のような見やすいダッシュボードが表示されます:

- 📊 Market Exposureゲージ
- 📈 Market Performanceテーブル
- 📉 VIX Analysisカード
- 🏭 Sector Performanceテーブル
- 📊 Power Law Indicatorsカード

## ⚙️ 技術仕様

### 依存関係

- **yfinance**: 株価データ取得
- **curl-cffi**: HTTPリクエスト
- **pandas**: データ処理
- **numpy**: 数値計算

### StageAlgoモジュール

- **data_fetcher**: 株価データ取得
- **indicators**: テクニカル指標計算
- **rs_calculator**: RS Rating計算
- **stage_detector**: ステージ判定

### 計算ロジック

#### Market Exposure Score

```python
exposure_score = 0

# 主要指数のステージスコア
# Stage 2 = +30%, Stage 1 = +10%, Stage 3 = -10%, Stage 4 = -30%
for ticker in ['SPY', 'QQQ', 'IWM']:
    stage = detect_stage(ticker)
    if stage == 2: exposure_score += 30
    elif stage == 1: exposure_score += 10
    elif stage == 3: exposure_score -= 10
    elif stage == 4: exposure_score -= 30

# VIX調整
if vix < 15: exposure_score += 10
elif vix > 30: exposure_score -= 20

# Market Breadth調整
if ad_ratio > 1.5: exposure_score += 10
elif ad_ratio < 0.67: exposure_score -= 10

# スコアを-60〜100に正規化
exposure_score = max(-60, min(100, exposure_score))
```

#### Market Level

- **Bullish**: 80-100
- **Positive**: 60-80
- **Neutral**: 20-60
- **Negative**: -20-20
- **Bearish**: -60-(-20)
- **Extreme Bearish**: < -60

## 🐛 トラブルシューティング

### データ取得エラー

```
Error fetching SPY: ...
```

**解決策**:
- インターネット接続を確認
- yfinanceのバージョンを更新: `pip install --upgrade yfinance`
- 時間を置いて再実行

### モジュールインポートエラー

```
ModuleNotFoundError: No module named 'yfinance'
```

**解決策**:
- 依存関係をインストール: `pip install yfinance curl-cffi pandas numpy`

### VIXデータが取得できない

```
VIX data not available
```

**解決策**:
- VIXティッカー (^VIX) がyfinanceで取得できない場合があります
- 代替として、VIX先物やVIX ETF (VXX) を使用できます

## 📈 今後の改善予定

1. **リアルタイム更新**
   - WebSocketによるリアルタイムデータ
   - 自動更新機能

2. **詳細なMarket Breadth**
   - 実際の上昇株/下落株データ
   - New Highs/New Lows
   - McClellan Oscillator

3. **チャート機能**
   - matplotlib/plotlyによるチャート生成
   - インタラクティブなチャート

4. **アラート機能**
   - 市場状況の変化を検知
   - メール/Slack通知

5. **バックテスト**
   - 過去のMarket Exposureとパフォーマンスの相関分析

## 📝 ライセンス

StageAlgoプロジェクトのライセンスに準じます。

## 🤝 貢献

バグ報告や機能リクエストは、GitHubのIssuesで受け付けています。

---

**作成者**: Claude Code
**作成日**: 2025-11-04
**バージョン**: 1.0.0
