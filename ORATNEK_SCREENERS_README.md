# Oratnek Screeners - IBD式スクリーニングシステム

## 📊 概要

Oratnek Screenersは、Investor's Business Daily (IBD) の手法に基づいた6つの強力なスクリーニングリストを提供します。
StageAlgoのMarket Dashboardに統合され、強力なモメンタム、機関投資家の蓄積、健全なチャート形状を持つ銘柄を特定します。

## 🎯 6つのスクリーニングリスト

### 1. **Momentum 97** 📈

**目的**: 短期・中期・長期すべてで上位3%に入る超強銘柄を発見

**スクリーニング基準**:
- 1ヶ月リターン: ≥97パーセンタイル (上位3%)
- 3ヶ月リターン: ≥97パーセンタイル (上位3%)
- 6ヶ月リターン: ≥97パーセンタイル (上位3%)

**使い方**:
- すべての期間で一貫して上昇している超強銘柄
- 継続的なアウトパフォーマンスを示す
- トレンドの継続性が高い

### 2. **Explosive EPS Growth** 🚀

**目的**: 爆発的な成長が予想される銘柄を早期発見

**スクリーニング基準**:
- RS Rating: ≥80 (上位20%)
- 50日平均出来高: ≥100,000株
- 価格: ≥50日移動平均線

**使い方**:
- 高成長企業の特定
- 相対的に強い銘柄
- 十分な流動性を確保

### 3. **Up on Volume** 📊

**目的**: 出来高を伴って上昇している機関投資家注目銘柄

**スクリーニング基準**:
- 当日変化率: ≥0% (上昇中)
- 出来高変化率: ≥20% (50日平均の120%以上)
- 価格: ≥$10
- 50日平均出来高: ≥100,000株
- RS Rating: ≥80
- A/D Rating: A, B, or C (機関投資家の蓄積)

**使い方**:
- 機関投資家が買い集めている可能性
- 出来高確認でブレイクアウトの信頼性向上
- 低位株を除外し品質重視

### 4. **Top 2% RS Rating** ⭐

**目的**: 相対的強さが最高クラスかつトレンドが完璧な銘柄

**スクリーニング基準**:
- RS Rating: ≥98 (上位2%)
- MA順序: 10日 > 21日 > 50日 (完璧なトレンド)
- 50日平均出来高: ≥100,000株
- 当日出来高: ≥100,000株

**使い方**:
- 最強銘柄のリスト
- トレンドが完璧に整っている
- エントリータイミングの候補

### 5. **4% Bullish Yesterday** 💥

**目的**: 昨日大きく上昇した銘柄のフォローアップ候補

**スクリーニング基準**:
- 昨日の上昇率: >4%
- 価格: ≥$1
- 相対出来高: >1.0 (平均以上)
- 寄り高から更に上昇: >0%
- 90日平均出来高: >100,000株

**使い方**:
- 急騰銘柄の早期発見
- 寄り高から更に上昇 = 強い買い圧力
- フォローアップのモニタリング

### 6. **Healthy Chart Watch List** ✅

**目的**: Mark MinerviniのSEPA基準に近い、最も厳格なスクリーニング

**スクリーニング基準**:
- 短期MA順序: 10日 > 21日 > 50日
- 長期MA順序: 50日 > 150日 > 200日 (Stage 2確認)
- RS Line新高値
- RS Rating: ≥90 (上位10%)
- A/D Rating: A or B (機関投資家の強い蓄積)
- Comp Rating: ≥80
- 50日平均出来高: ≥100,000株

**使い方**:
- 最高品質の銘柄リスト
- 長期保有候補
- Stage 2上昇トレンド確認済み

## 🔧 IBD指標の詳細

### RS Rating (Relative Strength Rating)

IBD式の相対的強さレーティング (0-100スケール)

**計算方法**:
```
RS Score = 40% × ROC(63日) + 20% × ROC(126日) + 20% × ROC(189日) + 20% × ROC(252日)
```

**解釈**:
- 90-100: 非常に強い
- 80-89: 強い
- 70-79: やや強い
- 50-69: 平均
- 50未満: 弱い

### A/D Rating (Accumulation/Distribution Rating)

機関投資家の蓄積/分散評価 (A-E)

**評価基準**:
- **A**: 非常に強い蓄積 (機関投資家が積極的に買い集めている)
- **B**: 蓄積 (買い圧力が強い)
- **C**: 中立 (買いと売りが均衡)
- **D**: 分散 (売り圧力が強い)
- **E**: 非常に強い分散 (機関投資家が売却している)

### Comp Rating (Composite Rating)

総合レーティング (0-100スケール)

**計算方法**:
```
Comp Rating = 60% × RS Rating + 40% × EPS Rating
```

## 💻 使用方法

### 基本的な使い方

```python
from market_dashboard import MarketDashboard

# ダッシュボードを初期化
dashboard = MarketDashboard()

# ダッシュボードを生成（スクリーニング結果を含む）
exposure, performance, vix, sectors, power_law, screener_results = dashboard.generate_dashboard()

# HTMLダッシュボードを生成
from dashboard_visualizer import DashboardVisualizer

visualizer = DashboardVisualizer()
html_content = visualizer.generate_html_dashboard(
    exposure, performance, vix, sectors, power_law, screener_results
)
visualizer.save_html(html_content, 'market_dashboard.html')
```

### スクリーナー単体での使用

```python
from oratnek_screeners import OratnekScreener, get_default_tickers

# 銘柄リストを取得
tickers = get_default_tickers()

# スクリーナーを初期化
screener = OratnekScreener(tickers)

# 全スクリーニングを実行
results = screener.run_all_screens()

# 個別スクリーニングの実行
momentum_97 = screener.screen_momentum_97()
explosive_eps = screener.screen_explosive_eps_growth()
up_on_volume = screener.screen_up_on_volume()
top_2_rs = screener.screen_top_2_percent_rs()
bullish_4pct = screener.screen_4_percent_bullish_yesterday()
healthy_chart = screener.screen_healthy_chart_watchlist()
```

### カスタム銘柄リストの使用

```python
from oratnek_screeners import OratnekScreener

# カスタム銘柄リスト
my_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']

# スクリーナーを初期化
screener = OratnekScreener(my_tickers)

# スクリーニング実行
results = screener.run_all_screens()

# 結果をCSVに保存
for name, df in results.items():
    if not df.empty:
        filename = f"my_screener_{name}.csv"
        df.to_csv(filename, index=False)
        print(f"Saved {name} to {filename}")
```

## 📂 ファイル構成

```
StageAlgo/
├── oratnek_screeners.py          # スクリーニングシステム本体
├── market_dashboard.py            # Market Dashboard (統合済み)
├── dashboard_visualizer.py        # HTMLダッシュボード生成
├── run_dashboard.py               # ダッシュボード実行スクリプト
├── test_oratnek_screeners.py     # テストスクリプト
└── ORATNEK_SCREENERS_README.md   # このドキュメント
```

## 🚀 実行方法

### Market Dashboardの実行

```bash
python3 run_dashboard.py
```

### スクリーナー単体の実行

```bash
python3 oratnek_screeners.py
```

### テストの実行

```bash
python3 test_oratnek_screeners.py
```

## 📊 出力例

### コンソール出力

```
================================================================================
ORATNEK SCREENER - Running All Screens
================================================================================

[Momentum 97] Screening...
  → Found 3 stocks

[Explosive EPS Growth] Screening...
  → Found 12 stocks

[Up on Volume] Screening...
  → Found 8 stocks

[Top 2% RS Rating] Screening...
  → Found 5 stocks

[4% Bullish Yesterday] Screening...
  → Found 7 stocks

[Healthy Chart Watch List] Screening...
  → Found 4 stocks

================================================================================
SCREENING SUMMARY
================================================================================
momentum_97              :   3 stocks
explosive_eps            :  12 stocks
up_on_volume             :   8 stocks
top_2_rs                 :   5 stocks
bullish_4pct             :   7 stocks
healthy_chart            :   4 stocks
================================================================================
```

### HTMLダッシュボード

`market_dashboard.html`ファイルが生成され、以下のセクションが含まれます:

1. Market Exposure
2. Market Performance Overview
3. VIX Analysis
4. Sector Performance
5. Power Law Indicators
6. **Oratnek Screeners** (新規追加)
   - Momentum 97
   - Explosive EPS Growth
   - Up on Volume
   - Top 2% RS Rating
   - 4% Bullish Yesterday
   - Healthy Chart Watch List

## 🎓 理論的背景

このスクリーニングシステムは以下の理論に基づいています:

### William O'Neil (IBD理論)
- **CANSLIM戦略**: 高成長銘柄の特定
- **RS Rating**: 相対的強さの重要性
- **機関投資家の蓄積**: A/D Rating

### Mark Minervini (SEPA基準)
- **Stage 2上昇トレンド**: 長期MA順序
- **トレンドテンプレート**: 短期MA順序
- **高品質銘柄**: Healthy Chart Watch List

### Stan Weinstein (Stage分析)
- **4つのステージ理論**: Stage 2での買い
- **Moving Average分析**: トレンド判定

## ⚙️ カスタマイズ

### スクリーニング基準の調整

`oratnek_screeners.py`の各スクリーニングメソッドで基準を調整できます:

```python
# 例: Momentum 97の基準を変更
def screen_momentum_97(self) -> pd.DataFrame:
    # パーセンタイルを95%に変更
    momentum_95 = df[
        (df['rank_1m_pct'] >= 95) &  # 97 → 95に変更
        (df['rank_3m_pct'] >= 95) &
        (df['rank_6m_pct'] >= 95)
    ]
    return momentum_95
```

### 新しいスクリーニングの追加

```python
def screen_custom(self) -> pd.DataFrame:
    """カスタムスクリーニング"""
    results = []

    for ticker in self.tickers:
        data = self._get_stock_data(ticker)
        if data is None:
            continue

        _, metrics = data

        # カスタム条件
        if (metrics['rs_rating'] >= 95 and
            metrics['price'] > metrics['sma_50'] * 1.05):
            results.append({
                'ticker': ticker,
                'rs_rating': metrics['rs_rating'],
                # ...
            })

    return pd.DataFrame(results)
```

## 🔍 注意事項

1. **データ取得**: 各銘柄のデータ取得には時間がかかります
2. **API制限**: yfinanceのレート制限に注意
3. **デフォルト銘柄**: デフォルトではS&P 100主要銘柄を使用
4. **EPS データ**: 実際のEPSデータ取得には制限があるため、RS Ratingで代用
5. **市場時間外**: 市場終了後のデータは最新の終値を使用

## 📈 推奨される使用シーン

### デイリーチェック
- **4% Bullish Yesterday**: 昨日の急騰銘柄をフォロー
- **Up on Volume**: 機関投資家の動きを確認

### ウィークリー分析
- **Top 2% RS Rating**: 最強銘柄のトレンド確認
- **Momentum 97**: 一貫した強さを持つ銘柄

### 新規ポジション検討
- **Healthy Chart Watch List**: 高品質な長期候補
- **Explosive EPS Growth**: 高成長銘柄の発掘

## 🤝 貢献

改善提案やバグ報告は歓迎します。

## 📝 ライセンス

このプロジェクトはStageAlgoプロジェクトの一部です。

---

**作成日**: 2025-11-05
**バージョン**: 1.0.0
**StageAlgo Market Dashboard統合版**
