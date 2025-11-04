# Minervini Template Detector 使用方法（改善版）

## 概要

minervini_template_detector.py を改善し、以下の機能を追加しました：

1. **柔軟なRS Rating閾値設定** - デフォルト: 80（推奨）
2. **複合スコアリングシステム** - 基準満足度70% + RSボーナス30%
3. **RS Ratingレベル別評価** - Top 5% から Top 30% まで
4. **61銘柄への絞込み最適化**

## 主な変更点

### 1. コンストラクタの変更

#### 変更前
```python
detector = MinerviniTemplateDetector(df)
```

#### 変更後
```python
# デフォルトRS閾値: 80（推奨）
detector = MinerviniTemplateDetector(df, rs_threshold=80)

# または異なる閾値を指定
detector = MinerviniTemplateDetector(df, rs_threshold=85)  # より厳格
detector = MinerviniTemplateDetector(df, rs_threshold=70)  # より緩い
```

### 2. RS閾値の選び方

| 閾値 | 説明 | 対象銘柄 | 推奨用途 |
|------|------|----------|----------|
| 70 | Minervini最低基準 | トップ30% | 広範囲スクリーニング |
| **80** | **Minervini理想的基準** | **トップ20%** | **61銘柄選定（推奨）** |
| 85 | IBD基準に近い | トップ15% | より厳格な選定 |
| 90 | エリート銘柄のみ | トップ10% | 最高品質のみ |

### 3. 出力内容の変更

#### check_template() の返り値に追加

```python
result = detector.check_template()

# 既存の項目
result['all_criteria_met']  # bool: 8基準すべて満たすか
result['score']             # float: 基準スコア（0-100）
result['criteria_met']      # int: 満たした基準数（0-8）
result['checks']            # dict: 各基準の詳細
result['interpretation']    # dict: 解釈とアクション

# 新規追加項目
result['composite_score']   # float: 複合スコア（基準70% + RS30%）
result['rs_rating']         # float: RS Rating値
result['rs_level']          # str: RSレベル（例: "Elite (Top 10%)"）
```

#### RS Levelの詳細

```python
result['rs_level'] の例:
- "Exceptional (Top 5%)"      # RS >= 95
- "Elite (Top 10%)"           # RS >= 90
- "Strong Leader (Top 15%)"   # RS >= 85
- "Above Average (Top 20%)"   # RS >= 80
- "Average+ (Top 30%)"        # RS >= 70
- "Below Average"             # RS < 70
```

## 使用例

### 例1: デフォルト設定（RS閾値80）

```python
from data_fetcher import fetch_stock_data
from indicators import calculate_all_basic_indicators
from rs_calculator import RSCalculator
from minervini_template_detector import MinerviniTemplateDetector

# データ取得
stock_df, _ = fetch_stock_data('TSLA', period='2y')
benchmark_df, _ = fetch_stock_data('SPY', period='2y')

# 指標計算
indicators_df = calculate_all_basic_indicators(stock_df)

# RS Rating計算
rs_calc = RSCalculator(indicators_df, benchmark_df)
rs_score_series = rs_calc.calculate_ibd_rs_score()
rs_rating = rs_calc.calculate_percentile_rating(rs_score_series.iloc[-1])
indicators_df['RS_Rating'] = rs_rating

# Minervini検出（デフォルトRS閾値80）
detector = MinerviniTemplateDetector(indicators_df)
result = detector.check_template()

print(f"基準満足数: {result['criteria_met']}/8")
print(f"複合スコア: {result['composite_score']:.1f}")
print(f"RS Rating: {result['rs_rating']:.1f}")
print(f"RS レベル: {result['rs_level']}")
print(f"判定: {result['interpretation']['status']}")
```

### 例2: RS閾値を変更（85）

```python
# より厳格な選定（IBD基準に近い）
detector = MinerviniTemplateDetector(indicators_df, rs_threshold=85)
result = detector.check_template()

if result['all_criteria_met']:
    print(f"✓ 8基準すべて満たす（RS >= 85）")
    print(f"  アクション: {result['interpretation']['action']}")
```

### 例3: 複数銘柄のスクリーニング

```python
from typing import List, Dict

def screen_stocks(tickers: List[str], rs_threshold: int = 80) -> List[Dict]:
    """
    複数銘柄をスクリーニング

    Args:
        tickers: 銘柄リスト
        rs_threshold: RS閾値

    Returns:
        合格銘柄のリスト（複合スコア順）
    """
    results = []

    # ベンチマーク取得
    _, benchmark_df = fetch_stock_data('SPY', period='2y')
    benchmark_df = calculate_all_basic_indicators(benchmark_df)

    for ticker in tickers:
        stock_df, _ = fetch_stock_data(ticker, period='2y')
        if stock_df is None:
            continue

        # 指標計算
        indicators_df = calculate_all_basic_indicators(stock_df)

        # RS Rating計算
        rs_calc = RSCalculator(indicators_df, benchmark_df)
        rs_score = rs_calc.calculate_ibd_rs_score().iloc[-1]
        rs_rating = rs_calc.calculate_percentile_rating(rs_score)
        indicators_df['RS_Rating'] = rs_rating

        # Minervini検出
        detector = MinerviniTemplateDetector(indicators_df, rs_threshold)
        result = detector.check_template()

        # 8基準すべて満たす銘柄のみ
        if result['all_criteria_met']:
            results.append({
                'ticker': ticker,
                'composite_score': result['composite_score'],
                'criteria_met': result['criteria_met'],
                'rs_rating': result['rs_rating'],
                'rs_level': result['rs_level']
            })

    # 複合スコア順にソート
    return sorted(results, key=lambda x: x['composite_score'], reverse=True)


# 使用例
target_tickers = [
    'TNK', 'CYTK', 'PSTG', 'WCC', 'TTMI', 'MIR', 'KRYS', 'FSLR',
    'NET', 'PLTR', 'TSLA', 'NVDA', 'AAPL', 'HOOD', 'QCOM'
]

# RS閾値80で絞込み
candidates = screen_stocks(target_tickers, rs_threshold=80)

print(f"\n合格銘柄数: {len(candidates)}")
print("\nTop 10:")
for i, stock in enumerate(candidates[:10], 1):
    print(f"{i}. {stock['ticker']}: "
          f"複合スコア={stock['composite_score']:.1f}, "
          f"RS={stock['rs_rating']:.0f} ({stock['rs_level']})")
```

### 例4: 61銘柄への絞込み

```python
# 指定された50銘柄
target_50_tickers = [
    'TNK', 'CYTK', 'PSTG', 'WCC', 'TTMI', 'MIR', 'KRYS', 'FSLR', 'SYM', 'NET',
    'SID', 'W', 'FN', 'PTGX', 'IMNM', 'MNMD', 'OSIS', 'ADPT', 'QS', 'KYMR',
    'LITE', 'CDTX', 'MDB', 'IDYA', 'PLTR', 'VSAT', 'CRDO', 'CNTA', 'ZBIO', 'NKTR',
    'NBIS', 'TGB', 'ACLX', 'WFRD', 'COGT', 'WULF', 'HUT', 'DQ', 'HOOD', 'RYTM',
    'SQM', 'TSLA', 'QCOM', 'NTGR', 'INCY', 'XPO', 'VSH', 'LASR', 'WGS', 'IVZ'
]

# 異なるRS閾値でテスト
for threshold in [70, 75, 80, 85, 90]:
    candidates = screen_stocks(target_50_tickers, rs_threshold=threshold)
    print(f"\nRS閾値 >= {threshold}: {len(candidates)}銘柄が8基準すべて満たす")

    if len(candidates) <= 61:
        print(f"  ✓ 61銘柄以下に絞込み成功！")
        break
```

## 複合スコアの計算式

```
複合スコア = 基準スコア × 0.70 + RS Rating × 0.30

例:
- 8/8基準満足、RS Rating = 90
  → 基準スコア: (8/8) × 70 = 70点
  → RSボーナス: (90/100) × 30 = 27点
  → 複合スコア: 70 + 27 = 97点

- 7/8基準満足、RS Rating = 85
  → 基準スコア: (7/8) × 70 = 61.25点
  → RSボーナス: (85/100) × 30 = 25.5点
  → 複合スコア: 61.25 + 25.5 = 86.75点
```

## 推奨ワークフロー

### ステップ1: 広範囲スクリーニング（RS >= 70）
```python
all_stocks = load_stock_list()  # 数千銘柄
candidates_70 = screen_stocks(all_stocks, rs_threshold=70)
print(f"RS>=70で{len(candidates_70)}銘柄検出")
```

### ステップ2: 厳格な選定（RS >= 80）
```python
top_tickers = [c['ticker'] for c in candidates_70]
candidates_80 = screen_stocks(top_tickers, rs_threshold=80)
print(f"RS>=80で{len(candidates_80)}銘柄検出")
```

### ステップ3: 最終選定（Top 61）
```python
# 複合スコア上位61銘柄
final_61 = candidates_80[:61]

print("\n最終選定61銘柄:")
for i, stock in enumerate(final_61, 1):
    print(f"{i}. {stock['ticker']}: "
          f"複合={stock['composite_score']:.1f}, "
          f"RS={stock['rs_rating']:.0f}")
```

## まとめ

### 61銘柄に絞り込むための推奨設定

1. **RS閾値**: **80以上**（Minervini理想的基準）
2. **基準満足数**: **8/8**（すべて満たす）
3. **ソート**: **複合スコア降順**
4. **上位選定**: **Top 61**

この設定により：
- ✓ 高品質な銘柄のみ選別
- ✓ Minervini/IBD理論に適合
- ✓ 61銘柄前後に絞込み可能
- ✓ 客観的なランキング

## 参考資料

- `RS_RATING_ANALYSIS.md`: RS閾値の理論的分析
- `minervini_template_detector.py`: 改善版実装
- `verify_minervini_detector_rs_rating.py`: 検証スクリプト
