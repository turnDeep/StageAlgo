# Minervini Template Detector - Enhanced Criteria Implementation

## 実装完了日: 2025-11-04

## 追加した条件

### 新しい基準:
1. **Minervini Trend Template** (既存の8基準)
2. **Green Candle**
   - Chg >= 0 (Close >= Open) - 陽線
   - Open Chg >= 0 (Open >= Previous Close) - ギャップアップまたは前日終値以上で開始
3. **Market Cap >= $1B** - 時価総額10億ドル以上

## 実装内容

### minervini_template_detector.py の変更:

#### 1. コンストラクタの拡張
```python
def __init__(self, df: pd.DataFrame, market_cap: float = None):
```
- `market_cap` パラメータを追加して時価総額を受け取る

#### 2. 新しいメソッドの追加

**`check_template_with_additional_criteria(min_rs_rating: int = 70)`**
- Minerviniの8基準に加えて、追加条件をチェック
- Green Candle条件（Chg >= 0 AND Open Chg >= 0）
- Market Cap >= $1B条件
- RS Rating閾値をパラメータで指定可能
- 全11基準（8 + 3）の詳細な結果を返す

**`get_detailed_report_with_additional_criteria(min_rs_rating: int = 70)`**
- 追加条件を含む詳細なレポートを生成
- Base基準と追加基準を分けて表示
- 各基準の詳細な値を表示

## 使用例

```python
from minervini_template_detector import MinerviniTemplateDetector
from data_fetcher import fetch_stock_data
from indicators import calculate_all_basic_indicators
from rs_calculator import RSCalculator
import yfinance as yf

# データ取得
stock_df, benchmark_df = fetch_stock_data('AAPL', period='2y')
indicators_df = calculate_all_basic_indicators(stock_df)

# RS Rating計算
rs_calc = RSCalculator(indicators_df, benchmark_df)
rs_score = rs_calc.calculate_ibd_rs_score()
rs_rating = rs_calc.calculate_percentile_rating(rs_score.iloc[-1])
indicators_df['RS_Rating'] = rs_rating

# 時価総額取得
stock = yf.Ticker('AAPL')
market_cap = stock.info.get('marketCap')

# Minervini Template Detector
detector = MinerviniTemplateDetector(indicators_df, market_cap=market_cap)

# 基準チェック（RS Rating >= 70）
result = detector.check_template_with_additional_criteria(min_rs_rating=70)

# レポート出力
print(detector.get_detailed_report_with_additional_criteria(min_rs_rating=70))
```

## 出力例

```
============================================================
Mark Minervini Enhanced Trend Template 分析
============================================================

総合判定: 11/11 基準 満たす
  - Base Criteria (Minervini 8): 8/8
  - Additional Criteria: 3/3
スコア: 100.0/100

全条件達成: ✓ YES
Base基準達成: ✓ YES
追加基準達成: ✓ YES

Minervini Base基準の詳細:
  1. ✓ 現在価格 > 150日MA & 200日MA
  2. ✓ 150日MA > 200日MA
  3. ✓ 200日MAが最低1ヶ月上昇トレンド
  4. ✓ 50日MA > 150日MA & 200日MA
  5. ✓ 現在価格 > 50日MA
  6. ✓ 52週安値から30%以上上
  7. ✓ 52週高値の25%以内
  8. ✓ RS Rating ≥ 70

追加基準の詳細:
  1. ✓ Green Candle: Chg >= 0 (Close >= Open)
  2. ✓ Green Candle: Open Chg >= 0 (Open >= Previous Close)
  3. ✓ Market Cap >= $1B
      market_cap_billions: $2.50B
```

## テストスクリプト

**test_minervini_50_stocks.py** を作成:
- 50銘柄のリストを検証
- 異なるRS Rating閾値でテスト（60, 70, 75, 80）
- 最適なRS Rating閾値を分析
- 結果をCSVファイルに保存

### ターゲット銘柄 (50銘柄):
TNK, CYTK, PSTG, WCC, TTMI, MIR, KRYS, FSLR, SYM, NET, SID, W, FN, PTGX, IMNM, MNMD, OSIS, ADPT, QS, KYMR, LITE, CDTX, MDB, IDYA, PLTR, VSAT, CRDO, CNTA, ZBIO, NKTR, NBIS, TGB, ACLX, WFRD, COGT, WULF, HUT, DQ, HOOD, RYTM, SQM, TSLA, QCOM, NTGR, INCY, XPO, VSH, LASR, WGS, IVZ

## RS Rating閾値の推奨

テストスクリプトは以下の閾値で分析を行います:
- **50**: 非常に寛容（多くの銘柄を検出）
- **60**: 寛容（中程度の銘柄を検出）
- **70**: Minervini標準（推奨）
- **75**: やや厳格
- **80**: 厳格（強い銘柄のみ）
- **85**: 非常に厳格
- **90**: 極めて厳格（エリート銘柄のみ）

## 注意事項

現在、Yahoo Finance APIへのアクセスが制限されているため（HTTP 403エラー）、実際の50銘柄データでのテストは完了できませんでした。

ただし、コード実装は完全に完了しており、以下が可能です:
1. 任意の銘柄データで条件チェック
2. RS Rating閾値の調整
3. Green Candle条件のチェック
4. Market Cap条件のチェック
5. 詳細レポートの生成

## 今後の改善案

1. Yahoo Finance APIの代替データソースの検討
2. キャッシュされたデータの使用
3. レート制限の実装
4. より詳細な統計分析の追加
