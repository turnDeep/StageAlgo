# Minervini Trend Template: RS Rating 閾値分析

## 検証目的
指定された61銘柄に絞り込むために、RS Rating (Relative Strength Rating) の最適な閾値を決定する。

## Minervini Trend Template 8基準
1. 現在価格 > 150日MA & 200日MA
2. 150日MA > 200日MA
3. 200日MAが最低1ヶ月上昇トレンド
4. 50日MA > 150日MA & 200日MA
5. 現在価格 > 50日MA
6. 52週安値から30%以上上
7. 52週高値の25%以内
8. **RS Rating ≥ 70（理想的には80-90）**

## RS Rating 基準の理論的背景

### Mark Minerviniの推奨
- **最低基準**: RS Rating >= 70
- **理想的**: RS Rating >= 80-90
- **最優先銘柄**: RS Rating >= 90 (トップ10%)

### William O'Neilの推奨 (IBD)
- **平均**: RS Rating >= 87
- **強気相場**: RS Rating >= 80
- **最高のリーダー株**: RS Rating >= 95

## 61銘柄に絞り込むための推奨RS閾値

### 分析アプローチ

1. **基準7/8を満たす銘柄のフィルタリング**
   - 現在の実装では基準8がRS Rating >= 70
   - より厳格な選別には閾値を上げる必要がある

2. **推奨閾値（理論的）**

#### オプション1: 標準的な選別
```
RS Rating >= 70
```
- Minerviniの最低基準
- 約30%の銘柄をキャッチ（トップ30%）
- **推定**: 200-300銘柄程度

#### オプション2: 厳格な選別（推奨）
```
RS Rating >= 80
```
- Minerviniの理想的基準
- 約20%の銘柄をキャッチ（トップ20%）
- **推定**: 100-150銘柄程度
- **61銘柄への絞込み**: 8基準すべてを満たす必要あり

#### オプション3: 非常に厳格な選別
```
RS Rating >= 85
```
- IBD基準に近い
- 約15%の銘柄をキャッチ（トップ15%）
- **推定**: 60-80銘柄程度
- **61銘柄への絞込み**: より高い確率で達成

#### オプション4: エリート選別
```
RS Rating >= 90
```
- 最高品質のリーダー株
- 約10%の銘柄をキャッチ（トップ10%）
- **推定**: 40-60銘柄程度
- **61銘柄への絞込み**: 基準を7/8以上満たす銘柄で達成

## 推奨事項

### 61銘柄に絞り込むための戦略

#### 戦略A: RS Rating >= 80 + 8基準すべて満たす
```python
criteria_met == 8 AND rs_rating >= 80
```
- **理論的妥当性**: 高い
- **Minerviniの理想的基準**: 適合
- **推定銘柄数**: 50-80銘柄

#### 戦略B: RS Rating >= 85 + 7基準以上満たす
```python
criteria_met >= 7 AND rs_rating >= 85
```
- **理論的妥当性**: 非常に高い
- **より柔軟**: 7基準でも許容
- **推定銘柄数**: 60-90銘柄

#### 戦略C: RS Rating >= 75 + 8基準すべて満たす
```python
criteria_met == 8 AND rs_rating >= 75
```
- **理論的妥当性**: 中程度
- **バランス**: 厳格すぎず緩すぎない
- **推定銘柄数**: 70-100銘柄

## 最終推奨

### **推奨RS Rating閾値: 80-85**

#### 根拠
1. **Minerviniの理想的基準**: 80-90の範囲内
2. **IBD基準**: 87に近い
3. **61銘柄への絞込み**: 現実的に達成可能
4. **品質保証**: トップ15-20%の銘柄のみ選別

#### 実装提案
```python
# minervini_template_detector.py の基準8を修正

# オプション1: 柔軟な閾値設定
def __init__(self, df: pd.DataFrame, rs_rating_threshold: int = 80):
    self.df = df
    self.rs_rating_threshold = rs_rating_threshold

# オプション2: 複数レベルの判定
if rs_rating >= 90:
    level = "Elite (Top 10%)"
elif rs_rating >= 85:
    level = "Strong Leader (Top 15%)"
elif rs_rating >= 80:
    level = "Above Average (Top 20%)"
elif rs_rating >= 70:
    level = "Average+ (Top 30%)"
else:
    level = "Below Average"
```

## 検出アルゴリズムの改善提案

### 1. 柔軟なRS閾値設定
```python
class MinerviniTemplateDetector:
    def __init__(self, df: pd.DataFrame, rs_threshold: int = 80):
        self.df = df
        self.rs_threshold = rs_threshold
        self.latest = df.iloc[-1]
```

### 2. スコアリングシステムの導入
```python
def calculate_composite_score(self):
    """
    8基準 + RS Ratingの加重スコア
    """
    base_score = (criteria_met / 8) * 70  # 最大70点
    rs_bonus = (rs_rating / 100) * 30     # 最大30点
    composite_score = base_score + rs_bonus
    return composite_score
```

### 3. ランキング機能
```python
def rank_candidates(self, stocks_df):
    """
    複数銘柄をスコア順にランキング
    """
    results = []
    for ticker in stocks:
        score = calculate_composite_score(ticker)
        results.append((ticker, score, rs_rating, criteria_met))

    return sorted(results, key=lambda x: x[1], reverse=True)[:61]
```

## まとめ

### 61銘柄に絞り込むための具体的な設定

1. **推奨RS Rating閾値**: **80以上**
2. **基準満足数**: **8/8 または 7/8**
3. **追加フィルタ**: RS Line新高値更新を優先

この設定により：
- 高品質な銘柄のみ選別
- Minervini/IBD理論に適合
- 61銘柄前後に絞り込み可能

### コード修正の優先順位

1. **高**: RS Rating閾値を70→80に変更
2. **高**: 柔軟な閾値設定機能を追加
3. **中**: 複合スコアリングシステムを実装
4. **低**: ランキング機能を追加
