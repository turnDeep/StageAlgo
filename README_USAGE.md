# 米国株スクリーニングシステム - 使い方

README.mdに記載された包括的な理論体系を完全実装した統合スクリーニングシステムです。

## 📋 システム構成

### コアモジュール

1. **data_fetcher.py** - データ取得（yfinance + curl-cffi）
2. **indicators.py** - 基礎指標計算（MA、ATR、OBV等）
3. **stage_detector.py** - ステージ判定（Minerviniテンプレート8基準）
4. **base_analyzer.py** - ベースパターン分析（O'Neil理論）
5. **vcp_detector.py** - VCPパターン検出（Minervini理論）
6. **rs_calculator.py** - RS Rating計算（IBD式）
7. **atr_analyzer.py** - ATR Multiple分析
8. **volume_analyzer.py** - 出来高分析（Wyckoff理論）
9. **vwap_analyzer.py** - 高値固定VWAP分析
10. **scoring_system.py** - 総合スコアリングシステム
11. **screener.py** - メインスクリーナー

### ユーティリティ

- **get_tickers.py** - NASDAQ/NYSEティッカー取得
- **Dockerfile**, **docker-compose.yml** - Docker環境

## 🚀 クイックスタート

### 1. 環境構築

```bash
# Pythonパッケージのインストール
pip install -r requirements.txt

# または Docker使用
docker-compose up -d
docker-compose exec app bash
```

### 2. ティッカーリスト取得

```bash
python get_tickers.py
```

これにより `stock.csv` が生成されます（NASDAQ/NYSE全銘柄）。

### 3. スクリーニング実行

```bash
# 並列処理で実行（推奨）
python screener.py

# 逐次処理で実行
python screener.py --sequential

# ワーカー数指定
python screener.py --workers 4
```

### 4. 結果確認

- **stage1or2.csv** - 詳細な分析結果
- **stage1or2_tradingview.txt** - TradingView用リスト

## 📊 出力される情報

### stage1or2.csv の列

| 列名 | 説明 |
|------|------|
| Ticker | ティッカーシンボル |
| Exchange | 取引所（NASDAQ/NYSE） |
| Current Stage | 現在のステージ（1または2） |
| Substage | サブステージ（2A, 2B等） |
| Stage Start Date | ステージ開始推定日 |
| Score | 総合スコア（100点満点） |
| Grade | 評価（A+, A, B, C） |
| Overheat | ATR Multiple（過熱度） |
| RS Rating | 相対強度レーティング |
| Action | 推奨アクション |
| Base Score | ベース品質スコア |
| VCP Detected | VCP検出フラグ |
| Volume Score | 出来高スコア |
| Wyckoff Phase | Wyckoffフェーズ |

### スコアリング基準

#### Stage 1候補（ベース形成中）

**配点（100点満点）:**
- ベース品質: 25点
- VCP完成度: 25点
- RS Rating: 25点
- 出来高: 15点
- ATR位置: 10点

**判定基準:**
- 90点以上: A+ 最優先監視
- 80-89点: A 優先監視
- 70-79点: B 監視継続
- 70点未満: C 低優先度

#### Stage 2候補（上昇トレンド中）

**配点（100点満点）:**
- トレンド強度: 20点（Minerviniテンプレート）
- ベース品質: 20点（過去ベース+カウント）
- RS Rating: 20点
- 出来高: 20点
- ATR位置: 10点
- MA配列: 10点

**判定基準:**
- 90点以上: A+ 即座に買い
- 85-89点: A 押し目待ち
- 75-84点: B 監視継続
- 75点未満: C 見送り

## 🎯 理論的背景

このシステムは以下の理論家の手法を統合しています:

### 1. Stan Weinstein - ステージ分析
- 4ステージサイクル理論
- 移動平均線の配置による判定

### 2. Mark Minervini - トレンドテンプレート
- **8つの基準すべて満たす必要**
  1. 価格 > 150日MA & 200日MA
  2. 150日MA > 200日MA
  3. 200日MA上昇トレンド
  4. 50日MA > 150日MA & 200日MA
  5. 価格 > 50日MA
  6. 52週安値から30%以上上
  7. 52週高値の25%以内
  8. RS Rating ≥ 70

### 3. William O'Neil - ベース理論
- Cup with Handle
- 7-8週間の最低期間
- ベースカウンティング（1-2番目推奨）

### 4. VCP理論（Minervini）
- **段階的収縮**: 各修正が前回の30-75%程度
- 出来高Dry Up
- より高い安値（テニスボールアクション）

### 5. Richard Wyckoff - 出来高理論
- Phase A-E蓄積サイクル
- Selling/Buying Climax
- Pocket Pivot検出

### 6. IBD - RS Rating
- 加重ROC: 40%×3ヶ月 + 20%×6ヶ月 + 20%×9ヶ月 + 20%×12ヶ月
- パーセンタイルランク（1-99）

### 7. ATR Multiple from MA
- **利確の重要閾値**: 7-10倍
- 過熱感の定量化
- 動的ストップロス

### 8. 高値固定VWAP
- 機関投資家の平均コスト
- 供給圧力の測定
- IPO株への特別適用

## 📈 実践的な使い方

### ワークフロー

1. **週末にスクリーニング実行**
   ```bash
   python screener.py
   ```

2. **Stage 2銘柄（優先度1）を確認**
   - スコア90点以上
   - RS Rating 85以上
   - ATR Multiple 0-5倍
   - Minerviniテンプレート8基準すべて満たす

3. **Stage 1銘柄（優先度2-3）を監視**
   - スコア80点以上
   - ブレイクアウト待ち
   - VCP完成間近

4. **TradingViewで視覚的確認**
   - stage1or2_tradingview.txt をインポート
   - チャートパターンを目視確認

### 買いシグナルチェックリスト

Stage 1→2転換時（12項目中10項目以上）:

- [ ] Minerviniテンプレート8基準すべて満たす
- [ ] ベース期間7週以上
- [ ] ベース深さ15-35%
- [ ] ベースカウント1-2番目
- [ ] VCP: 段階的縮小確認
- [ ] 出来高Dry Up（平均の30-50%以下）
- [ ] BO時出来高2-3倍
- [ ] RS Rating ≥ 85
- [ ] ATR Multiple -0.5〜+1.0
- [ ] 高値固定VWAP上抜け
- [ ] S&P500がStage 2
- [ ] セクター強い

### 利確タイミング

**ATR Multiple基準:**
- 5.0倍: 初期利確20-25%
- 7.0倍: 第1利確25-30%（**重要閾値**）
- 10.0倍: 第2利確30-35%
- トレーリングストップで残り

## 🔍 各モジュールの個別テスト

```bash
# 指標計算テスト
python indicators.py

# ステージ検出テスト
python stage_detector.py

# VCP検出テスト
python vcp_detector.py

# ベース分析テスト
python base_analyzer.py

# RS Rating計算テスト
python rs_calculator.py

# ATR分析テスト
python atr_analyzer.py

# 出来高分析テスト
python volume_analyzer.py

# VWAP分析テスト
python vwap_analyzer.py

# 総合スコアリングテスト
python scoring_system.py
```

## ⚠️ 重要な注意事項

### システムの制約

1. **データ品質依存**
   - yfinanceのデータ精度に依存
   - curl-cffiでブロック回避

2. **計算負荷**
   - 全銘柄分析は時間がかかる
   - 並列処理推奨（デフォルト）

3. **ベースカウントの簡易実装**
   - 現在は簡易版
   - より正確な実装は今後の改善点

### 理論の適用

1. **市場環境の考慮**
   - 強気市場: ATR閾値を1.3倍に調整
   - 弱気市場: ATR閾値を0.7倍に調整

2. **質が量を圧倒**
   - 週に1-3銘柄見つかれば十分
   - スコア90点以上に集中

3. **厳格な基準の重要性**
   - Minerviniテンプレート: 妥協なし
   - VCP段階的縮小: 柔軟だが品質重視

## 📚 参考資料

詳細な理論体系は **README.md** を参照してください。

## 🤝 貢献

改善提案やバグ報告は歓迎します。

## 📄 ライセンス

個人利用・研究目的での使用を想定。
