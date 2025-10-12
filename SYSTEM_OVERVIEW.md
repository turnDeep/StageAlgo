# 米国株スクリーニングシステム - システム概要

## 🎯 システムの目的

README.mdに記載された包括的な理論体系（Stan Weinstein、Mark Minervini、William O'Neil、Richard Wyckoff等）を完全実装し、計算資源を最大限活用してStage 1/2の優良銘柄を自動抽出するシステム。

## 📦 ファイル構成（14ファイル）

### コアモジュール（11個）

1. **data_fetcher.py** (2.7KB)
   - yfinance + curl-cffiでデータ取得
   - 403エラー回避

2. **indicators.py** (5.9KB)
   - 基礎指標計算（MA、ATR、OBV、52週高値/安値）
   - フェーズ1: 基礎データ計算

3. **stage_detector.py** (12KB)
   - Minerviniトレンドテンプレート8基準
   - ステージ判定とサブステージ分類
   - フェーズ2: ステージ判定

4. **base_analyzer.py** (14KB)
   - O'Neilベース理論
   - 7種類のベースパターン識別
   - ベース品質100点スコアリング
   - フェーズ3: ベース分析

5. **vcp_detector.py** (11KB)
   - MinerviniのVCP理論
   - 段階的収縮パターン検出
   - より高い安値チェック
   - フェーズ4: VCP検出

6. **rs_calculator.py** (7.5KB)
   - IBD式RS Rating計算
   - 加重ROC（40%×3M + 20%×6M + 20%×9M + 20%×12M）
   - パーセンタイルランク変換
   - フェーズ6: RS Rating計算

7. **atr_analyzer.py** (11KB)
   - ATR Multiple from MA理論
   - 7-10倍で利確開始の重要閾値
   - 動的ストップロス計算
   - 市場環境別調整

8. **volume_analyzer.py** (14KB)
   - Wyckoff理論（Phase A-E）
   - Pocket Pivot検出
   - Climax Volume検出
   - Up/Down出来高比率
   - フェーズ5: 出来高分析

9. **vwap_analyzer.py** (9.4KB)
   - 高値固定VWAP理論
   - システムアクティブ化条件
   - 機関投資家の平均コスト測定
   - フェーズ7: 高値固定VWAP計算

10. **scoring_system.py** (13KB)
    - Stage別統合スコアリング
    - Stage 1: ベース品質重視（100点）
    - Stage 2: トレンド強度重視（100点）
    - フェーズ8: 総合スコアリング

11. **screener.py** (11KB)
    - メインスクリーナー
    - 並列処理対応
    - CSV/TXT出力
    - フェーズ9: 全銘柄スクリーニング

### ユーティリティ（3個）

12. **get_tickers.py** (3.8KB)
    - NASDAQ/NYSEティッカー取得
    - フィルタリング（5文字除外、接尾辞除外）
    - stock.csv生成

13. **requirements.txt** (53B)
    - 依存パッケージリスト

14. **README_USAGE.md** (7.3KB)
    - 詳細な使い方マニュアル

## 🔄 実行フロー

```
1. get_tickers.py
   ↓ stock.csv生成
2. screener.py（並列処理）
   ├→ data_fetcher.py（各銘柄）
   ├→ indicators.py
   ├→ stage_detector.py
   ├→ base_analyzer.py
   ├→ vcp_detector.py
   ├→ rs_calculator.py
   ├→ atr_analyzer.py
   ├→ volume_analyzer.py
   ├→ vwap_analyzer.py
   └→ scoring_system.py
   ↓
3. stage1or2.csv + stage1or2_tradingview.txt
```

## 📊 理論統合マトリクス

| 理論 | モジュール | スコア配分 | 重要指標 |
|------|----------|----------|---------|
| Weinstein ステージ | stage_detector | - | 4ステージ |
| Minervini テンプレート | stage_detector | 20点 | 8基準 |
| O'Neil ベース | base_analyzer | 25点 | 7-8週 |
| Minervini VCP | vcp_detector | 25点 | 段階的縮小 |
| IBD RS Rating | rs_calculator | 20-25点 | ≥85推奨 |
| ATR Multiple | atr_analyzer | 10点 | 7-10倍閾値 |
| Wyckoff 出来高 | volume_analyzer | 15-20点 | Phase D-E |
| VWAP | vwap_analyzer | 参考 | 上抜け |

## 🎯 スコアリング基準

### Stage 1候補（100点満点）

```
ベース品質    25点  ← base_analyzer
VCP完成度     25点  ← vcp_detector
RS Rating     25点  ← rs_calculator
出来高        15点  ← volume_analyzer
ATR位置       10点  ← atr_analyzer
─────────────────
合計         100点

90点以上: A+ 最優先監視
80-89点:  A  優先監視
70-79点:  B  監視継続
```

### Stage 2候補（100点満点）

```
トレンド強度  20点  ← stage_detector (Minerviniテンプレート)
ベース品質    20点  ← base_analyzer + カウント
RS Rating     20点  ← rs_calculator
出来高        20点  ← volume_analyzer
ATR位置       10点  ← atr_analyzer
MA配列        10点  ← indicators
─────────────────
合計         100点

90点以上: A+ 即座に買い
85-89点:  A  押し目待ち
75-84点:  B  監視継続
```

## 💡 重要な特徴

### 1. 厳格な基準適用

- **Minerviniテンプレート**: 8基準すべて満たす必要
- **VCP**: 段階的縮小を柔軟に評価（30-75%程度）
- **ベースカウント**: 1-2番目推奨、4番目以降非推奨

### 2. 定量的閾値

- **ATR Multiple 7-10倍**: 利確開始の重要閾値
- **RS Rating ≥ 85**: 最強銘柄の基準
- **出来高Dry Up < 30-50%**: 供給枯渇の確認
- **52週高値の25%以内**: Minervini基準7

### 3. 並列処理による高速化

- デフォルトで並列処理
- CPU コア数-1のワーカー
- 全銘柄を効率的に分析

## 🚀 クイックスタート

```bash
# 1. ティッカー取得
python get_tickers.py

# 2. スクリーニング実行
python screener.py

# 3. 結果確認
cat stage1or2.csv
cat stage1or2_tradingview.txt
```

## 📈 期待される成果

- **週に1-3銘柄**: 高品質な候補
- **スコア90点以上**: 即座にアクション推奨
- **質が量を圧倒**: 厳格な基準で優良銘柄のみ抽出

## 🔬 各モジュールの独立性

すべてのモジュールは独立してテスト可能:

```bash
python indicators.py        # 指標計算テスト
python stage_detector.py    # ステージ判定テスト
python vcp_detector.py      # VCP検出テスト
python base_analyzer.py     # ベース分析テスト
python rs_calculator.py     # RS Rating計算テスト
python atr_analyzer.py      # ATR分析テスト
python volume_analyzer.py   # 出来高分析テスト
python vwap_analyzer.py     # VWAP分析テスト
python scoring_system.py    # 統合スコアリングテスト
```

## 📚 理論的背景

詳細は **README.md**（包括的な理論体系）を参照。

- Stan Weinstein - ステージ分析
- Mark Minervini - トレンドテンプレート & VCP
- William O'Neil - ベース理論 & CANSLIM
- Richard Wyckoff - 供給と需要の法則
- IBD - RS Rating
- Brian Shannon - 高値固定VWAP

## ⚙️ 技術スタック

- **Python 3.12**
- **pandas** - データ処理
- **numpy** - 数値計算
- **scipy** - 信号処理（VCP検出）
- **yfinance + curl-cffi** - データ取得
- **pandas-ta** - テクニカル指標（補助）
- **tqdm** - プログレスバー
- **multiprocessing** - 並列処理

## 🎓 教育的価値

このシステムは、以下を学ぶ教材としても有用:

1. **テクニカル分析の実践**
2. **複数理論の統合手法**
3. **大規模データ処理**
4. **並列処理の実装**
5. **モジュラー設計**

## 📝 今後の改善点

1. ベースカウントのより正確な実装
2. リアルタイム監視機能
3. バックテスト機能
4. GUIインターフェース
5. データベース統合

---

**総ファイルサイズ**: 約122KB（圧縮前）
**総行数**: 約3,500行
**実装期間**: README.mdの理論体系を完全実装

このシステムは計算資源を最大限活用し、7つの理論を完全統合した時系列分析システムです。
