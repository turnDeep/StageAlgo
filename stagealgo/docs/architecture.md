# StageAlgo アーキテクチャドキュメント

## アーキテクチャ概要

StageAlgoは5層のレイヤードアーキテクチャを採用しています。

```
┌─────────────────────────────────────────┐
│  Layer 5: Presentation (CLI/UI)         │  ← cli/
├─────────────────────────────────────────┤
│  Layer 4: Application (ワークフロー)    │  ← screeners/, dashboard/
├─────────────────────────────────────────┤
│  Layer 3: Domain (分析ロジック)         │  ← analysis/
├─────────────────────────────────────────┤
│  Layer 2: Core (基礎計算)               │  ← core/indicators/
├─────────────────────────────────────────┤
│  Layer 1: Infrastructure (データ取得)   │  ← core/infrastructure/
└─────────────────────────────────────────┘
```

## レイヤー詳細

### Layer 1: Infrastructure (インフラストラクチャ層)

**場所**: `stagealgo/core/infrastructure/`

**責務**: 外部データソースからのデータ取得

**モジュール**:
- `fmp_client.py`: FinancialModelingPrep API クライアント
- `data_fetcher.py`: データ取得の統合インターフェース

**依存関係**: 外部APIのみ

### Layer 2: Core Indicators (基礎指標層)

**場所**: `stagealgo/core/indicators/`

**責務**: 基本的なテクニカル指標の計算

**モジュール**:
- `basic.py`: SMA, EMA, RSI, MACD などの基本指標

**依存関係**: Layer 1 (Infrastructure)

### Layer 3: Analysis (分析エンジン層)

**場所**: `stagealgo/analysis/`

**責務**: ドメイン固有の分析ロジック

**サブモジュール**:

#### Stage Analysis (`analysis/stage/`)
- `detector.py`: Stage検出ロジック
- `history_manager.py`: Stage履歴管理

#### Strength Analysis (`analysis/strength/`)
- `rs_calculator.py`: 相対強度計算

#### Volume Analysis (`analysis/volume/`)
- `analyzer.py`: 出来高分析
- `vcp_detector.py`: VCPパターン検出
- `vwap_analyzer.py`: VWAP分析

#### Pattern Analysis (`analysis/pattern/`)
- `base_analyzer.py`: ベースパターン分析
- `minervini_template.py`: Minerviniテンプレート検出

#### Risk Analysis (`analysis/risk/`)
- `atr_analyzer.py`: ATRベースのリスク分析

**依存関係**: Layer 1, Layer 2

### Layer 4: Application (アプリケーション層)

**場所**: `stagealgo/screeners/`, `stagealgo/dashboard/`

**責務**: ビジネスワークフローの実装

**Screeners** (`screeners/`):
- `base_screener.py`: 基底スクリーナークラス（並列処理機能提供）
- `stage_screener.py`: Stage 2銘柄スクリーナー
- `base_pattern_screener.py`: ベースパターンスクリーナー
- `oratnek_screeners.py`: Oratnekスクリーナー

**Dashboard** (`dashboard/`):
- `market_analyzer.py`: マーケット分析ダッシュボード
- `breadth_analyzer.py`: マーケット幅分析
- `visualizer.py`: データ可視化

**依存関係**: Layer 1, Layer 2, Layer 3

### Layer 5: Presentation (プレゼンテーション層)

**場所**: `stagealgo/cli/`

**責務**: ユーザーインターフェース

**モジュール**:
- `run_screener.py`: スクリーナーCLI
- `run_dashboard.py`: ダッシュボードCLI

**依存関係**: 全てのレイヤー

## 依存関係ルール

1. **下位レイヤーのみに依存**: 各レイヤーは自身より下位のレイヤーのみに依存できます
2. **循環依存の禁止**: レイヤー間での循環依存は禁止
3. **インターフェース重視**: 上位レイヤーは下位レイヤーの公開インターフェースのみを使用

## モジュールインポート規則

### 推奨されるインポート方法

```python
# トップレベルからのインポート（推奨）
from stagealgo import StageDetector, fetch_stock_data

# レイヤー指定のインポート
from stagealgo.core.infrastructure import FMPDataFetcher
from stagealgo.analysis.stage import StageDetector
from stagealgo.screeners import BaseScreener

# 相対インポート（モジュール内部のみ）
from .detector import StageDetector
from ..infrastructure import fetch_stock_data
```

### 非推奨のインポート

```python
# 絶対パスの直接インポート（旧構造）
from stage_detector import StageDetector  # ❌

# レイヤーをスキップしたインポート
from stagealgo import atr_analyzer  # ❌ 明示的なパスを使用すべき
```

## 新機能の追加方法

### 1. 新しい分析エンジンの追加

```python
# stagealgo/analysis/momentum/momentum_analyzer.py
class MomentumAnalyzer:
    def __init__(self):
        pass

    def analyze(self, data):
        # 分析ロジック
        pass
```

### 2. 新しいスクリーナーの追加

```python
# stagealgo/screeners/momentum_screener.py
from .base_screener import BaseScreener
from ..analysis.momentum import MomentumAnalyzer

class MomentumScreener(BaseScreener):
    def screen_single_ticker(self, ticker: str):
        # MomentumAnalyzerを使用したスクリーニング
        pass
```

### 3. 新しいCLIコマンドの追加

```python
# stagealgo/cli/run_momentum.py
from ..screeners import MomentumScreener

def main():
    # CLIロジック
    pass
```

## テスト戦略

### ユニットテスト

各レイヤーは独立してテスト可能：

```python
# tests/unit/test_stage_detector.py
from stagealgo.analysis.stage import StageDetector

def test_stage_detection():
    detector = StageDetector()
    # テストロジック
```

### 統合テスト

複数レイヤーにまたがるテスト：

```python
# tests/integration/test_full_workflow.py
from stagealgo import fetch_stock_data, StageDetector

def test_full_screening_workflow():
    # 完全なワークフローのテスト
```

## パフォーマンス最適化

### 並列処理

`BaseScreener`クラスが自動的に並列処理を提供：

```python
screener = MyScreener(tickers, workers=8)
results = screener.run()
```

### キャッシング

FMPDataFetcherは自動的にデータをキャッシュします。

## マイグレーションガイド

### 旧コードの更新方法

```python
# Before (旧構造)
from data_fetcher import fetch_stock_data
from stage_detector import StageDetector

# After (新構造)
from stagealgo import fetch_stock_data, StageDetector
# または
from stagealgo.core.infrastructure import fetch_stock_data
from stagealgo.analysis.stage import StageDetector
```

## 今後の拡張計画

1. **Layer 1 拡張**: 複数のデータプロバイダー対応
2. **Layer 3 拡張**: 機械学習ベースの分析エンジン
3. **Layer 5 拡張**: Web UI / REST API の追加
4. **キャッシング層**: Redis/Memcached統合
5. **モニタリング**: ロギングとメトリクス収集

## まとめ

このアーキテクチャにより：
- ✅ **保守性**: 変更の影響範囲が明確
- ✅ **拡張性**: 新機能追加が容易
- ✅ **テスト性**: レイヤーごとの独立テスト
- ✅ **可読性**: コードの配置が予測可能
- ✅ **スケーラビリティ**: チーム開発に対応
