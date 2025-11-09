# 株式ステージ分析ツール

## 概要

このプロジェクトは、Stan Weinstein（スタン・ワインスタイン）の市場ステージ分析理論に基づいた株式分析ツールです。指定された銘柄リスト（`stock.csv`）を読み込み、各銘柄が現在4つのステージ（1: 底固め、2: 上昇、3: 天井圏、4: 下降）のいずれにあるかを判定します。

さらに、ステージ間の移行の強さを客観的に評価するためのスコアリングシステムを実装しており、規律ある投資判断をサポートします。

## 主な機能

-   **ステージ分析**: 50日移動平均線の傾きや過去の価格帯に基づき、現在の市場ステージを自動で判定します。
-   **移行スコアリング**: 各ステージの移行（例: ステージ1→2）の可能性と質を0〜100のスコアで評価し、具体的なアクションプランを提示します。
-   **複数銘柄の一括分析**: `stock.csv` ファイルに記載されたティッカーシンボルをすべて読み込み、一括で分析を実行します。
-   **堅牢なデータ取得**: Financial Modeling Prep（FMP）APIを使用して、安定的に株価データを取得します。
-   **高度なテクニカル指標**: トレンドの強さを正規化して比較可能にする「MA傾き」や、市場全体に対する個別株の強さを示す「RS Rating」など、カスタム指標を計算します。
-   **過去データ分析**: `historical_analyzer.py` を使用して、特定の銘柄の過去のステージ移行履歴を分析することも可能です。

## インストール方法

### パッケージとしてインストール（推奨）

```bash
# リポジトリをクローン
git clone https://github.com/yourusername/stagealgo.git
cd StageAlgo

# パッケージをインストール
pip install -e .

# 環境変数の設定
export FMP_API_KEY='your_fmp_api_key_here'
```

### 開発環境のセットアップ

#### オプション1: Dev Container を使用（推奨）

VS Code の Dev Container 機能を使用すると、一貫した開発環境を簡単に構築できます。

**必要なもの:**
- [Visual Studio Code](https://code.visualstudio.com/)
- [Docker Desktop](https://www.docker.com/products/docker-desktop)
- VS Code 拡張機能: [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

**手順:**
1. このリポジトリをクローン
2. VS Code でプロジェクトを開く
3. コマンドパレット (Ctrl+Shift+P / Cmd+Shift+P) を開く
4. "Dev Containers: Reopen in Container" を選択
5. コンテナのビルドと起動を待つ（初回は数分かかります）

詳細は [.devcontainer/README.md](.devcontainer/README.md) を参照してください。

#### オプション2: ローカル環境でのセットアップ

```bash
# 依存ライブラリのインストール
pip install -r requirements.txt

# パッケージを開発モードでインストール
pip install -e .
```

## 実行方法

### 新しいレイヤードアーキテクチャでの使用

**Python モジュールとして使用**:

```python
# データ取得
from stagealgo import fetch_stock_data

stock_data, benchmark_data = fetch_stock_data('AAPL')

# Stage分析
from stagealgo import StageDetector

detector = StageDetector()
stage = detector.detect_stage(stock_data)

# スクリーナーの使用
from stagealgo.screeners import BaseScreener

class MyScreener(BaseScreener):
    def screen_single_ticker(self, ticker):
        # カスタムロジック
        pass

screener = MyScreener.from_csv('stagealgo/data/stock.csv')
results = screener.run()
```

**CLIコマンドで使用**:

```bash
# スクリーナーの実行
stagealgo-screener --input stagealgo/data/stock.csv --output results/

# ダッシュボードの実行
stagealgo-dashboard
```

### Market DashboardのHTML生成

Market Dashboardは、市場全体の状況を視覚的に把握するための総合的なダッシュボードです。以下の手順でHTMLファイルを生成できます。

#### 機能概要

Market Dashboardには以下の情報が含まれます：

1. **Market Exposure (12要因評価)**
   - 市場全体の強さを12の要因で評価し、0-100%のスコアで表示
   - レベル判定: Bullish/Positive/Neutral/Negative/Bearish
   - 評価対象: 主要指数6銘柄 + セクターETF 29銘柄

2. **Market Performance Overview**
   - 主要指数（SPY、QQQ、MAGS、RSP、QQEW、IWM）の詳細分析
   - 各指標: 価格、日次変化率、RS Rating、パフォーマンス（YTD、1W、1M、1Y）、52週高値からの距離、移動平均線の状態

3. **Sector Performance**
   - 29種類のセクターETFとテーマETFの分析
   - 対象: 地域（EPOL、EWG、IEV、INDA）、テーマ（CIBR、IBIT、BLOK、NLR、TAN、UFO、IPO）、セクター（XLF、XLU、XLP、XLK、XLE、ITB、IBB）など
   - 各指標: Market Performanceと同じ

4. **Macro Performance**
   - マクロ経済指標の分析
   - 対象: U.S. Dollar（NYICDX）、VIX（^VIX）、20年超債券（TLT）
   - 各指標: Market Performanceと同じ

5. **Oratnek Screeners**
   - IBDスタイルの6つのスクリーニングリスト
   - Momentum 97、Explosive EPS Growth、Healthy Chart、Up on Volume、Top 2% RS Rating、4% Bullish Yesterday

#### Market Exposure (12要因評価) の詳細計算方法

Market Exposureは、市場全体の強さを客観的に評価するための指標です。以下の12要因を評価し、総合スコアを算出します。

**評価対象銘柄**: 主要指数6銘柄（SPY、QQQ、MAGS、RSP、QQEW、IWM） + セクターETF 29銘柄 = 合計35銘柄

**12の評価要因**:

1. **パフォーマンス評価（4要因）**
   - YTD > 0: 年初来プラスの銘柄が過半数（50%以上）か
   - 1W > 0: 1週間プラスの銘柄が過半数か
   - 1M > 0: 1ヶ月プラスの銘柄が過半数か
   - 1Y > 0: 1年プラスの銘柄が過半数か

2. **52週高値からの位置（1要因）**
   - 52週高値の90%以上: 52週高値の90%以上の水準にある銘柄が過半数か

3. **VIX状態（1要因）**
   - VIX < 20: VIXが20未満か（市場の恐怖指数が低い状態）

4. **移動平均線の状態（6要因）**
   - 10MA以上: 10日移動平均線以上の銘柄が過半数か
   - 20MA以上: 20日移動平均線以上の銘柄が過半数か
   - 50MA以上: 50日移動平均線以上の銘柄が過半数か
   - 200MA以上: 200日移動平均線以上の銘柄が過半数か
   - MA順序: 10>20>50>200の完璧な順序の銘柄が過半数か
   - MA上昇トレンド: すべてのMAが上昇トレンドの銘柄が過半数か

**スコアリング方法**:
```
各要因について、条件を満たす銘柄数が全体の50%以上の場合: 1点
各要因について、条件を満たす銘柄数が全体の50%未満の場合: 0点

Market Exposure Score = (ポジティブな要因の数 / 12) × 100
```

**レベル判定**:
- **Bullish** (80-100%): 12要因中10個以上がポジティブ（市場は非常に強い）
- **Positive** (60-80%): 12要因中8-9個がポジティブ（市場は強い）
- **Neutral** (40-60%): 12要因中5-7個がポジティブ（市場は中立）
- **Negative** (20-40%): 12要因中3-4個がポジティブ（市場は弱い）
- **Bearish** (0-20%): 12要因中0-2個がポジティブ（市場は非常に弱い）

**計算例**:
35銘柄のうち、18銘柄（51%）のYTDがプラスの場合、「YTD > 0」要因は1点獲得。12要因中8個が1点の場合、スコアは66.7%となり「Positive」レベルと判定されます。

#### 実行方法

**Python スクリプトで実行**:

```bash
# リポジトリのルートディレクトリで実行
python run_dashboard.py
```

このコマンドを実行すると、以下の処理が自動的に行われます：

1. FMP APIから株価データを取得（主要指数、セクターETF、マクロ指標）
2. Market Exposure（12要因評価）を計算
3. Market Performance（主要指数6銘柄）を計算
4. Sector Performance（セクターETF 29銘柄）を計算
5. Macro Performance（Dollar、VIX、TLT）を計算
6. Oratnekスクリーニングを実行（6つのリスト）
7. HTMLダッシュボードを生成して `market_dashboard.html` として保存
8. Market Exposure履歴を `market_exposure_history.csv` に保存

**生成されるファイル**:

- `market_dashboard.html`: ブラウザで開くことができるインタラクティブなダッシュボード
- `market_exposure_history.csv`: Market Exposureスコアの履歴データ（日次更新）

**使用方法**:

```bash
# 1. HTMLダッシュボードを生成
python run_dashboard.py

# 2. ブラウザでHTMLファイルを開く
# macOS
open market_dashboard.html

# Linux
xdg-open market_dashboard.html

# Windows
start market_dashboard.html
```

#### 注意事項

- FMP (FinancialModelingPrep) APIキーが環境変数 `FMP_API_KEY` に設定されている必要があります
- データ取得には時間がかかる場合があります（数分程度）
- Oratnekスクリーニングを無効にする場合は、`MarketDashboard(enable_screeners=False)` で初期化してください

### A) ローカル環境での実行（レガシー）

旧バージョンのスクリプトを直接実行する場合:

```bash
# 依存ライブラリのインストール
pip install -r requirements.txt

# ティッカーリストの生成
python stagealgo/utils/ticker_loader.py

# 分析の実行（旧スクリプト - ルートディレクトリに残存）
# 注意: これらは将来のバージョンで削除される可能性があります
python screener.py  # stagealgo/cli/run_screener.py を推奨
```

### B) Dockerを利用した実行

DockerとDocker Composeを利用すると、OSに依存しない簡単なコマンドで環境を構築し、各スクリプトを実行できます。

#### 1. スクリプトの実行
プロジェクトのルートディレクトリで、以下の`docker-compose run`コマンドを使用します。初回実行時に、必要なDockerイメージが自動的にビルドされます。

**ステップ1: ティッカーリストの準備**
`get_tickers.py` を実行し、分析対象のティッカーリスト `stock.csv` を生成します。
```bash
docker-compose run --rm app python get_tickers.py
```
このコマンドにより、`stock.csv` がカレントディレクトリに作成されます。

**ステップ2: ステージ分析の実行**
`stock.csv` を基に、各種分析スクリプトを実行します。

- **A) 有望な銘柄を抽出しCSVに出力**
  `stage1or2.py` を実行すると、分析結果が `stage1or2.csv` として保存されます。
  ```bash
  docker-compose run --rm app python stage1or2.py
  ```

- **B) 全銘柄の最新ステージを分析**
  `main.py` は、全銘柄の分析結果をコンソールに直接出力します。
  ```bash
  docker-compose run --rm app python main.py
  ```

**ステップ3: 個別銘柄の過去分析 (オプション)**
`historical_analyzer.py` を使って、特定の銘柄（例: AAPL）の過去のステージ移行履歴を分析します。
```bash
docker-compose run --rm app python historical_analyzer.py AAPL
```

#### 2. コンテナへのアクセス (オプション)
コンテナをバックグラウンドで起動し、中に入って作業することもできます。
```bash
# コンテナをバックグラウンドで起動
docker-compose up -d

# 起動したコンテナ内でbashを起動
docker-compose exec app bash
```

## Oratnek Screener - SQLiteベースの高速スクリーニング

Oratnek Screenerは、IBD (Investor's Business Daily) 手法に基づく6つのスクリーニングリストを提供します。
**Version 2.0では、SQLiteベースのデータ管理とマルチプロセス化により大幅に高速化されました。**

### データ取得とキャッシュ

Oratnek Screenerは、以下のデータをFinancialModelingPrep (FMP) APIから取得します：

#### 株価データ
- 日次OHLCV（始値、高値、安値、終値、出来高）
- 週次OHLCV
- 移動平均線（SMA 10/21/50/150/200、EMA 200）

#### ファンダメンタルデータ
- **市場情報**: 時価総額、セクター、業種
- **EPS関連データ**:
  - **実績EPS**: 四半期損益計算書から取得（`get_income_statement()`）
  - **予想EPS**: アナリスト予想データから取得（`get_earnings_surprises()`）
  - **EPS成長率の計算**:
    - `eps_growth_last_qtr`: 前四半期比EPS成長率 (%)
      ```python
      eps_growth_last_qtr = ((最新四半期EPS - 1四半期前EPS) / |1四半期前EPS|) × 100
      ```
    - `eps_est_cur_qtr_growth`: 今四半期予想EPS成長率（YoY, %)
      ```python
      eps_est_cur_qtr_growth = ((予想EPS - 4四半期前EPS) / |4四半期前EPS|) × 100
      ```

**データ保存先**:
- SQLiteデータベース（`data/oratnek/oratnek_cache.db`）
  - `daily_prices`: 日次株価 + 移動平均
  - `weekly_prices`: 週次株価
  - `fundamental_data`: ファンダメンタル情報（**EPS、市場キャップ、セクター等**）
  - `data_metadata`: データメタ情報

**キャッシュ戦略**:
- 株価データ: 日次更新（差分取得）
- ファンダメンタルデータ: 1日に1回更新
- 初回実行時は全履歴を取得、2回目以降は差分のみ取得

### IBD指標の詳細計算方法

#### 1. RS Rating (相対的強さレーティング)

**計算方法** (oratnek_screeners.py:53-82):

RS Ratingは、個別銘柄のパフォーマンスを市場全体（ベンチマーク: SPY）と比較し、0-100のスケールで評価します。

**加重平均の計算式**:
```
RS Rating = 40% × 直近3ヶ月（63営業日）の収益率
          + 20% × 直近6ヶ月（126営業日）の収益率
          + 20% × 直近9ヶ月（189営業日）の収益率
          + 20% × 直近12ヶ月（252営業日）の収益率
```

**実装の特徴**:
- RSCalculator クラスを使用してIBDスタイルのRSスコアを計算
- 最新のRSスコアを0-100にスケーリング（中心を50に調整）
- 252営業日（約1年）未満のデータの場合はデフォルト値50.0を返す
- パーセンタイルランキングに変換し、銘柄を相対的に評価

**用途**:
- RS Rating ≥ 90: 市場全体の上位10%（強い）
- RS Rating ≥ 98: 市場全体の上位2%（非常に強い）
- RS Rating < 40: 市場全体の下位40%（弱い）

#### 2. A/D Rating (蓄積/分散レーティング)

**計算方法** (oratnek_screeners.py:85-139):

A/D Ratingは、機関投資家による買い（蓄積）と売り（分散）の動向を評価します。

**計算ステップ**:
1. 直近13週間（約65営業日）のデータを取得
2. 各日について以下を計算:
   ```
   当日が上昇日の場合: AD値 += 当日の出来高
   当日が下落日の場合: AD値 -= 当日の出来高
   横ばいの場合: AD値は変化なし
   ```
3. 平均出来高で正規化:
   ```
   正規化AD = AD値 / (平均出来高 × 評価期間の日数)
   ```
4. レーティングに変換:
   - **A**: 正規化AD > 0.5 （非常に強い蓄積）
   - **B**: 0.2 < 正規化AD ≤ 0.5 （蓄積）
   - **C**: -0.2 ≤ 正規化AD ≤ 0.2 （中立）
   - **D**: -0.5 ≤ 正規化AD < -0.2 （分散）
   - **E**: 正規化AD < -0.5 （非常に強い分散）

**解釈**:
- A/B: 機関投資家が積極的に買い集めている（買いシグナル）
- C: 中立
- D/E: 機関投資家が売却している（売りシグナル）

#### 3. Comp Rating (総合レーティング)

**計算方法** (oratnek_screeners.py:142-155):

Comp Ratingは、RS RatingとEPS Ratingを組み合わせた総合評価指標です。

**計算式**:
```
Comp Rating = RS Rating × 0.6 + EPS Rating × 0.4
```

**実装の特徴**:
- RS Ratingをやや重視（60%）
- EPS Ratingは40%の重み（現在はデフォルト50.0を使用）
- 最終スコアは0-100の範囲に正規化

**用途**:
- Comp Rating ≥ 80: 高品質銘柄
- Comp Rating < 50: 低品質銘柄

#### 4. 相対出来高 (Relative Volume)

**計算方法** (oratnek_screeners.py:158-180):

相対出来高は、当日の出来高が平均的な出来高と比較してどの程度活発かを示します。

**計算式**:
```
相対出来高 = 当日の出来高 / 過去50日間の平均出来高
```

**解釈**:
- 相対出来高 > 1.5: 非常に活発（150%以上）
- 相対出来高 > 1.2: 活発（120%以上）
- 相対出来高 ≈ 1.0: 通常レベル
- 相対出来高 < 0.5: 閑散

#### 5. パフォーマンス計算

**リターン計算** (oratnek_screeners.py:296-305):

各期間のリターンを以下のように計算します:

```python
returns_1m = (現在価格 / 21営業日前の価格 - 1) × 100  # 1ヶ月リターン(%)
returns_3m = (現在価格 / 63営業日前の価格 - 1) × 100  # 3ヶ月リターン(%)
returns_6m = (現在価格 / 126営業日前の価格 - 1) × 100 # 6ヶ月リターン(%)
returns_1y = (現在価格 / 252営業日前の価格 - 1) × 100 # 1年リターン(%)
```

**日次変化率** (oratnek_screeners.py:308-320):
```python
price_change_pct = (当日終値 - 前日終値) / 前日終値 × 100
change_from_open_pct = (当日終値 - 当日始値) / 当日始値 × 100
```

#### 6. RS Line新高値チェック

**計算方法** (oratnek_screeners.py:329):

簡易版の実装として、RS Ratingの閾値で判定:
```python
rs_line_new_high = (rs_rating >= 90)
```

**本来の意味**:
- RS Lineは、個別銘柄のパフォーマンスを市場ベンチマーク（S&P 500）で割った比率
- RS Line新高値は、個別銘柄が市場に対して過去最高の相対的強さを示している状態

### 主な改善点

- **SQLiteキャッシュ**: FinancialModelingPrepからダウンロードしたデータをSQLiteに保存し、2回目以降は差分更新のみを実行
- **マルチプロセス化**: 最大10並列で銘柄データを取得・計算（`ORATNEK_MAX_WORKERS`で調整可能）
- **バッチ処理**: 50銘柄ずつバッチ処理して効率化（`ORATNEK_BATCH_SIZE`で調整可能）
- **移動平均の事前計算**: SQLiteに保存時にSMA/EMAを計算済み

### 6つのスクリーニングリストの詳細条件

#### 1. Momentum 97 (oratnek_screeners.py:340-390)

**目的**: 短期・中期・長期すべての期間で最高のモメンタムを持つ銘柄を発見

**スクリーニング条件**:
```python
# 1ヶ月、3ヶ月、6ヶ月リターンのパーセンタイルランキングを計算
rank_1m_pct = 銘柄の1ヶ月リターンの全体でのパーセンタイル順位
rank_3m_pct = 銘柄の3ヶ月リターンの全体でのパーセンタイル順位
rank_6m_pct = 銘柄の6ヶ月リターンの全体でのパーセンタイル順位

# 条件: すべての期間で上位3%（97パーセンタイル以上）
条件 = (rank_1m_pct >= 97) AND (rank_3m_pct >= 97) AND (rank_6m_pct >= 97)
```

**出力データ**:
- ticker: ティッカーシンボル
- returns_1m: 1ヶ月リターン (%)
- returns_3m: 3ヶ月リターン (%)
- returns_6m: 6ヶ月リターン (%)
- rank_1m_pct, rank_3m_pct, rank_6m_pct: 各パーセンタイル順位

**ソート**: 1ヶ月リターンの降順

**トレード戦略**:
- 強いモメンタムを持つ銘柄を発見
- ブレイクアウト後の追撃買い候補
- リスク: 既に大きく上昇しているため天井圏の可能性もある

#### 2. Explosive EPS Growth (oratnek_screeners.py:500-550)

**目的**: 高い利益成長率を持つ強い銘柄を発見

**スクリーニング条件**:
```python
条件1: rs_rating >= 80                    # 上位20%の相対的強さ
条件2: eps_est_cur_qtr_growth >= 100      # 今四半期予想EPS成長率が100%以上（YoY）
                                          # ※EPSデータが取得できない場合はこの条件をスキップ
条件3: avg_volume_50d >= 100,000          # 十分な流動性
条件4: price >= sma_50                    # 50日移動平均線より上
```

**EPSデータの取得と使用**:
- FMP APIの`get_income_statement()`で四半期EPSを取得
- FMP APIの`get_earnings_surprises()`で予想EPSを取得
- `eps_est_cur_qtr_growth`を計算:
  ```python
  # 予想EPSと前年同期（4四半期前）の実績EPSから計算
  eps_est_cur_qtr_growth = ((予想EPS - 4四半期前EPS) / |4四半期前EPS|) × 100
  ```
- EPSデータが利用できない銘柄でも、RS Rating条件のみで通過可能

**すべての条件を満たす銘柄を抽出**

**出力データ**:
- ticker: ティッカーシンボル
- price: 現在価格
- rs_rating: RS Rating
- eps_est_cur_qtr_growth: 今四半期予想EPS成長率 (%)
- avg_volume_50d: 50日平均出来高
- price_vs_sma50_pct: 50日移動平均線からの乖離率 (%)

**ソート**: EPS成長率の降順

**トレード戦略**:
- 強いトレンド中の成長株を発見
- 50日移動平均線がサポートラインとして機能
- 高いRS Ratingは市場全体に対する強さを示す

#### 3. Up on Volume (oratnek_screeners.py:552-610)

**目的**: 出来高を伴って上昇している機関投資家注目銘柄を発見

**スクリーニング条件**:
```python
条件1: price_change_pct >= 0          # 当日上昇中
条件2: vol_change_pct >= 20           # 出来高が50日平均の120%以上
条件3: price >= 10                    # 最低価格フィルタ
条件4: avg_volume_50d >= 100,000      # 十分な流動性
条件5: market_cap >= 250              # 時価総額 $250M以上
条件6: rs_rating >= 80                # 強い相対的強さ
条件7: eps_growth_last_qtr >= 20      # 前四半期比EPS成長率が20%以上
条件8: ad_rating in ['A', 'B', 'C']  # 蓄積または中立
```

**EPSデータの使用**:
- `eps_growth_last_qtr`（前四半期比EPS成長率）を条件に使用
- FMP APIの`get_income_statement()`から四半期EPSを取得し、以下を計算:
  ```python
  eps_growth_last_qtr = ((最新四半期EPS - 1四半期前EPS) / |1四半期前EPS|) × 100
  ```

**すべての条件を満たす銘柄を抽出**

**出力データ**:
- ticker: ティッカーシンボル
- price: 現在価格
- price_change_pct: 当日変化率 (%)
- vol_change_pct: 出来高変化率 (%)
- market_cap: 時価総額 ($M)
- eps_growth_last_qtr: 前四半期比EPS成長率 (%)
- rs_rating: RS Rating
- ad_rating: A/D Rating
- avg_volume_50d: 50日平均出来高

**ソート**: 出来高変化率の降順

**トレード戦略**:
- 機関投資家が買い集めている銘柄を発見
- 出来高を伴う上昇は信頼性が高い
- 当日エントリーまたは翌日寄り付きでのエントリー候補

#### 4. Top 2% RS Rating (oratnek_screeners.py:612-668)

**目的**: 市場全体で最強の相対的強さを持ち、完璧なトレンドを示す銘柄を発見

**スクリーニング条件**:
```python
条件1: rs_rating >= 98              # 上位2%の相対的強さ
条件2: sma_10 > sma_21              # 短期MAが中期MAより上
条件3: sma_21 > sma_50              # 中期MAが長期MAより上（完璧な順序）
条件4: avg_volume_50d >= 100,000    # 十分な流動性
条件5: volume >= 100,000            # 当日も十分な出来高
```

**すべての条件を満たす銘柄を抽出**

**出力データ**:
- ticker: ティッカーシンボル
- price: 現在価格
- rs_rating: RS Rating
- sma_10, sma_21, sma_50: 各移動平均線
- avg_volume_50d: 50日平均出来高

**ソート**: RS Ratingの降順

**トレード戦略**:
- 市場で最も強い銘柄を発見
- 移動平均線の順序が完璧なため、強いトレンドが継続中
- 押し目買いの候補

#### 5. 4% Bullish Yesterday (oratnek_screeners.py:670-734)

**目的**: 前日に急騰した銘柄で、当日も勢いが継続している銘柄を発見

**スクリーニング条件**:
```python
# 昨日の変化率を計算
yesterday_change = (昨日終値 - 一昨日終値) / 一昨日終値 × 100

条件1: yesterday_change > 4.0           # 昨日4%以上上昇
条件2: price >= 1.0                     # 最低価格フィルタ
条件3: rel_volume > 1.0                 # 平均以上の出来高
条件4: change_from_open_pct > 0         # 寄り高から更に上昇
条件5: avg_volume_90d > 100,000         # 十分な流動性
```

**すべての条件を満たす銘柄を抽出**

**出力データ**:
- ticker: ティッカーシンボル
- price: 現在価格
- yesterday_change_pct: 昨日の変化率 (%)
- rel_volume: 相対出来高
- change_from_open_pct: 寄り高からの変化率 (%)
- avg_volume_90d: 90日平均出来高

**ソート**: 昨日の変化率の降順

**トレード戦略**:
- ブレイクアウトまたはニュース材料による急騰銘柄を発見
- 当日も寄り高から上昇しているため勢いが継続
- デイトレードまたは短期スイングトレード候補

#### 6. Healthy Chart Watch List (oratnek_screeners.py:736-802)

**目的**: Stage 2（上昇トレンド）にあり、健全なチャート形状を持つ高品質銘柄を発見

**スクリーニング条件**:
```python
# 短期MA順序（Stage 2確認）
条件1: sma_10 > sma_21                  # 10日MA > 21日MA
条件2: sma_21 > sma_50                  # 21日MA > 50日MA

# 長期MA順序（Stage 2確認）
条件3: sma_50 > sma_150                 # 50日MA > 150日MA
条件4: sma_150 > sma_200                # 150日MA > 200日MA

# 強度フィルタ
条件5: rs_line_new_high                 # RS Line新高値（rs_rating >= 90）
条件6: rs_rating >= 90                  # 上位10%の相対的強さ
条件7: ad_rating in ['A', 'B']         # 強い蓄積
条件8: comp_rating >= 80                # 高い総合レーティング
条件9: avg_volume_50d >= 100,000        # 十分な流動性
```

**すべての条件を満たす銘柄を抽出**

**出力データ**:
- ticker: ティッカーシンボル
- price: 現在価格
- rs_rating: RS Rating
- ad_rating: A/D Rating
- comp_rating: Composite Rating
- sma_50, sma_150, sma_200: 各移動平均線
- avg_volume_50d: 50日平均出来高

**ソート**: Composite Ratingの降順

**トレード戦略**:
- 長期保有に適した高品質銘柄を発見
- すべての移動平均線が完璧な順序で、強いトレンド継続中
- Stan Weinstein のStage 2銘柄
- ポートフォリオのコア銘柄候補

### 使用方法

#### 基本的な使い方

```python
from oratnek_screeners import run_oratnek_screener

# デフォルト銘柄でスクリーニング実行（マルチプロセス有効）
results = run_oratnek_screener()

# 結果の確認
for screen_name, df in results.items():
    print(f"{screen_name}: {len(df)} stocks")
    print(df.head())
```

#### カスタム銘柄リストでの実行

```python
# カスタム銘柄リスト
my_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']

# スクリーニング実行
results = run_oratnek_screener(tickers=my_tickers, use_multiprocessing=True)
```

#### コマンドラインから実行

```bash
# マルチプロセス有効（デフォルト）
python oratnek_screeners.py

# マルチプロセス無効
python oratnek_screeners.py --no-mp

# テストスクリプト実行
python test_oratnek_screener.py

# ベンチマーク付きテスト
python test_oratnek_screener.py --benchmark
```

### 環境変数

`.env`ファイルに以下の変数を設定できます：

```bash
# FMP API設定
FMP_API_KEY=your_api_key_here
FMP_RATE_LIMIT=750  # Premium Plan: 750 req/min

# Oratnek Screener設定
ORATNEK_MAX_WORKERS=10   # 並列ワーカー数
ORATNEK_BATCH_SIZE=50    # バッチサイズ
```

### スクリーニング結果の保存先

**出力先**: `data/oratnek/results/`

- `latest.json`: 最新のスクリーニング結果
- `screening_YYYYMMDD_HHMMSS.json`: タイムスタンプ付き結果
- `screener_[name]_YYYYMMDD.csv`: 各スクリーナーのCSV

### パフォーマンス

- **初回実行**: 全データをダウンロード（時間がかかる）
- **2回目以降**: 差分更新のみ（高速）
- **マルチプロセス**: 約5-10倍の高速化（銘柄数とワーカー数による）

### Market Dashboardとの統合

Market Dashboardから自動的にOratnekスクリーニングが実行されます：

```python
from market_dashboard import MarketDashboard

# スクリーニング有効（デフォルト）
dashboard = MarketDashboard()
exposure, market_performance, sectors_performance, macro_performance, screener_results = dashboard.generate_dashboard()

# スクリーニング無効
dashboard = MarketDashboard(enable_screeners=False)
exposure, market_performance, sectors_performance, macro_performance, screener_results = dashboard.generate_dashboard()
```

## プロジェクト構造

**Version 2.0 - レイヤードアーキテクチャ採用**

```
stagealgo/                        # メインパッケージ
│
├── 📁 core/                      # Layer 1-2: 基盤システム
│   ├── infrastructure/           # Layer 1: データ取得
│   │   ├── fmp_client.py        # FMP API クライアント
│   │   └── data_fetcher.py      # データ取得統合
│   │
│   └── indicators/               # Layer 2: 基礎指標
│       └── basic.py              # SMA, EMA, RSI等
│
├── 📁 analysis/                  # Layer 3: 分析エンジン
│   ├── stage/                    # Stage分析
│   │   ├── detector.py          # Stage検出
│   │   └── history_manager.py   # 履歴管理
│   │
│   ├── strength/                 # 相対強度分析
│   │   └── rs_calculator.py     # RS計算
│   │
│   ├── volume/                   # 出来高分析
│   │   ├── analyzer.py          # 出来高分析
│   │   ├── vcp_detector.py      # VCPパターン
│   │   └── vwap_analyzer.py     # VWAP分析
│   │
│   ├── pattern/                  # パターン分析
│   │   ├── base_analyzer.py     # ベースパターン
│   │   └── minervini_template.py # Minerviniテンプレート
│   │
│   └── risk/                     # リスク分析
│       └── atr_analyzer.py      # ATR分析
│
├── 📁 screeners/                 # Layer 4: スクリーニング
│   ├── base_screener.py         # 基底クラス（並列処理）
│   ├── stage_screener.py        # Stageスクリーナー
│   ├── base_pattern_screener.py # パターンスクリーナー
│   └── oratnek_screeners.py     # Oratnekスクリーナー
│
├── 📁 dashboard/                 # Layer 4: ダッシュボード
│   ├── market_analyzer.py       # マーケット分析
│   ├── breadth_analyzer.py      # 市場幅分析
│   └── visualizer.py            # 可視化
│
├── 📁 cli/                       # Layer 5: CLI
│   ├── run_screener.py          # スクリーナーCLI
│   └── run_dashboard.py         # ダッシュボードCLI
│
├── 📁 utils/                     # ユーティリティ
│   └── ticker_loader.py         # ティッカー取得
│
├── 📁 data/                      # データファイル
│   └── stock.csv                # 銘柄リスト
│
├── 📁 docs/                      # ドキュメント
│   └── architecture.md          # アーキテクチャ詳細
│
├── setup.py                      # パッケージ設定
├── requirements.txt              # 依存ライブラリ
└── README.md                     # このファイル
```

詳細なアーキテクチャについては [architecture.md](stagealgo/docs/architecture.md) を参照してください。

### アーキテクチャの利点

- ✅ **保守性**: 変更の影響範囲が明確
- ✅ **拡張性**: 新機能追加が容易
- ✅ **テスト性**: レイヤーごとの独立テスト
- ✅ **可読性**: コードの配置が予測可能
- ✅ **スケーラビリティ**: チーム開発に対応

## 分析ロジック詳細

### 第1部：現在のステージ判定ロジック

本システムは、まず50日移動平均線（MA50）の傾きと、過去の価格帯における現在価格の位置に基づいて、現在のステージを総合的に判断します。

1.  **ステージ4 (下降期)**: MA50の傾きが明確な下降トレンドを示している場合。
2.  **ステージ2 (上昇期)**: MA50の傾きが明確な上昇トレンドにあり、かつステージ1→2への移行スコアがB判定（70点）以上の場合。これにより、弱い上昇シグナルを除外します。
3.  **ステージ1 (底固め期) / ステージ3 (天井圏)**: 上記以外の場合（MA50が横ばい、または上昇が弱い場合）、過去の高値からの下落率に基づいて判断します。
    -   過去1年間の高値から大きく下落している場合は、**ステージ1**と見なします。
    -   過去150日間の高値圏で推移している場合は、**ステージ3**と見なします。

### 第2部：ステージ移行スコアリング

#### 1. ステージ1 → 2【上昇期への移行】

**目的**: 本格的な上昇トレンドの始まり（ブレイクアウト）を捉え、最適なエントリーポイントを特定します。

| カテゴリ | 評価項目 | 配点 |
|:---|:---|:---|
| 出来高 | ブレイクアウト時の出来高急増 | 25点 |
| 価格 | 50日高値からのブレイクアウト | 10点 |
| トレンド | MA50のゴールデンクロスと傾き | 15点 |
| 相対強度 | RS Ratingの強さ | 15点 |
| ボラティリティ | ATRを用いたトレンド初動の検知 | 15点 |
| 市場環境 | ベンチマーク(SPY)の全体相場 | 20点 |
| **合計** | | **100点** |

| 判定 | 条件 | 推奨アクション |
|:---|:---|:---|
| A判定 | スコア85点以上 + 確認済み | 自信を持ってエントリーを検討すべき理想的なシグナル。 |
| B判定 | スコア70点以上 + 確認済み | エントリーを検討。リスク管理を推奨。 |
| C判定 | 上記以外 | エントリーは見送り。スコアが高くても未確認なら「確認待ち」。 |

#### 2. ステージ2 → 3【天井圏への移行】

**目的**: 上昇トレンドの勢いの衰えと、大口投資家による利益確定売り（Distribution）の兆候を捉えます。

| カテゴリ | 評価項目 | 配点 |
|:---|:---|:---|
| 過熱感 | ATRを用いたMAからの乖離 | 30点 |
| 大口の売り | 下落日の出来高増 (Distribution Day) | 25点 |
| 反転サイン | 長い上ヒゲの出現 | 20点 |
| トレンド鈍化 | MA50の傾きの平坦化 | 15点 |
| 相対強度 | RS Ratingの低下傾向 | 10点 |
| **合計** | | **100点** |

| スコア | 判定 | 推奨アクション |
|:---|:---|:---|
| 75点以上 | 危険 | ポジションの大部分の利益確定を強く推奨。 |
| 50-74点 | 警告 | 新規の買いは見送り、一部を利益確定。 |
| 50点未満 | 安全 | ポジションを維持し、トレンド継続を期待。 |

#### 3. ステージ3 → 4【下降期への移行】

**目的**: レンジ相場の終焉と、本格的な下降トレンドへの突入（ブレイクダウン）を特定します。

| 評価項目 | 配点 |
|:---|:---|
| 過去50日間の最安値を更新 | 30点 |
| 出来高を伴うブレイクダウン | 30点 |
| MA50が下降トレンドに転換 | 25点 |
| RS Ratingが40以下に低迷 | 15点 |
| **合計** | **100点** |

| スコア | 判定 | 推奨アクション |
|:---|:---|:---|
| 75点以上 | 危険 | 全ポジションを決済し、損失を限定する。 |
| 75点未満 | 警告 | 保有は避け、ブレイクダウンに最大限警戒する。 |

#### 4. ステージ4 → 1【底固め期への移行】

**目的**: 下降トレンドが終了し、次の上昇サイクルに向けたエネルギーを溜める期間に入ったことを判定します。

| 評価項目 | 配点 |
|:---|:---|
| MA50の傾きが横ばいになり、新安値をつけなくなる | 35点 |
| 出来高が減少し、売りが枯渇する | 30点 |
| 価格のボラティリティが低下する | 20点 |
| RS Ratingが底を打ち、上昇に転じる | 15点 |
| **合計** | **100点** |

| スコア | 判定 | 推奨アクション |
|:---|:---|:---|
| 70点以上 | 底打ち完了 | 監視リストに追加し、将来のステージ2移行を待つ。 |
| 70点未満 | 下降継続 | 底打ちの条件は未達。引き続き手を出さない。 |