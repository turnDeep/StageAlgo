## yfinanceで取得可能なデータと関数の完全ガイド

### 1. **Tickerクラス - 単一銘柄データ取得**

#### 基本的な使用方法
```python
import yfinance as yf
ticker = yf.Ticker("AAPL")
```

#### 取得可能なプロパティ

**基本情報・メタデータ**
- `info`: 企業の基本情報（時価総額、セクター、業種など）
- `fast_info`: 高速アクセス用の基本情報（最終価格、前日終値など）
- `history_metadata`: 履歴データのメタデータ
- `isin`: ISINコード

**価格・取引データ**
- `history()`: 過去の株価データ（OHLCV）
  - **period**: '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'
  - **interval**: '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'
  - **注意**: 1分足データは過去7日間のみ、日中データは過去60日間のみ
- `actions`: 配当と株式分割の履歴
- `dividends`: 配当履歴
- `splits`: 株式分割履歴
- `capital_gains`: キャピタルゲイン履歴

**財務諸表データ**
- `income_stmt` / `incomestmt`: 損益計算書（年次）
- `quarterly_income_stmt`: 損益計算書（四半期）
- `ttm_income_stmt`: 直近12ヶ月の損益計算書
- `balance_sheet` / `balancesheet`: 貸借対照表（年次）
- `quarterly_balance_sheet`: 貸借対照表（四半期）
- `cash_flow` / `cashflow`: キャッシュフロー計算書（年次）
- `quarterly_cash_flow`: キャッシュフロー計算書（四半期）
- `ttm_cash_flow`: 直近12ヶ月のキャッシュフロー
- `financials`: 主要財務データ（年次）
- `quarterly_financials`: 主要財務データ（四半期）
- `ttm_financials`: 直近12ヶ月の財務データ

**収益・業績データ**
- `earnings`: 収益データ（年次・四半期）
- `quarterly_earnings`: 四半期収益
- `earnings_dates`: 決算発表日（過去・将来）
- `earnings_estimate`: 収益予想
- `earnings_history`: 収益履歴
- `revenue_estimate`: 売上予想
- `eps_trend`: EPS（1株当たり利益）のトレンド
- `eps_revisions`: EPSの修正

**アナリスト情報**
- `analyst_price_targets`: アナリストの目標株価
- `recommendations`: アナリストの推奨（買い、売りなど）
- `recommendations_summary`: 推奨のサマリー
- `upgrades_downgrades`: 格上げ・格下げ情報
- `growth_estimates`: 成長予想

**イベント・カレンダー**
- `calendar`: 決算日、配当日などのイベントカレンダー

**保有者情報**
- `major_holders`: 主要株主
- `institutional_holders`: 機関投資家の保有状況
- `mutualfund_holders`: 投資信託の保有状況
- `insider_purchases`: インサイダー購入
- `insider_roster_holders`: インサイダー保有者名簿
- `insider_transactions`: インサイダー取引

**オプションデータ**
- `options`: 利用可能なオプション満期日のリスト
- `option_chain(date)`: 特定の満期日のオプションチェーン（コール・プット）

**その他**
- `news`: ニュース記事（最大10件）
- `sustainability`: ESG・サステナビリティ評価
- `sec_filings`: SEC提出書類
- `shares`: 発行済株式数の履歴

**投資信託/ETF専用**
- `funds_data`: ファンドデータオブジェクト
  - `description`: ファンド説明
  - `fund_overview`: ファンド概要
  - `fund_operations`: 運用情報
  - `asset_classes`: 資産クラス
  - `top_holdings`: 主要保有銘柄
  - `equity_holdings`: 株式保有
  - `bond_holdings`: 債券保有
  - `bond_ratings`: 債券格付け
  - `sector_weightings`: セクター配分

---

### 2. **download関数 - 複数銘柄の一括データ取得**

```python
yf.download(tickers, start=None, end=None, period=None, interval='1d', ...)
```

**主要パラメータ**
- **tickers**: 文字列またはリスト（例: "AAPL" または ["AAPL", "MSFT"]）
- **period**: '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'
- **interval**: '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'
- **start/end**: 日付範囲（'YYYY-MM-DD'形式）
- **threads**: マルチスレッドでの高速ダウンロード（True/False/整数）
- **group_by**: 'ticker' または 'column'
- **auto_adjust**: OHLCを自動調整（デフォルト: True）
- **actions**: 配当・分割データを含む（デフォルト: False）
- **prepost**: プレ/アフターマーケットデータを含む
- **repair**: 価格の異常値を修復

---

### 3. **Tickersクラス - 複数銘柄の管理**

```python
tickers = yf.Tickers('MSFT AAPL GOOG')
tickers.tickers['MSFT'].info
tickers.tickers['AAPL'].history(period="1mo")
```

---

### 4. **WebSocket - リアルタイム価格ストリーミング**

#### 同期WebSocket
```python
with yf.WebSocket() as ws:
    ws.subscribe(["AAPL", "BTC-USD"])
    ws.listen(message_handler)
```

#### 非同期WebSocket
```python
async with yf.AsyncWebSocket() as ws:
    await ws.subscribe(["AAPL", "NVDA"])
    await ws.listen()
```

**特徴**
- リアルタイムの価格更新
- Yahoo FinanceのWebSocketエンドポイントに接続
- protobuf形式のメッセージ

---

### 5. **Market - 市場サマリー**

```python
market = yf.Market("US")
status = market.status  # 市場の状態
summary = market.summary  # 市場サマリー
```

**利用可能な市場**
- US, GB, ASIA, EUROPE, RATES, COMMODITIES, CURRENCIES, CRYPTOCURRENCIES

---

### 6. **Search - 検索機能**

```python
search = yf.Search("AAPL", max_results=10)
quotes = search.quotes  # 検索結果の銘柄リスト
news = search.news  # 関連ニュース
research = search.research  # リサーチレポート（include_research=True時）
```

---

### 7. **Lookup - ティッカー検索**

```python
lookup = yf.Lookup("AAPL")
all_results = lookup.all  # すべての結果
stocks = lookup.stock  # 株式のみ
etfs = lookup.etf  # ETFのみ
mutualfunds = lookup.mutualfund  # 投資信託のみ
indices = lookup.index  # インデックスのみ
futures = lookup.future  # 先物のみ
currencies = lookup.currency  # 通貨のみ
crypto = lookup.cryptocurrency  # 暗号通貨のみ
```

各メソッドは`get_xxx(count=100)`形式でも使用可能

---

### 8. **Sector & Industry - セクター・業種情報**

```python
tech = yf.Sector('technology')
software = yf.Industry('software-infrastructure')

# 共通プロパティ
tech.key  # キー
tech.name  # 名称
tech.symbol  # シンボル
tech.ticker  # Tickerオブジェクト
tech.overview  # 概要
tech.top_companies  # 主要企業
tech.research_reports  # リサーチレポート

# Sector固有
tech.top_etfs  # 主要ETF
tech.top_mutual_funds  # 主要投資信託
tech.industries  # 業種一覧

# Industry固有
software.sector_key  # セクターキー
software.sector_name  # セクター名
software.top_performing_companies  # 好調な企業
software.top_growth_companies  # 成長企業
```

**Tickerとの連携**
```python
msft = yf.Ticker('MSFT')
tech = yf.Sector(msft.info.get('sectorKey'))
software = yf.Industry(msft.info.get('industryKey'))
```

---

### 9. **EquityQuery & FundQuery - スクリーニング**

#### EquityQuery（株式スクリーニング）
```python
query = yf.EquityQuery('and', [
    yf.EquityQuery('is-in', ['exchange', 'NMS', 'NYQ']),
    yf.EquityQuery('lt', ["epsgrowth.lasttwelvemonths", 15])
])
```

**演算子**
- 比較: EQ, GT, LT, GTE, LTE, BTWN
- 論理: AND, OR

**利用可能なフィールド（一部）**
- 基本情報: region, sector, exchange, peer_group
- 市場指標: intradaymarketcap, lastclosemarketcap
- 成長率: epsgrowth, revenuegrowth
- 財務比率: pe, pb, ps, roe, roa
- その他多数のメトリクス

#### FundQuery（投資信託スクリーニング）
```python
fund_query = yf.FundQuery('eq', ['fundType', 'equity'])
```

---

### 10. **screen関数 - スクリーニング実行**

```python
# カスタムクエリ
results = yf.screen(query, size=100, sortField='ticker')

# 事前定義クエリ
predefined = yf.screen('aggressive_small_caps', count=25)

# 利用可能な事前定義クエリ
yf.PREDEFINED_SCREENER_QUERIES.keys()
```

**パラメータ**
- **query**: カスタムクエリまたは事前定義クエリ名
- **offset**: 結果のオフセット
- **size**: 返す結果数（最大250）
- **count**: 事前定義クエリ用の結果数
- **sortField**: ソートフィールド
- **sortAsc**: 昇順ソート（True/False）

---

### 11. **その他のユーティリティ関数**

```python
# デバッグモード有効化
yf.enable_debug_mode()

# タイムゾーンキャッシュの場所設定
yf.set_tz_cache_location(path)
```

---

### データ制限と注意事項

1. **時間制限**
   - 1分足: 過去7日間のみ
   - 日中データ(<1日): 過去60日間のみ

2. **レート制限**
   - 過度なリクエストでブロックされる可能性

3. **データの信頼性**
   - 非公式API（Yahoo非承認）
   - HTMLスクレイピング使用箇所あり
   - 本番トレーディングには推奨されない

4. **履歴オプションデータ**
   - 現在有効なオプション契約のみ取得可能
   - 期限切れオプションの履歴は取得不可

このまとめは2025年10月時点の情報に基づいています。yfinanceは活発に開発されているため、最新の公式ドキュメント（https://ranaroussi.github.io/yfinance/）も併せてご確認ください。