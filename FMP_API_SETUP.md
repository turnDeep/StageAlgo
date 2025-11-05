# FinancialModelingPrep API セットアップガイド

## 概要

StageAlgoは、yfinanceからFinancialModelingPrep (FMP) APIに移行しました。FMP APIは、より安定した株価データとファンダメンタルデータを提供します。

## なぜFMP APIに移行したのか？

1. **信頼性の向上**: yfinanceはYahoo FinanceのWebスクレイピングに依存しており、頻繁にエラーが発生していました
2. **データの豊富さ**: FMP APIは、株価データに加えて、財務諸表、EPS、マーケットキャップなど、豊富なファンダメンタルデータを提供します
3. **公式API**: FMP APIは公式のAPIであり、長期的なサポートが期待できます

## APIキーの取得方法

### 1. FMPアカウントの作成

1. [FinancialModelingPrep](https://site.financialmodelingprep.com/) にアクセス
2. 右上の「Sign Up」をクリック
3. メールアドレスとパスワードを入力してアカウントを作成

### 2. APIキーの取得

1. ログイン後、ダッシュボードに移動
2. 「Dashboard」→「API Keys」を選択
3. APIキーが表示されます（例: `abc123def456ghi789jkl012mno345pq`）

### 3. プランの選択

あなたは **Starter Plan** を使用しています。

**Starter Planの制限:**
- 20GB/月の帯域制限（Free: 500MB/月）
- 一部のエンドポイントでティッカー数制限あり
- 月額: 約$14-15（詳細は[料金ページ](https://site.financialmodelingprep.com/pricing-plans)を確認）

**無料プラン (Free Plan) でも試用可能:**
- 500MB/月の帯域制限
- 250 API calls/day
- まずは無料プランで試してから、必要に応じてStarterプランにアップグレード可能

## APIキーの設定方法

### オプション1: 環境変数に設定（推奨）

#### Linux/Mac:

```bash
# .bashrc または .zshrc に追加
export FMP_API_KEY='your_api_key_here'

# または、現在のセッションのみで設定
export FMP_API_KEY='your_api_key_here'
```

#### Windows (PowerShell):

```powershell
# 現在のセッションのみ
$env:FMP_API_KEY='your_api_key_here'

# 永続的に設定（システム環境変数）
[System.Environment]::SetEnvironmentVariable('FMP_API_KEY','your_api_key_here','User')
```

#### Windows (コマンドプロンプト):

```cmd
set FMP_API_KEY=your_api_key_here
```

### オプション2: .envファイルを使用

プロジェクトルートに `.env` ファイルを作成:

```bash
# .env
FMP_API_KEY=your_api_key_here
```

そして、Pythonコードで `python-dotenv` を使用:

```python
from dotenv import load_dotenv
load_dotenv()
```

## テスト方法

APIキーが正しく設定されているか確認:

```bash
# 環境変数の確認
echo $FMP_API_KEY  # Linux/Mac
echo %FMP_API_KEY%  # Windows

# テストスクリプトの実行
python fmp_data_fetcher.py
```

成功すると、AAPLの株価データが表示されます:

```
Testing FMP Data Fetcher...
API Key found: abc123def45...

1. Testing historical price data for AAPL...

AAPL Historical Data (last 5 rows):
            Open    High     Low   Close      Volume  Adj Close
date
2024-11-01  150.0  152.5  149.5  151.2  50000000.0      151.2
...

2. Testing quote data for AAPL...
Current Price: $151.50
Volume: 50,000,000
Market Cap: $2,500,000,000,000

3. Testing profile data for AAPL...
Company: Apple Inc.
Sector: Technology
Industry: Consumer Electronics
Market Cap: $2,500,000,000,000

Test completed!
```

## トラブルシューティング

### エラー: "FMP API Key is required"

環境変数 `FMP_API_KEY` が設定されていません。

**解決策:**
1. 環境変数を設定（上記参照）
2. ターミナルを再起動して環境変数を再読み込み

### エラー: "API request failed: 401 Unauthorized"

APIキーが無効または期限切れです。

**解決策:**
1. FMPダッシュボードで新しいAPIキーを生成
2. 環境変数を更新

### エラー: "API request failed: 429 Too Many Requests"

API呼び出し制限に達しました。

**解決策:**
1. Free Planの場合: Starter Planにアップグレード
2. Starter Planの場合: リクエスト頻度を下げる、またはPremium Planにアップグレード

### データが取得できない（空のDataFrame）

ティッカーシンボルが間違っているか、FMP APIがそのティッカーをサポートしていません。

**解決策:**
1. ティッカーシンボルを確認（例: AAPL, MSFT, GOOGL）
2. FMP APIのドキュメントで対応ティッカーを確認

## FMP API リファレンス

### 主要なエンドポイント

| エンドポイント | 説明 | 使用例 |
|---------------|------|--------|
| `/historical-price-full/{ticker}` | 株価の履歴データ | `fetcher.get_historical_price('AAPL')` |
| `/quote/{ticker}` | リアルタイムクォート | `fetcher.get_quote('AAPL')` |
| `/profile/{ticker}` | 企業プロフィール（セクター、マーケットキャップなど） | `fetcher.get_profile('AAPL')` |
| `/key-metrics/{ticker}` | 主要財務指標（P/E、EPS、ROEなど） | `fetcher.get_key_metrics('AAPL')` |
| `/income-statement/{ticker}` | 損益計算書（EPSデータ含む） | `fetcher.get_income_statement('AAPL')` |
| `/earnings-surprises/{ticker}` | 四半期決算サプライズ | `fetcher.get_earnings_surprises('AAPL')` |

### 公式ドキュメント

- [FMP API Documentation](https://site.financialmodelingprep.com/developer/docs)
- [料金プラン](https://site.financialmodelingprep.com/pricing-plans)
- [FAQ](https://site.financialmodelingprep.com/faqs)

## 使用例

### 基本的な使用方法

```python
from fmp_data_fetcher import FMPDataFetcher

# FMPフェッチャーの初期化
fetcher = FMPDataFetcher()

# 株価の履歴データを取得
df = fetcher.get_historical_price('AAPL', from_date='2024-01-01', to_date='2024-12-31')
print(df.head())

# リアルタイムクォート
quote = fetcher.get_quote('AAPL')
print(f"現在価格: ${quote['price']}")

# 企業プロフィール
profile = fetcher.get_profile('AAPL')
print(f"セクター: {profile['sector']}")
print(f"マーケットキャップ: ${profile['mktCap']:,}")
```

### data_fetcherを使用（既存コードとの互換性）

```python
from data_fetcher import fetch_stock_data

# yfinanceと同じインターフェース
stock_df, benchmark_df = fetch_stock_data('AAPL', period='1y')
print(stock_df.tail())
```

## まとめ

1. FMPアカウントを作成してAPIキーを取得
2. 環境変数 `FMP_API_KEY` を設定
3. `python fmp_data_fetcher.py` でテスト
4. 既存のコード（`market_dashboard.py`、`oratnek_screeners.py`など）はそのまま動作します

問題が発生した場合は、[FMP FAQ](https://site.financialmodelingprep.com/faqs) または [StageAlgo Issue](https://github.com/turnDeep/StageAlgo/issues) で質問してください。
