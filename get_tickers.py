import pandas as pd
import yfinance as yf
from curl_cffi.requests import Session
from tqdm import tqdm
import time

def get_and_save_tickers():
    """
    DataHub.ioからNASDAQとNYSEのティッカーリストを取得し、
    yfinanceで会社名を確認してETF/Fundを除外した後、
    取引所情報を付与して1つのCSVファイルに保存します。
    """
    # 1. CSVファイルのURLを定義
    nasdaq_url = "https://datahub.io/core/nasdaq-listings/_r/-/data/nasdaq-listed-symbols.csv"
    nyse_url = "https://datahub.io/core/nyse-other-listings/_r/-/data/nyse-listed.csv"

    print("ティッカーリストの取得を開始します...")

    try:
        # 2. URLから直接CSVを読み込み、取引所情報を付与
        print(f"NASDAQのティッカーをダウンロード中: {nasdaq_url}")
        nasdaq_df = pd.read_csv(nasdaq_url)
        nasdaq_df.dropna(subset=['Symbol'], inplace=True)
        nasdaq_df = nasdaq_df[['Symbol']].copy()
        nasdaq_df.rename(columns={'Symbol': 'Ticker'}, inplace=True)
        nasdaq_df['Exchange'] = 'NASDAQ'
        print(f"{len(nasdaq_df)} 件のNASDAQティッカーを取得しました。")

        print(f"NYSEのティッカーをダウンロード中: {nyse_url}")
        nyse_df = pd.read_csv(nyse_url)
        nyse_df.dropna(subset=['ACT Symbol'], inplace=True)
        nyse_df = nyse_df[['ACT Symbol']].copy()
        nyse_df.rename(columns={'ACT Symbol': 'Ticker'}, inplace=True)
        nyse_df['Exchange'] = 'NYSE'
        print(f"{len(nyse_df)} 件のNYSEティッカーを取得しました。")

    except Exception as e:
        print(f"エラー: CSVファイルのダウンロードまたは読み込み中にエラーが発生しました: {e}")
        return

    # 3. DataFrameを結合し、ティッカーの重複を排除
    combined_df = pd.concat([nasdaq_df, nyse_df], ignore_index=True)
    combined_df.drop_duplicates(subset=['Ticker'], keep='first', inplace=True)
    combined_df['Ticker'] = combined_df['Ticker'].astype(str)

    print(f"合計 {len(combined_df)} 件のユニークなティッカーが見つかりました。")

    # 4. 除外条件に基づいてティッカーをフィルタリング
    print("指定された条件に基づいてティッカーをフィルタリングします...")
    excluded_suffixes = ['.U', '.W', '.A', '.B', '.R']
    initial_count = len(combined_df)

    # 条件①：ティッカーの文字数が5文字
    combined_df = combined_df[combined_df['Ticker'].str.len() != 5]
    count_after_len = len(combined_df)
    print(f"文字数が5文字のティッカーを除外: {initial_count - count_after_len} 件")

    # 条件②：特定の接尾辞が付いている
    combined_df = combined_df[~combined_df['Ticker'].str.contains('|'.join(s.replace('.', r'\.') for s in excluded_suffixes), na=False)]
    count_after_suffix = len(combined_df)
    print(f"特定の接尾辞を持つティッカーを除外: {count_after_len - count_after_suffix} 件")

    # 条件③：ティッカーの文字の中に$が入っている
    combined_df = combined_df[~combined_df['Ticker'].str.contains(r'\$', na=False)]
    count_after_dollar = len(combined_df)
    print(f"'$'を含むティッカーを除外: {count_after_suffix - count_after_dollar} 件")

    # 条件④：'FILE CREATION TIME'で始まる無効なティッカーを除外
    combined_df = combined_df[~combined_df['Ticker'].str.startswith('FILE CREATION TIME', na=False)]
    count_after_invalid = len(combined_df)
    print(f"'FILE CREATION TIME'で始まる無効なティッカーを除外: {count_after_dollar - count_after_invalid} 件")

    # 5. yfinanceでETF/Fundを除外
    print("\nyfinanceで会社名を取得し、ETF/Fundを除外します...")
    print("※ この処理には時間がかかります（数分〜十数分）")
    
    # curl-cffiのSessionを作成（API制限回避）
    yf_session = Session(impersonate="safari15_5")
    
    batch_size = 30  # HanaViewと同じバッチサイズ
    tickers_to_check = combined_df['Ticker'].tolist()
    valid_tickers = []
    etf_fund_count = 0
    error_count = 0
    
    for i in tqdm(range(0, len(tickers_to_check), batch_size), desc="銘柄情報取得中"):
        batch = tickers_to_check[i:i+batch_size]
        
        for ticker_symbol in batch:
            try:
                ticker_obj = yf.Ticker(ticker_symbol, session=yf_session)
                info = ticker_obj.info
                
                # 会社名を取得
                company_name = info.get('shortName', '') or info.get('longName', '')
                
                # ETFまたはFundが含まれているかチェック
                if company_name:
                    name_upper = company_name.upper()
                    if 'ETF' in name_upper or 'FUND' in name_upper:
                        etf_fund_count += 1
                        continue
                
                # quoteTypeでも二重チェック
                quote_type = info.get('quoteType', '')
                if quote_type in ['ETF', 'MUTUALFUND']:
                    etf_fund_count += 1
                    continue
                
                # 通過した銘柄を有効リストに追加
                valid_tickers.append(ticker_symbol)
                
            except Exception as e:
                # エラーが出た銘柄は念のため除外
                error_count += 1
                continue
        
        # バッチ間で待機（API制限回避）
        if i + batch_size < len(tickers_to_check):
            time.sleep(3)
    
    print(f"\nETF/Fund除外: {etf_fund_count} 件")
    print(f"エラー/データ取得失敗: {error_count} 件")
    
    # 6. 有効なティッカーのみでDataFrameを再構築
    combined_df = combined_df[combined_df['Ticker'].isin(valid_tickers)].copy()
    
    final_count = len(combined_df)
    print(f"最終的に {final_count} 件のティッカーが残りました。")

    # 7. stock.csvに保存（ヘッダー付き、インデックスなし）
    try:
        combined_df.to_csv('stock.csv', index=False, columns=['Ticker', 'Exchange'])
        print("ティッカーリストを stock.csv に正常に保存しました。")
    except Exception as e:
        print(f"エラー: stock.csv への書き込み中にエラーが発生しました: {e}")


if __name__ == '__main__':
    get_and_save_tickers()
