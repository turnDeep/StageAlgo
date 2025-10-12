import pandas as pd

def get_and_save_tickers():
    """
    DataHub.ioからNASDAQとNYSEのティッカーリストを取得し、
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
    excluded_suffixes = ['.U', '.W', '.A', '.B']
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

    final_count = len(combined_df)
    print(f"フィルタリングにより {initial_count - final_count} 件のティッカーが除外されました。")
    print(f"フィルタリング後、{final_count} 件のティッカーが残りました。")

    # 5. stock.csvに保存（ヘッダー付き、インデックスなし）
    try:
        combined_df.to_csv('stock.csv', index=False, columns=['Ticker', 'Exchange'])
        print("ティッカーリストを stock.csv に正常に保存しました。")
    except Exception as e:
        print(f"エラー: stock.csv への書き込み中にエラーが発生しました: {e}")


if __name__ == '__main__':
    get_and_save_tickers()
