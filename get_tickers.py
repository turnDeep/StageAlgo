import pandas as pd

def get_and_save_tickers():
    """
    DataHub.ioからNASDAQとNYSEのティッカーリストを取得し、
    重複を排除して1つのCSVファイルに保存します。
    """
    # 1. CSVファイルのURLを定義 (404エラーを修正)
    nasdaq_url = "https://datahub.io/core/nasdaq-listings/_r/-/data/nasdaq-listed-symbols.csv"
    nyse_url = "https://datahub.io/core/nyse-other-listings/_r/-/data/nyse-listed.csv"

    print("ティッカーリストの取得を開始します...")

    try:
        # 2. URLから直接CSVを読み込む
        print(f"NASDAQのティッカーをダウンロード中: {nasdaq_url}")
        nasdaq_df = pd.read_csv(nasdaq_url)
        # 'Symbol' 列からティッカーを取得し、NaNを削除して文字列に変換
        nasdaq_df.dropna(subset=['Symbol'], inplace=True)
        nasdaq_tickers = nasdaq_df['Symbol'].astype(str).tolist()
        print(f"{len(nasdaq_tickers)} 件のNASDAQティッカーを取得しました。")

        print(f"NYSEのティッカーをダウンロード中: {nyse_url}")
        nyse_df = pd.read_csv(nyse_url)
        # 'ACT Symbol' 列からティッカーを取得し、NaNを削除して文字列に変換
        nyse_df.dropna(subset=['ACT Symbol'], inplace=True)
        nyse_tickers = nyse_df['ACT Symbol'].astype(str).tolist()
        print(f"{len(nyse_tickers)} 件のNYSEティッカーを取得しました。")

    except Exception as e:
        print(f"エラー: CSVファイルのダウンロードまたは読み込み中にエラーが発生しました: {e}")
        return

    # 3. リストを結合し、重複を排除
    all_tickers = nasdaq_tickers + nyse_tickers
    unique_tickers = sorted(list(set(all_tickers)))

    print(f"合計 {len(unique_tickers)} 件のユニークなティッカーが見つかりました。")

    # 4. stock.csvに保存
    try:
        with open('stock.csv', 'w') as f:
            for ticker in unique_tickers:
                f.write(f"{ticker}\n")
        print("ティッカーリストを stock.csv に正常に保存しました。")
    except Exception as e:
        print(f"エラー: stock.csv への書き込み中にエラーが発生しました: {e}")


if __name__ == '__main__':
    get_and_save_tickers()