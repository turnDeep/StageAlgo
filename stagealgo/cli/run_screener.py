"""
Stock Screener Orchestrator

【役割】
株式分析ワークフローの全体制御

【責任】
1. ステージ分析スクリプトの実行
2. ベース分析スクリプトの実行
3. 各分析ステップ間のデータ（ファイル名）の引き渡し

【連携】
- run_stage_analyzer.py: ステージ2スクリーニング機能を呼び出す
- run_base_analyzer.py: ベース分析機能を呼び出す
"""
import sys
from datetime import datetime
import pytz

# 分析モジュールをインポート
from run_stage_analyzer import run_stage_screening
from run_base_analyzer import run_base_analysis


def main():
    """
    メイン処理
    ステージ分析とベース分析を順番に実行する
    """
    print("=" * 70)
    print("Stock Analysis Workflow Started")
    print("=" * 70)
    print()

    # --- ステップ1: ステージ分析 ---
    print(">>> Step 1: Running Stage 2 Screener...")
    try:
        # run_stage_analyzer.pyのスクリーニング関数を呼び出す
        # 成功すると、結果が保存されたCSVファイル名が返される
        stage_output_filename = run_stage_screening(input_filename='stock.csv')
        
        # ステージ分析で対象銘柄が見つからなかった場合
        if stage_output_filename is None:
            print("\n" + "=" * 70)
            print("ステージ2の銘柄が見つからなかったため、処理を終了します。")
            print("=" * 70)
            sys.exit(0)

        print("✓ Stage 2 Screener finished successfully.")
        print(f"  - Output file: {stage_output_filename}")

    except Exception as e:
        print(f"エラー: ステージ分析の実行中に予期せぬエラーが発生しました: {e}")
        sys.exit(1)


    # --- ステップ2: ベース分析 ---
    print("\n" + "=" * 70)
    print(">>> Step 2: Running Base Analyzer...")

    try:
        # 日付プレフィックスからベース分析用の出力ファイル名を生成
        date_prefix = stage_output_filename.split('-')[0]
        base_output_filename = f"{date_prefix}-base.csv"

        # run_base_analyzer.pyの分析関数を呼び出す
        # ステージ分析の結果をインプットとして渡す
        run_base_analysis(
            output_filename=base_output_filename,
            input_filename=stage_output_filename
        )
        print("✓ Base Analyzer finished successfully.")
        print(f"  - Output file: {base_output_filename}")

    except Exception as e:
        print(f"エラー: ベース分析の実行中にエラーが発生しました: {e}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("All analysis workflows completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
