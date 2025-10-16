“””
ベースカウンティングとブレイクアウト検出モジュール
William O’NeilとMark Minerviniの理論に基づく

主要機能:

1. ベース期間の識別（横ばい統合期間）
1. Stage 1ベース検出とサブステージ判定（1A, 1, 1B）
1. Stage 2内のベースカウンティング（StageDetectorと連携）
1. ブレイクアウトの検出と検証
1. ベース品質の評価
1. Stage情報を含む包括的分析

StageDetectorとの連携:

- analyze_with_stage(): StageDetectorインスタンスを渡して詳細分析
- detect_stage1_bases(): Stage 1のベース検出とサブステージ判定
- count_bases_in_stage2(): Stage 2内のベース数をカウント
- detect_breakouts(): ブレイクアウト時のStage情報を記録
- get_stage2_breakouts(): Stage 2内のブレイクアウトのみを抽出
  “””