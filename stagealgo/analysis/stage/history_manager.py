"""
Stage History Manager (完全改訂版)

【重要な改善】
1. Stage Detectorとの完全連携（初期化を正確に）
2. すべてのステージ移行パターンを網羅
3. Stage 3はStage 1の逆ロジックで実装
4. 4ステージが相補的な関係
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict, field
from enum import Enum
from stage_detector import StageDetector

class StageHistoryManager:
    """
    ステージ履歴管理システム（完全改訂版）

    【役割】
    - Stage Detectorの判定に基づき、時系列でのステージ移行を追跡・記録する
    """

    def __init__(self, ticker: str, data_dir: str = "./stage_history"):
        """
        Args:
            ticker: ティッカーシンボル
            data_dir: 履歴保存ディレクトリ
        """
        self.ticker = ticker
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.history_file = self.data_dir / f"{ticker}_stage_history.json"
        self.history = self._load_history()

    def _load_history(self) -> Dict:
        """履歴ファイルを読み込み"""
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                return json.load(f)
        return self._initialize_history()
    
    def _initialize_history(self) -> Dict:
        """履歴を初期化"""
        return {
            'ticker': self.ticker,
            'created_at': datetime.now().isoformat(),
            'current_stage': None,
            'stage_transitions': [],
            'last_updated': None,
            'statistics': {
                'total_transitions': 0,
                'stage1_to_stage2': 0,
                'stage2_to_stage3': 0,
                'stage3_to_stage2': 0,
                'stage3_to_stage4': 0,
                'stage4_to_stage1': 0,
                'other_transitions': 0
            }
        }
    
    def _save_history(self):
        """履歴を保存"""
        self.history['last_updated'] = datetime.now().isoformat()
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2, default=str)
    
    def analyze_and_update(self, daily_df: pd.DataFrame, benchmark_daily_df: pd.DataFrame, weekly_df: pd.DataFrame):
        """
        日足と週足データを用いてステージを分析・更新
        """
        latest_date = daily_df.index[-1]

        # --- 1. 週足ステージを判定 (長期トレンドフィルター) ---
        weekly_detector = StageDetector(weekly_df, interval='1wk')
        weekly_stage = weekly_detector.determine_stage()

        # --- 2. 日足ステージを判定 ---
        daily_detector = StageDetector(daily_df, benchmark_daily_df, interval='1d')
        rs_rating = daily_df.iloc[-1].get('RS_Rating')
        previous_stage = self.history.get('current_stage')

        # 早期シグナル検出
        early_signals = {}
        if previous_stage == 2:
            early_stage3, stage3_details = daily_detector._detect_early_stage3_signals()
            if early_stage3:
                early_signals['stage3_early_signal'] = {
                    'detected': True,
                    'strength': stage3_details.get('signal_strength', 0),
                    'reasons': stage3_details.get('reasons', []),
                    'confidence': stage3_details.get('confidence', 'unknown')
                }
        if previous_stage == 4:
            early_stage1, stage1_details = daily_detector._detect_early_stage1_signals()
            if early_stage1:
                early_signals['stage1_early_signal'] = {
                    'detected': True,
                    'strength': stage1_details.get('signal_strength', 0),
                    'reasons': stage1_details.get('reasons', []),
                    'confidence': stage1_details.get('confidence', 'unknown')
                }

        detected_stage_daily = daily_detector.determine_stage(
            rs_rating=rs_rating,
            previous_stage=previous_stage
        )

        # --- 3. 週足フィルターを適用した最終ステージ判定 ---
        final_stage = self._apply_weekly_filter(detected_stage_daily, weekly_stage, previous_stage)

        # --- 4. 履歴の更新 ---
        # 初回実行時
        if self.history['current_stage'] is None:
            self._initialize_stage(latest_date, final_stage)
            if early_signals:
                self.history['early_signals'] = early_signals
            self._save_history()
            return

        # ステージに変化があったか確認
        if final_stage != previous_stage:
            reason = f"週足 Stage {weekly_stage} のフィルター適用後、Stage {previous_stage} → {final_stage} へ移行"
            if early_signals:
                reason += f" (早期シグナル: {', '.join(early_signals.keys())})"

            self._execute_stage_transition(latest_date, final_stage, reason, early_signals)

        self._save_history()

    def _apply_weekly_filter(self, daily_stage: int, weekly_stage: int, previous_stage: Optional[int]) -> int:
        """
        週足ステージをフィルターとして使い、最終的なステージを決定する。
        """
        # ルール1: 週足が下降トレンド(S4)の場合、日足がS2になってもS1(底固め)として扱う
        if weekly_stage == 4 and daily_stage == 2:
            return 1  # 上昇のダマシと判断

        # ルール2: 週足が上昇トレンド(S2)の場合、日足がS4になってもS3(天井圏)として扱う
        if weekly_stage == 2 and daily_stage == 4:
            return 3  # 下降のダマシと判断

        # ルール3: 週足がS1またはS2でない限り、日足のS2移行は認めない
        if daily_stage == 2 and weekly_stage not in [1, 2]:
            if previous_stage:
                return previous_stage # ステージ移行を保留
            else:
                return 1 # 初期状態ならS1

        # 上記ルールに当てはまらない場合は、日足の判定を優先
        return daily_stage

    def _initialize_stage(self, date: pd.Timestamp, stage: int):
        """初期ステージを設定"""
        self.history.update({
            'current_stage': stage,
            'stage_start_date': date.isoformat()
        })
        self.history['stage_transitions'].append({
            'date': date.isoformat(),
            'from': None,
            'to': stage,
            'reason': 'Initial detection'
        })

    def _execute_stage_transition(self, date: pd.Timestamp, new_stage: int,
                                  reason: str, early_signals: Dict = None):
        """ステージ移行を実行（早期シグナル対応）"""
        old_stage = self.history['current_stage']

        # 安定化ロック
        if self.history['stage_transitions']:
            last_transition_date = pd.Timestamp(self.history['stage_transitions'][-1]['date'])
            if (date - last_transition_date).days < 5:
                return

        if old_stage == new_stage:
            return

        self.history.update({
            'current_stage': new_stage,
            'stage_start_date': date.isoformat()
        })

        transition = {
            'date': date.isoformat(),
            'from': old_stage,
            'to': new_stage,
            'reason': reason
        }

        # **【追加】早期シグナル情報を記録**
        if early_signals:
            transition['early_signals'] = early_signals

        self.history['stage_transitions'].append(transition)
        self._update_statistics(old_stage, new_stage)

    def _update_statistics(self, from_stage: int, to_stage: int):
        """統計情報を更新"""
        stats = self.history['statistics']
        stats['total_transitions'] += 1
        key = f"stage{from_stage}_to_stage{to_stage}"
        if key in stats:
            stats[key] += 1
        else:
            stats['other_transitions'] += 1

    def print_summary(self):
        """サマリーを表示"""
        status = self.history
        print("=" * 70)
        print(f"Stage History Summary: {status['ticker']}")
        print("=" * 70)
        print(f"Current Stage: {status.get('current_stage')}")
        if status.get('stage_transitions'):
            print(f"\n【Stage Transition History】")
            for t in status['stage_transitions']:
                from_stage = t.get('from', 'N/A')
                print(f"  - {t['date']}: Stage {from_stage} → {t['to']} - Reason: {t['reason']}")
        print(f"\n【Statistics】")
        stats = status['statistics']
        print(f"  Total Transitions: {stats['total_transitions']}")
        for key, value in stats.items():
            if "stage" in key:
                print(f"  {key.replace('_', ' ').replace('stage', 'Stage ')}: {value}")
        print(f"  Other: {stats['other_transitions']}")

    def load_from_json(self, ticker: str):
        """指定されたティッカーのJSONファイルを読み込む"""
        json_file_path = self.data_dir / f"{ticker}_stage_history.json"
        if json_file_path.exists():
            with open(json_file_path, 'r') as f:
                self.history = json.load(f)
        else:
            self.history = self._initialize_history()
            self.history['ticker'] = ticker

    def get_history_as_df(self) -> pd.DataFrame:
        """ステージ移行履歴をDataFrameとして取得"""
        transitions = self.history.get('stage_transitions', [])
        if not transitions:
            return pd.DataFrame()

        df = pd.DataFrame(transitions)
        df['ticker'] = self.history.get('ticker')
        return df[['ticker', 'date', 'from', 'to', 'reason']]
