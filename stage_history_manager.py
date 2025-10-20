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
    
    def analyze_and_update(self, df: pd.DataFrame, benchmark_df: pd.DataFrame):
        """
        データを分析してステージを更新（早期検出対応版）
        """
        latest_date = df.index[-1]

        # 1. StageDetectorで現在のステージを判定
        detector = StageDetector(df, benchmark_df)
        rs_rating = df.iloc[-1].get('RS_Rating')
        previous_stage = self.history.get('current_stage')

        # **【追加】早期シグナルの記録**
        early_signals = {}

        if previous_stage == 2:
            early_stage3, stage3_details = detector._detect_early_stage3_signals()
            if early_stage3:
                early_signals['stage3_early_signal'] = {
                    'detected': True,
                    'strength': stage3_details.get('signal_strength', 0),
                    'reasons': stage3_details.get('reasons', []),
                    'confidence': stage3_details.get('confidence', 'unknown')
                }

        if previous_stage == 4:
            early_stage1, stage1_details = detector._detect_early_stage1_signals()
            if early_stage1:
                early_signals['stage1_early_signal'] = {
                    'detected': True,
                    'strength': stage1_details.get('signal_strength', 0),
                    'reasons': stage1_details.get('reasons', []),
                    'confidence': stage1_details.get('confidence', 'unknown')
                }

        detected_stage = detector.determine_stage(
            rs_rating=rs_rating,
            previous_stage=previous_stage
        )

        # 2. 初回実行時の初期化
        if self.history['current_stage'] is None:
            self._initialize_stage(latest_date, detected_stage)
            if early_signals:
                self.history['early_signals'] = early_signals
            self._save_history()
            return

        # 3. ステージに変化があったか確認
        previous_stage = self.history['current_stage']
        if detected_stage != previous_stage:
            # **【追加】早期シグナル情報を含める**
            reason = f"Detected change from Stage {previous_stage} to {detected_stage}"
            if early_signals:
                reason += f" (Early signals: {', '.join(early_signals.keys())})"

            self._execute_stage_transition(latest_date, detected_stage, reason, early_signals)

        # **【追加】早期シグナルを履歴に記録**
        if early_signals:
            if 'early_signals_history' not in self.history:
                self.history['early_signals_history'] = []

            self.history['early_signals_history'].append({
                'date': latest_date.isoformat(),
                'current_stage': previous_stage,
                'signals': early_signals
            })

        self._save_history()

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
