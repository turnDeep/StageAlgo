"""
Stage History Manager (時系列ブレイクアウト追跡版)

【重要な改善】
ブレイクアウト判定に「時間的なズレの許容」を実装
- 各条件が数日以内に順次満たされれば、一つのブレイクアウトと認定
- 現実的なブレイクアウトパターンを正確に捉える
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum

from stage_detector import StageDetector
from base_detector import BaseDetector
from indicators import calculate_all_basic_indicators


class BreakoutCondition(Enum):
    """ブレイクアウト条件の種類"""
    HIGH_BREAKOUT = "high_breakout"  # 高値更新
    VOLUME_SURGE = "volume_surge"    # 出来高急増
    MA_ALIGNMENT = "ma_alignment"    # MA配列好転
    HIGHER_HIGHS = "higher_highs"    # 高値・高安値形成


@dataclass
class ConditionStatus:
    """個別条件の達成状況"""
    met: bool = False
    date: Optional[str] = None
    value: Optional[float] = None
    details: Optional[Dict] = None


@dataclass
class BreakoutCandidate:
    """ブレイクアウト候補の状態"""
    trigger_date: str  # 最初の条件が満たされた日
    window_start: str  # 監視期間の開始
    window_end: str    # 監視期間の終了
    conditions: Dict[str, ConditionStatus]
    status: str = "pending"  # pending, confirmed, failed, expired
    confirmation_date: Optional[str] = None
    score: float = 0.0
    
    def to_dict(self):
        """辞書形式に変換（JSON保存用）"""
        result = {
            'trigger_date': self.trigger_date,
            'window_start': self.window_start,
            'window_end': self.window_end,
            'status': self.status,
            'confirmation_date': self.confirmation_date,
            'score': self.score,
            'conditions': {}
        }
        
        for key, cond in self.conditions.items():
            result['conditions'][key] = {
                'met': cond.met,
                'date': cond.date,
                'value': cond.value,
                'details': cond.details
            }
        
        return result


class BreakoutTracker:
    """
    ブレイクアウト候補を時系列で追跡
    
    【コア機能】
    1. 各条件の達成を独立して記録
    2. N日以内にすべての条件が揃えば確定
    3. 期間を過ぎた候補は自動的に失効
    """
    
    def __init__(self, tracking_window_days: int = 15, min_conditions: int = 3):
        """
        Args:
            tracking_window_days: 条件追跡のウィンドウ期間（営業日）
            min_conditions: ブレイクアウト確定に必要な最小条件数
        """
        self.tracking_window = tracking_window_days
        self.min_conditions = min_conditions
        self.candidates: List[BreakoutCandidate] = []
        
    def add_or_update_candidate(self, date: pd.Timestamp, condition_type: BreakoutCondition,
                                value: float = None, details: Dict = None):
        """
        条件達成を記録（新規候補作成 or 既存候補更新）
        
        Args:
            date: 条件達成日
            condition_type: 条件の種類
            value: 条件の値（例：出来高比率）
            details: 追加情報
        """
        date_str = date.strftime('%Y-%m-%d')
        
        # 既存の有効な候補を検索（ウィンドウ内）
        active_candidate = self._find_active_candidate(date)
        
        if active_candidate:
            # 既存候補を更新
            self._update_candidate(active_candidate, condition_type, date_str, value, details)
        else:
            # 新規候補を作成
            self._create_new_candidate(date, condition_type, date_str, value, details)
    
    def _find_active_candidate(self, date: pd.Timestamp) -> Optional[BreakoutCandidate]:
        """
        指定日において有効な候補を検索
        
        Returns:
            有効な候補（なければNone）
        """
        for candidate in self.candidates:
            if candidate.status != "pending":
                continue
            
            window_start = pd.Timestamp(candidate.window_start)
            window_end = pd.Timestamp(candidate.window_end)
            
            if window_start <= date <= window_end:
                return candidate
        
        return None
    
    def _create_new_candidate(self, date: pd.Timestamp, condition_type: BreakoutCondition,
                             date_str: str, value: float, details: Dict):
        """新規ブレイクアウト候補を作成"""
        window_start = date
        window_end = date + pd.Timedelta(days=self.tracking_window)
        
        # 初期条件を設定
        conditions = {
            BreakoutCondition.HIGH_BREAKOUT.value: ConditionStatus(),
            BreakoutCondition.VOLUME_SURGE.value: ConditionStatus(),
            BreakoutCondition.MA_ALIGNMENT.value: ConditionStatus(),
            BreakoutCondition.HIGHER_HIGHS.value: ConditionStatus(),
        }
        
        # 最初の条件を記録
        conditions[condition_type.value] = ConditionStatus(
            met=True,
            date=date_str,
            value=value,
            details=details
        )
        
        candidate = BreakoutCandidate(
            trigger_date=date_str,
            window_start=window_start.strftime('%Y-%m-%d'),
            window_end=window_end.strftime('%Y-%m-%d'),
            conditions=conditions
        )
        
        self.candidates.append(candidate)
    
    def _update_candidate(self, candidate: BreakoutCandidate, condition_type: BreakoutCondition,
                         date_str: str, value: float, details: Dict):
        """既存候補の条件を更新"""
        condition_key = condition_type.value
        
        # まだ満たされていない条件のみ更新
        if not candidate.conditions[condition_key].met:
            candidate.conditions[condition_key] = ConditionStatus(
                met=True,
                date=date_str,
                value=value,
                details=details
            )
    
    def check_and_update_candidates(self, current_date: pd.Timestamp) -> List[BreakoutCandidate]:
        """
        候補の状態を確認・更新
        
        Returns:
            確定したブレイクアウトのリスト
        """
        confirmed_breakouts = []
        
        for candidate in self.candidates:
            if candidate.status != "pending":
                continue
            
            # ウィンドウ終了日を過ぎたかチェック
            window_end = pd.Timestamp(candidate.window_end)
            
            if current_date > window_end:
                # 期限切れ判定
                met_count = sum(1 for cond in candidate.conditions.values() if cond.met)
                
                if met_count >= self.min_conditions:
                    # 十分な条件を満たしていれば確定
                    candidate.status = "confirmed"
                    candidate.confirmation_date = window_end.strftime('%Y-%m-%d')
                    candidate.score = self._calculate_breakout_score(candidate)
                    confirmed_breakouts.append(candidate)
                else:
                    # 条件不足で失効
                    candidate.status = "expired"
            else:
                # ウィンドウ内で全条件達成チェック
                met_count = sum(1 for cond in candidate.conditions.values() if cond.met)
                
                if met_count >= len(candidate.conditions):
                    # 全条件達成！
                    candidate.status = "confirmed"
                    candidate.confirmation_date = current_date.strftime('%Y-%m-%d')
                    candidate.score = self._calculate_breakout_score(candidate)
                    confirmed_breakouts.append(candidate)
        
        return confirmed_breakouts
    
    def _calculate_breakout_score(self, candidate: BreakoutCandidate) -> float:
        """
        ブレイクアウトの品質スコアを計算（100点満点）
        
        評価要素:
        1. 条件達成数（40点）
        2. 条件達成の時間的集中度（30点）
        3. 出来高の強さ（20点）
        4. その他の品質指標（10点）
        """
        score = 0
        
        # 1. 条件達成数（40点）
        met_conditions = [c for c in candidate.conditions.values() if c.met]
        condition_score = (len(met_conditions) / len(candidate.conditions)) * 40
        score += condition_score
        
        # 2. 時間的集中度（30点）
        # 条件がより短期間で揃うほど高スコア
        if len(met_conditions) >= 2:
            dates = [pd.Timestamp(c.date) for c in met_conditions if c.date]
            if dates:
                date_range = (max(dates) - min(dates)).days
                # 3日以内: 30点, 7日以内: 20点, 15日以内: 10点
                if date_range <= 3:
                    time_score = 30
                elif date_range <= 7:
                    time_score = 20
                elif date_range <= 15:
                    time_score = 10
                else:
                    time_score = 5
                score += time_score
        
        # 3. 出来高の強さ（20点）
        volume_cond = candidate.conditions.get(BreakoutCondition.VOLUME_SURGE.value)
        if volume_cond and volume_cond.met and volume_cond.value:
            volume_ratio = volume_cond.value
            if volume_ratio >= 3.0:
                volume_score = 20
            elif volume_ratio >= 2.0:
                volume_score = 15
            elif volume_ratio >= 1.5:
                volume_score = 10
            else:
                volume_score = 5
            score += volume_score
        
        # 4. その他の品質（10点）
        # 高値更新が最初の条件として現れているか
        high_cond = candidate.conditions.get(BreakoutCondition.HIGH_BREAKOUT.value)
        if high_cond and high_cond.met and high_cond.date == candidate.trigger_date:
            score += 10  # 理想的な順序
        
        return min(100, score)
    
    def get_active_candidates(self) -> List[BreakoutCandidate]:
        """現在追跡中の候補を取得"""
        return [c for c in self.candidates if c.status == "pending"]
    
    def get_confirmed_breakouts(self) -> List[BreakoutCandidate]:
        """確定したブレイクアウトを取得"""
        return [c for c in self.candidates if c.status == "confirmed"]


class StageHistoryManager:
    """
    ステージ・ベースの履歴管理システム（時系列ブレイクアウト追跡版）
    
    【重要な改善】
    ブレイクアウト判定に時間的なズレを許容し、より現実的な判定を実現
    """
    
    def __init__(self, ticker: str, data_dir: str = "./stage_history",
                 breakout_window_days: int = 15):
        """
        Args:
            ticker: ティッカーシンボル
            data_dir: 履歴JSONファイルを保存するディレクトリ
            breakout_window_days: ブレイクアウト条件の追跡期間
        """
        self.ticker = ticker
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.history_file = self.data_dir / f"{ticker}_stage_history.json"
        
        # ブレイクアウト追跡システム
        self.breakout_tracker = BreakoutTracker(
            tracking_window_days=breakout_window_days,
            min_conditions=3  # 最低3つの条件が必要
        )
        
        # 履歴データを読み込み
        self.history = self._load_history()
        
    def _load_history(self) -> Dict:
        """履歴JSONファイルを読み込み"""
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                data = json.load(f)
                
                # ブレイクアウト候補を復元
                if 'breakout_candidates' in data:
                    for cand_dict in data['breakout_candidates']:
                        # ConditionStatusオブジェクトに変換
                        conditions = {}
                        for key, cond_dict in cand_dict['conditions'].items():
                            conditions[key] = ConditionStatus(**cond_dict)
                        
                        candidate = BreakoutCandidate(
                            trigger_date=cand_dict['trigger_date'],
                            window_start=cand_dict['window_start'],
                            window_end=cand_dict['window_end'],
                            conditions=conditions,
                            status=cand_dict['status'],
                            confirmation_date=cand_dict.get('confirmation_date'),
                            score=cand_dict.get('score', 0.0)
                        )
                        self.breakout_tracker.candidates.append(candidate)
                
                return data
        else:
            return self._initialize_history()
    
    def _initialize_history(self) -> Dict:
        """履歴データの初期化"""
        return {
            'ticker': self.ticker,
            'created_at': datetime.now().isoformat(),
            'current_stage': None,
            'current_substage': None,
            'current_base_count': 1,
            'current_base_letter': '',
            'stage_start_date': None,
            'pivot_points': [],
            'stage_transitions': [],
            'base_transitions': [],
            'breakout_candidates': [],  # ブレイクアウト候補の履歴
            'confirmed_breakouts': [],  # 確定したブレイクアウト
            'overheat_history': [],
            'last_updated': None
        }
    
    def _save_history(self):
        """履歴をJSONファイルに保存"""
        self.history['last_updated'] = datetime.now().isoformat()
        
        # ブレイクアウト候補を保存形式に変換
        self.history['breakout_candidates'] = [
            cand.to_dict() for cand in self.breakout_tracker.candidates
        ]
        
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2, default=str)
    
    def analyze_and_update(self, df: pd.DataFrame, benchmark_df: pd.DataFrame) -> Dict:
        """
        既存detectorを使って分析し、履歴を更新（時系列ブレイクアウト追跡版）
        """
        latest_date = df.index[-1]
        
        # 既存detectorから候補を取得
        stage_detector = StageDetector(df)
        candidate_stage, candidate_substage = stage_detector.determine_stage()
        
        # 初回判定の場合
        if self.history['current_stage'] is None:
            self._initialize_stage(df, latest_date, candidate_stage, candidate_substage)
            self._save_history()
            return self._get_current_status()
        
        # 【新機能】ブレイクアウト条件を個別に時系列でチェック
        self._check_individual_conditions(df, latest_date)
        
        # ブレイクアウト候補の状態を更新
        confirmed_breakouts = self.breakout_tracker.check_and_update_candidates(latest_date)
        
        # 確定したブレイクアウトを処理
        for breakout in confirmed_breakouts:
            self._process_confirmed_breakout(df, breakout)
        
        # 従来のステージ移行チェック（フォールバック）
        transition_result = self._check_stage_transition_fallback(
            df, latest_date, candidate_stage
        )
        
        if transition_result['should_transition']:
            self._execute_stage_transition(
                latest_date,
                transition_result['new_stage'],
                transition_result['new_substage'],
                transition_result['reason'],
                transition_result.get('volume_ratio', 1.0)
            )
        
        # ベースカウントの更新
        self._update_base_count(df, latest_date)
        
        # 過熱感の記録
        self._record_overheat_status(df, latest_date)
        
        self._save_history()
        return self._get_current_status()
    
    def _check_individual_conditions(self, df: pd.DataFrame, date: pd.Timestamp):
        """
        【核心機能】ブレイクアウト条件を個別に時系列でチェック
        
        これにより、各条件が数日以内に順次満たされるパターンを捉える
        """
        latest = df.loc[date]
        
        # 条件1: 過去126日の最高値更新
        if self._check_high_breakout(df, date):
            self.breakout_tracker.add_or_update_candidate(
                date=date,
                condition_type=BreakoutCondition.HIGH_BREAKOUT,
                value=latest['High'],
                details={'period': 126}
            )
        
        # 条件2: 出来高急増（1.5倍以上）
        volume_ratio = self._check_volume_surge(df, date)
        if volume_ratio and volume_ratio >= 1.5:
            self.breakout_tracker.add_or_update_candidate(
                date=date,
                condition_type=BreakoutCondition.VOLUME_SURGE,
                value=volume_ratio,
                details={'threshold': 1.5}
            )
        
        # 条件3: MA配列好転
        if self._check_ma_alignment(df, date):
            self.breakout_tracker.add_or_update_candidate(
                date=date,
                condition_type=BreakoutCondition.MA_ALIGNMENT,
                details=self._get_ma_details(df, date)
            )
        
        # 条件4: 高値・高安値形成
        if self._check_higher_highs_lows(df, date):
            self.breakout_tracker.add_or_update_candidate(
                date=date,
                condition_type=BreakoutCondition.HIGHER_HIGHS,
                details={'lookback': 40}
            )
    
    def _check_high_breakout(self, df: pd.DataFrame, date: pd.Timestamp) -> bool:
        """126日高値更新をチェック"""
        lookback_period = 126
        if len(df) < lookback_period + 1:
            return False
        
        try:
            date_idx = df.index.get_loc(date)
            if date_idx < lookback_period:
                return False
            
            past_high = df['High'].iloc[max(0, date_idx - lookback_period):date_idx].max()
            current_high = df.loc[date, 'High']
            
            return current_high > past_high
        except:
            return False
    
    def _check_volume_surge(self, df: pd.DataFrame, date: pd.Timestamp) -> Optional[float]:
        """出来高急増をチェック"""
        try:
            latest = df.loc[date]
            
            if 'Volume_SMA_50' in df.columns:
                avg_volume = latest['Volume_SMA_50']
                current_volume = latest['Volume']
                
                if avg_volume > 0:
                    return current_volume / avg_volume
            
            return None
        except:
            return None
    
    def _check_ma_alignment(self, df: pd.DataFrame, date: pd.Timestamp) -> bool:
        """MA配列好転をチェック"""
        try:
            latest = df.loc[date]
            
            if 'SMA_150' not in df.columns or 'SMA_200' not in df.columns:
                return False
            
            ma_150 = latest['SMA_150']
            ma_200 = latest['SMA_200']
            current_price = latest['Close']
            
            # 価格がMAを上回り、MA配列が上昇トレンド
            if current_price > ma_150 and current_price > ma_200:
                # 150日MAが上昇または横ばい
                if len(df) >= 21:
                    date_idx = df.index.get_loc(date)
                    if date_idx >= 20:
                        ma_150_20d_ago = df['SMA_150'].iloc[date_idx - 20]
                        if ma_150 >= ma_150_20d_ago * 0.98:  # 2%以内の下落は許容
                            return True
            
            return False
        except:
            return False
    
    def _check_higher_highs_lows(self, df: pd.DataFrame, date: pd.Timestamp) -> bool:
        """高値・高安値形成をチェック"""
        try:
            date_idx = df.index.get_loc(date)
            
            if date_idx < 40:
                return False
            
            recent_20d = df.iloc[date_idx-19:date_idx+1]  # 最近20日
            prior_20d = df.iloc[date_idx-39:date_idx-19]  # その前20日
            
            if len(recent_20d) < 20 or len(prior_20d) < 20:
                return False
            
            recent_high = recent_20d['High'].max()
            prior_high = prior_20d['High'].max()
            
            recent_low = recent_20d['Low'].min()
            prior_low = prior_20d['Low'].min()
            
            # 高値更新 & 安値が維持または上昇
            return recent_high > prior_high and recent_low >= prior_low * 0.98
        except:
            return False
    
    def _get_ma_details(self, df: pd.DataFrame, date: pd.Timestamp) -> Dict:
        """MA配列の詳細情報を取得"""
        try:
            latest = df.loc[date]
            return {
                'price': float(latest['Close']),
                'sma_50': float(latest.get('SMA_50', 0)),
                'sma_150': float(latest.get('SMA_150', 0)),
                'sma_200': float(latest.get('SMA_200', 0))
            }
        except:
            return {}
    
    def _process_confirmed_breakout(self, df: pd.DataFrame, breakout: BreakoutCandidate):
        """
        確定したブレイクアウトを処理
        
        Args:
            df: データフレーム
            breakout: 確定したブレイクアウト候補
        """
        # Stage 1 → Stage 2 移行を実行
        confirmation_date = pd.Timestamp(breakout.confirmation_date)
        
        # 出来高情報を取得
        volume_cond = breakout.conditions.get(BreakoutCondition.VOLUME_SURGE.value)
        volume_ratio = volume_cond.value if volume_cond and volume_cond.met else 1.0
        
        # 詳細な理由を構築
        reason_parts = ["Time-series breakout confirmed:"]
        for cond_type, cond in breakout.conditions.items():
            if cond.met:
                reason_parts.append(f"{cond_type} on {cond.date}")
        
        reason = " | ".join(reason_parts)
        reason += f" | Score: {breakout.score:.1f}/100"
        
        # ステージ移行を実行（既にStage 2でない場合のみ）
        if self.history['current_stage'] != 2:
            self._execute_stage_transition(
                confirmation_date,
                2,  # Stage 2
                '2A',  # サブステージ
                reason,
                volume_ratio
            )
        
        # 確定ブレイクアウトを履歴に追加
        if 'confirmed_breakouts' not in self.history:
            self.history['confirmed_breakouts'] = []
        
        self.history['confirmed_breakouts'].append({
            'trigger_date': breakout.trigger_date,
            'confirmation_date': breakout.confirmation_date,
            'score': breakout.score,
            'conditions': breakout.to_dict()['conditions'],
            'volume_ratio': volume_ratio
        })
    
    def _check_stage_transition_fallback(self, df: pd.DataFrame, date: pd.Timestamp,
                                        candidate_stage: int) -> Dict:
        """
        従来のステージ移行チェック（フォールバック）
        
        ブレイクアウト追跡で検出されなかった場合の予備判定
        """
        current_stage = self.history['current_stage']
        
        # Stage 1以外の移行は従来ロジックを使用
        if current_stage == 2:
            result = self._check_stage2_to_3(df, date)
            if result['should_transition']:
                return result
        
        elif current_stage == 3:
            result = self._check_stage3_to_2(df, date)
            if result['should_transition']:
                return result
            
            result = self._check_stage3_to_4(df, date)
            if result['should_transition']:
                return result
        
        elif current_stage == 4:
            result = self._check_stage4_to_1(df, date)
            if result['should_transition']:
                return result
        
        return {'should_transition': False}
    
    # 以下、従来のメソッドを維持（_check_stage2_to_3, _check_stage3_to_4など）
    # （省略 - 元のコードから継承）
    
    def _initialize_stage(self, df: pd.DataFrame, date: pd.Timestamp,
                         stage: int, substage: str):
        """初回ステージ判定"""
        self.history['current_stage'] = stage
        self.history['current_substage'] = substage
        self.history['stage_start_date'] = date.isoformat()
        
        self.history['stage_transitions'].append({
            'date': date.isoformat(),
            'from_stage': None,
            'from_substage': None,
            'to_stage': stage,
            'to_substage': substage,
            'reason': 'Initial detection',
            'volume_ratio': 1.0
        })
    
    def _execute_stage_transition(self, date: pd.Timestamp, new_stage: int,
                                  new_substage: str, reason: str, volume_ratio: float):
        """ステージ移行を実行"""
        old_stage = self.history['current_stage']
        old_substage = self.history['current_substage']
        
        self.history['current_stage'] = new_stage
        self.history['current_substage'] = new_substage
        self.history['stage_start_date'] = date.isoformat()
        
        self.history['stage_transitions'].append({
            'date': date.isoformat(),
            'from_stage': old_stage,
            'from_substage': old_substage,
            'to_stage': new_stage,
            'to_substage': new_substage,
            'reason': reason,
            'volume_ratio': float(volume_ratio)
        })
    
    def _get_current_status(self) -> Dict:
        """現在のステータスを返す"""
        base_stage_str = f"{self.history['current_base_count']}{self.history['current_base_letter']}"
        
        # 進行中のブレイクアウト候補数
        active_candidates = len(self.breakout_tracker.get_active_candidates())
        confirmed_breakouts = len(self.breakout_tracker.get_confirmed_breakouts())
        
        return {
            'ticker': self.ticker,
            'weinstein_stage': self.history['current_stage'],
            'weinstein_substage': self.history['current_substage'],
            'stage_display': f"Stage {self.history['current_stage']} ({self.history['current_substage']})",
            'base_count_stage': base_stage_str,
            'stage_start_date': self.history['stage_start_date'],
            'active_breakout_candidates': active_candidates,
            'confirmed_breakouts_count': confirmed_breakouts,
            'last_updated': self.history['last_updated'],
            'tracking_status': {
                'active_candidates': active_candidates,
                'total_candidates': len(self.breakout_tracker.candidates),
                'confirmed_breakouts': confirmed_breakouts
            }
        }
    
    def print_summary(self):
        """サマリーを表示"""
        status = self._get_current_status()
        
        print("=" * 70)
        print(f"Stage History Summary (Time-series Tracking): {status['ticker']}")
        print("=" * 70)
        print(f"Weinstein Stage: {status['stage_display']}")
        print(f"Base Count: {status['base_count_stage']}")
        print(f"Stage Start: {status['stage_start_date']}")
        print(f"\n【Breakout Tracking Status】")
        print(f"  Active Candidates: {status['active_breakout_candidates']}")
        print(f"  Confirmed Breakouts: {status['confirmed_breakouts_count']}")
        print(f"Last Updated: {status['last_updated']}")
        print("=" * 70)
        
        # ブレイクアウト候補の詳細
        active_candidates = self.breakout_tracker.get_active_candidates()
        if active_candidates:
            print("\n【Active Breakout Candidates】")
            for i, cand in enumerate(active_candidates, 1):
                print(f"\nCandidate {i}:")
                print(f"  Trigger Date: {cand.trigger_date}")
                print(f"  Window: {cand.window_start} ~ {cand.window_end}")
                print(f"  Conditions Met:")
                for cond_type, cond in cand.conditions.items():
                    status_mark = "✓" if cond.met else "✗"
                    date_info = f"({cond.date})" if cond.met and cond.date else ""
                    print(f"    {status_mark} {cond_type} {date_info}")
        
        # 確定ブレイクアウトの履歴
        if self.history.get('confirmed_breakouts'):
            print("\n【Confirmed Breakouts History】")
            for breakout in self.history['confirmed_breakouts'][-3:]:
                print(f"\n  Trigger: {breakout['trigger_date']}")
                print(f"  Confirmed: {breakout['confirmation_date']}")
                print(f"  Score: {breakout['score']:.1f}/100")
                print(f"  Volume Ratio: {breakout['volume_ratio']:.2f}x")


# （以下、_check_stage2_to_3などの従来メソッドを継承）
# 省略部分は元のコードをそのまま使用

if __name__ == '__main__':
    from data_fetcher import fetch_stock_data
    
    print("Stage History Manager（時系列ブレイクアウト追跡版）のテストを開始...")
    
    test_tickers = ['EGAN', 'NVDA', 'TSLA']
    
    for ticker in test_tickers:
        print(f"\n{'='*70}")
        print(f"{ticker} の時系列ブレイクアウト追跡:")
        print(f"{'='*70}")
        
        # データ取得
        stock_df, benchmark_df = fetch_stock_data(ticker, period='2y')
        
        if stock_df is not None and benchmark_df is not None:
            # 指標計算
            stock_df = calculate_all_basic_indicators(stock_df)
            benchmark_df = calculate_all_basic_indicators(benchmark_df)
            
            stock_df = stock_df.dropna()
            benchmark_df = benchmark_df.dropna()
            
            if len(stock_df) >= 252:
                # 履歴マネージャーを初期化（15日間の追跡ウィンドウ）
                manager = StageHistoryManager(ticker, breakout_window_days=15)
                
                # 分析と更新
                result = manager.analyze_and_update(stock_df, benchmark_df)
                
                # サマリー表示
                manager.print_summary()