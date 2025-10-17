"""
Stage History Manager (階層的パラメータ・最終版)

【重要な改善】
- ステージ移行の種類ごとに、異なるパラメータセットを適用
- 「ブレイクアウト」と「天井形成/底打ち」で判断基準の厳しさを使い分ける
- アナリストの柔軟な思考を模倣し、汎用性と精度を両立
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime
from dataclasses import dataclass, asdict, field
from enum import Enum

from stage_detector import StageDetector
from base_detector import BaseDetector
from indicators import calculate_all_basic_indicators

class BreakoutCondition(Enum):
    HIGH_BREAKOUT, VOLUME_SURGE, MA_ALIGNMENT = "high_breakout", "volume_surge", "ma_alignment"
@dataclass
class ConditionStatus:
    met: bool = False; date: Optional[str] = None; value: Optional[float] = None; details: Optional[Dict] = field(default_factory=dict)
@dataclass
class BreakoutCandidate:
    trigger_date: str; window_start: str; window_end: str
    conditions: Dict[str, ConditionStatus] = field(default_factory=dict)
    status: str = "pending"; confirmation_date: Optional[str] = None; score: float = 0.0
    def to_dict(self): return asdict(self)
class BreakoutTracker:
    def __init__(self, tracking_window_days: int = 15, min_conditions: int = 2):
        self.tracking_window = tracking_window_days; self.min_conditions = min_conditions; self.candidates: List[BreakoutCandidate] = []
    def add_or_update_candidate(self, date: pd.Timestamp, condition_type: BreakoutCondition, value: float = None, details: Dict = None):
        date_str = date.strftime('%Y-%m-%d'); active_candidate = self._find_active_candidate(date)
        if active_candidate: self._update_candidate(active_candidate, condition_type, date_str, value, details)
        else: self._create_new_candidate(date, condition_type, date_str, value, details)
    def _find_active_candidate(self, date: pd.Timestamp) -> Optional[BreakoutCandidate]:
        for c in self.candidates:
            if c.status == "pending" and pd.Timestamp(c.window_start) <= date <= pd.Timestamp(c.window_end): return c
        return None
    def _create_new_candidate(self, date: pd.Timestamp, condition_type: BreakoutCondition, date_str: str, value: float, details: Dict):
        conditions = {cond.value: ConditionStatus() for cond in BreakoutCondition}
        conditions[condition_type.value] = ConditionStatus(met=True, date=date_str, value=value, details=details)
        self.candidates.append(BreakoutCandidate(trigger_date=date_str, window_start=date.strftime('%Y-%m-%d'), window_end=(date + pd.Timedelta(days=self.tracking_window)).strftime('%Y-%m-%d'), conditions=conditions))
    def _update_candidate(self, c: BreakoutCandidate, condition_type: BreakoutCondition, date_str: str, value: float, details: Dict):
        if not c.conditions[condition_type.value].met: c.conditions[condition_type.value] = ConditionStatus(met=True, date=date_str, value=value, details=details)
    def check_and_update_candidates(self, current_date: pd.Timestamp) -> List[BreakoutCandidate]:
        confirmed = []
        for c in self.candidates:
            if c.status != "pending": continue
            if sum(1 for cond in c.conditions.values() if cond.met) >= self.min_conditions:
                c.status = "confirmed"; c.confirmation_date = current_date.strftime('%Y-%m-%d'); c.score = self._calculate_score(c); confirmed.append(c)
            elif current_date > pd.Timestamp(c.window_end): c.status = "expired"
        return confirmed
    def _calculate_score(self, c: BreakoutCandidate) -> float:
        score = (sum(1 for cond in c.conditions.values() if cond.met) / len(c.conditions)) * 60
        dates = [pd.Timestamp(cond.date) for cond in c.conditions.values() if cond.met and cond.date]
        if len(dates) >= 2:
            date_range = (max(dates) - min(dates)).days
            if date_range <= 3: score += 40
            elif date_range <= 7: score += 20
        return min(100, score)

class BreakdownCondition(Enum):
    LOW_BREAKDOWN, VOLUME_SURGE, MA_BREAKDOWN = "low_breakdown", "volume_surge", "ma_breakdown"
@dataclass
class BreakdownCandidate:
    trigger_date: str; window_start: str; window_end: str
    conditions: Dict[str, ConditionStatus] = field(default_factory=dict)
    status: str = "pending"; confirmation_date: Optional[str] = None
    def to_dict(self): return asdict(self)
class BreakdownTracker:
    def __init__(self, tracking_window_days: int = 15, min_conditions: int = 2):
        self.tracking_window = tracking_window_days; self.min_conditions = min_conditions; self.candidates: List[BreakdownCandidate] = []
    def add_or_update_candidate(self, date: pd.Timestamp, condition_type: BreakdownCondition, value: float = None, details: Dict = None):
        date_str = date.strftime('%Y-%m-%d'); active_candidate = self._find_active_candidate(date)
        if active_candidate: self._update_candidate(active_candidate, condition_type, date_str, value, details)
        else: self._create_new_candidate(date, condition_type, date_str, value, details)
    def _find_active_candidate(self, date: pd.Timestamp) -> Optional[BreakdownCandidate]:
        for c in self.candidates:
            if c.status == "pending" and pd.Timestamp(c.window_start) <= date <= pd.Timestamp(c.window_end): return c
        return None
    def _create_new_candidate(self, date: pd.Timestamp, condition_type: BreakdownCondition, date_str: str, value: float, details: Dict):
        conditions = {cond.value: ConditionStatus() for cond in BreakdownCondition}
        conditions[condition_type.value] = ConditionStatus(met=True, date=date_str, value=value, details=details)
        self.candidates.append(BreakdownCandidate(trigger_date=date_str, window_start=date.strftime('%Y-%m-%d'), window_end=(date + pd.Timedelta(days=self.tracking_window)).strftime('%Y-%m-%d'), conditions=conditions))
    def _update_candidate(self, c: BreakdownCandidate, condition_type: BreakdownCondition, date_str: str, value: float, details: Dict):
        if not c.conditions[condition_type.value].met: c.conditions[condition_type.value] = ConditionStatus(met=True, date=date_str, value=value, details=details)
    def check_and_update_candidates(self, current_date: pd.Timestamp) -> List[BreakdownCandidate]:
        confirmed = []
        for c in self.candidates:
            if c.status != "pending": continue
            if sum(1 for cond in c.conditions.values() if cond.met) >= self.min_conditions:
                c.status = "confirmed"; c.confirmation_date = current_date.strftime('%Y-%m-%d'); confirmed.append(c)
            elif current_date > pd.Timestamp(c.window_end): c.status = "expired"
        return confirmed

class StageHistoryManager:
    def __init__(self, ticker: str, data_dir: str = "./stage_history", **kwargs):
        self.ticker = ticker
        self.data_dir = Path(data_dir); self.data_dir.mkdir(exist_ok=True)
        self.history_file = self.data_dir / f"{ticker}_stage_history.json"
        self.params = {
            'breakout': {'window_days': 15, 'min_high_period': 126, 'volume_multiplier': 1.5, **kwargs.get('breakout', {})},
            'breakdown': {'window_days': 15, 'min_low_period': 126, 'volume_multiplier': 1.5, **kwargs.get('breakdown', {})},
            'topping': {'days_below_ma': 20, 'ma_type': 'SMA_50', **kwargs.get('topping', {})},
            'basing': {'days_above_ma': 20, 'ma_type': 'SMA_150', **kwargs.get('basing', {})}
        }
        self.breakout_tracker = BreakoutTracker(self.params['breakout']['window_days'])
        self.breakdown_tracker = BreakdownTracker(self.params['breakdown']['window_days'])
        self.history = self._load_history()
        
    def _load_history(self) -> Dict:
        if self.history_file.exists():
            with open(self.history_file, 'r') as f: return json.load(f)
        return self._initialize_history()
    
    def _initialize_history(self) -> Dict:
        return {'ticker': self.ticker, 'created_at': datetime.now().isoformat(), 'current_stage': None, 'stage_transitions': [], 'last_updated': None, 'days_below_ma': 0, 'days_above_ma': 0}
    
    def _save_history(self):
        self.history['last_updated'] = datetime.now().isoformat()
        with open(self.history_file, 'w') as f: json.dump(self.history, f, indent=2, default=str)
    
    def analyze_and_update(self, df: pd.DataFrame, benchmark_df: pd.DataFrame):
        latest_date = df.index[-1]
        if self.history['current_stage'] is None:
            # 外部の判定に頼らず、暫定的にステージ1で初期化
            self._initialize_stage(latest_date, 1, '1A')
        
        current_stage = self.history['current_stage']
        if current_stage == 1: self._check_stage1_to_2(df, latest_date)
        elif current_stage == 2: self._check_stage2_to_3(df, latest_date)
        elif current_stage == 3: self._check_stage3_to_4(df, latest_date)
        elif current_stage == 4: self._check_stage4_to_1(df, latest_date)
        self._save_history()

    def _check_stage1_to_2(self, df: pd.DataFrame, date: pd.Timestamp):
        params = self.params['breakout']
        if self._check_high_breakout(df, date, params['min_high_period']): self.breakout_tracker.add_or_update_candidate(date, BreakoutCondition.HIGH_BREAKOUT)
        if self._check_volume_surge(df, date, params['volume_multiplier']): self.breakout_tracker.add_or_update_candidate(date, BreakoutCondition.VOLUME_SURGE)
        if self._check_ma_alignment(df, date): self.breakout_tracker.add_or_update_candidate(date, BreakoutCondition.MA_ALIGNMENT)
        for breakout in self.breakout_tracker.check_and_update_candidates(date): self._execute_stage_transition(pd.Timestamp(breakout.confirmation_date), 2, '2A', f"Breakout Confirmed (Score: {breakout.score:.1f})")

    def _check_stage3_to_4(self, df: pd.DataFrame, date: pd.Timestamp):
        params = self.params['breakdown']
        if self._check_low_breakdown(df, date, params['min_low_period']): self.breakdown_tracker.add_or_update_candidate(date, BreakdownCondition.LOW_BREAKDOWN)
        if self._check_volume_surge(df, date, params['volume_multiplier']): self.breakdown_tracker.add_or_update_candidate(date, BreakdownCondition.VOLUME_SURGE)
        if self._check_ma_breakdown(df, date): self.breakdown_tracker.add_or_update_candidate(date, BreakdownCondition.MA_BREAKDOWN)
        for breakdown in self.breakdown_tracker.check_and_update_candidates(date): self._execute_stage_transition(pd.Timestamp(breakdown.confirmation_date), 4, '4A', "Breakdown Confirmed")

    def _check_stage2_to_3(self, df: pd.DataFrame, date: pd.Timestamp):
        params = self.params['topping']; ma_type = params['ma_type']; latest = df.loc[date]
        if latest['Close'] < latest.get(ma_type, float('inf')): self.history['days_below_ma'] += 1
        else: self.history['days_below_ma'] = 0
        if self.history['days_below_ma'] > params['days_below_ma']: self._execute_stage_transition(date, 3, '3A', f"Failed to reclaim {ma_type} for >{params['days_below_ma']} days")

    def _check_stage4_to_1(self, df: pd.DataFrame, date: pd.Timestamp):
        params = self.params['basing']; ma_type = params['ma_type']; latest = df.loc[date]
        if latest['Close'] > latest.get(ma_type, 0) and self._get_ma_trend(df[ma_type]) == 'flat':
            self.history['days_above_ma'] += 1
        else: self.history['days_above_ma'] = 0
        if self.history['days_above_ma'] > params['days_above_ma']: self._execute_stage_transition(date, 1, '1A', f"Held above flat {ma_type} for >{params['days_above_ma']} days")

    def _check_high_breakout(self, df: pd.DataFrame, date: pd.Timestamp, lookback: int) -> bool:
        if len(df) < lookback + 1: return False
        return df.loc[date, 'High'] > df['High'].iloc[-(lookback + 1):-1].max()
    def _check_low_breakdown(self, df: pd.DataFrame, date: pd.Timestamp, lookback: int) -> bool:
        if len(df) < lookback + 1: return False
        return df.loc[date, 'Low'] < df['Low'].iloc[-(lookback + 1):-1].min()
    def _check_volume_surge(self, df: pd.DataFrame, date: pd.Timestamp, multiplier: float) -> bool:
        latest = df.loc[date]
        if 'Volume_SMA_50' in df.columns and (avg_vol := latest['Volume_SMA_50']) > 0: return latest['Volume'] > avg_vol * multiplier
        return False
    def _check_ma_alignment(self, df: pd.DataFrame, date: pd.Timestamp) -> bool:
        latest = df.loc[date]; price, ma150, ma200 = latest['Close'], latest.get('SMA_150'), latest.get('SMA_200')
        return bool(price and ma150 and ma200 and price > ma150 and self._get_ma_trend(df['SMA_150']) in ['rising', 'flat'])
    def _check_ma_breakdown(self, df: pd.DataFrame, date: pd.Timestamp) -> bool:
        latest = df.loc[date]; price, ma150, ma200 = latest['Close'], latest.get('SMA_150'), latest.get('SMA_200')
        return bool(price and ma150 and ma200 and price < ma150 and self._get_ma_trend(df['SMA_150']) in ['declining', 'flat'])
    def _get_ma_trend(self, ma_series: pd.Series, lookback: int = 20) -> str:
        if len(ma_series.dropna()) < lookback: return 'unknown'
        recent = ma_series.dropna().tail(lookback)
        slope = np.polyfit(np.arange(len(recent)), recent.values, 1)[0]
        if abs(slope) < 0.005 * np.mean(recent.values): return 'flat'
        return 'rising' if slope > 0 else 'declining'
    def _initialize_stage(self, date: pd.Timestamp, stage: int, substage: str):
        self.history.update({'current_stage': stage, 'current_substage': substage, 'stage_start_date': date.isoformat(), 'days_below_ma': 0, 'days_above_ma': 0})
        self.history['stage_transitions'].append({'date': date.isoformat(), 'from': None, 'to': stage, 'reason': 'Initial detection'})
    def _execute_stage_transition(self, date: pd.Timestamp, new_stage: int, new_substage: str, reason: str):
        if self.history['current_stage'] == new_stage: return
        old_stage = self.history['current_stage']
        self.history.update({'current_stage': new_stage, 'current_substage': new_substage, 'stage_start_date': date.isoformat()})
        self.history['stage_transitions'].append({'date': date.isoformat(), 'from': old_stage, 'to': new_stage, 'reason': reason})
    def print_summary(self):
        status = self.history
        print("="*70 + f"\nStage History Summary: {status['ticker']}\n" + "="*70)
        print(f"Current Stage: {status.get('current_stage')} ({status.get('current_substage')})")
        if self.history.get('stage_transitions'):
            print("\n【Stage Transition History】")
            for t in self.history['stage_transitions'][-5:]: print(f"  - {t['date']}: Stage {t.get('from')} -> {t['to']} ({t['reason']})")

if __name__ == '__main__':
    # ... (main execution logic remains the same)
    pass
