"""
Stage History Manager (時系列ブレイクアウト追跡・完全ライフサイクル版)

【重要な改善】
ブレイクアウト判定に加え、ステージ2以降の移行（天井形成、下落）も網羅的に追跡
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
    HIGH_BREAKOUT, VOLUME_SURGE, MA_ALIGNMENT, HIGHER_HIGHS = "high_breakout", "volume_surge", "ma_alignment", "higher_highs"

@dataclass
class ConditionStatus:
    met: bool = False
    date: Optional[str] = None
    value: Optional[float] = None
    details: Optional[Dict] = field(default_factory=dict)

@dataclass
class BreakoutCandidate:
    trigger_date: str
    window_start: str
    window_end: str
    conditions: Dict[str, ConditionStatus] = field(default_factory=dict)
    status: str = "pending"
    confirmation_date: Optional[str] = None
    score: float = 0.0
    def to_dict(self): return asdict(self)


class BreakoutTracker:
    def __init__(self, tracking_window_days: int = 15, min_conditions: int = 3):
        self.tracking_window = tracking_window_days
        self.min_conditions = min_conditions
        self.candidates: List[BreakoutCandidate] = []
        
    def add_or_update_candidate(self, date: pd.Timestamp, condition_type: BreakoutCondition, value: float = None, details: Dict = None):
        date_str = date.strftime('%Y-%m-%d')
        active_candidate = self._find_active_candidate(date)
        if active_candidate:
            self._update_candidate(active_candidate, condition_type, date_str, value, details)
        else:
            self._create_new_candidate(date, condition_type, date_str, value, details)
    
    def _find_active_candidate(self, date: pd.Timestamp) -> Optional[BreakoutCandidate]:
        for candidate in self.candidates:
            if candidate.status == "pending" and pd.Timestamp(candidate.window_start) <= date <= pd.Timestamp(candidate.window_end):
                return candidate
        return None
    
    def _create_new_candidate(self, date: pd.Timestamp, condition_type: BreakoutCondition, date_str: str, value: float, details: Dict):
        conditions = {cond.value: ConditionStatus() for cond in BreakoutCondition}
        conditions[condition_type.value] = ConditionStatus(met=True, date=date_str, value=value, details=details)
        candidate = BreakoutCandidate(
            trigger_date=date_str,
            window_start=date.strftime('%Y-%m-%d'),
            window_end=(date + pd.Timedelta(days=self.tracking_window)).strftime('%Y-%m-%d'),
            conditions=conditions)
        self.candidates.append(candidate)
    
    def _update_candidate(self, candidate: BreakoutCandidate, condition_type: BreakoutCondition, date_str: str, value: float, details: Dict):
        if not candidate.conditions[condition_type.value].met:
            candidate.conditions[condition_type.value] = ConditionStatus(met=True, date=date_str, value=value, details=details)
    
    def check_and_update_candidates(self, current_date: pd.Timestamp) -> List[BreakoutCandidate]:
        confirmed_breakouts = []
        for candidate in self.candidates:
            if candidate.status != "pending": continue
            met_count = sum(1 for cond in candidate.conditions.values() if cond.met)
            if met_count >= self.min_conditions:
                candidate.status = "confirmed"
                candidate.confirmation_date = current_date.strftime('%Y-%m-%d')
                candidate.score = self._calculate_breakout_score(candidate)
                confirmed_breakouts.append(candidate)
            elif current_date > pd.Timestamp(candidate.window_end):
                candidate.status = "expired"
        return confirmed_breakouts
    
    def _calculate_breakout_score(self, candidate: BreakoutCandidate) -> float:
        score = 0
        met_conditions = [c for c in candidate.conditions.values() if c.met]
        score += (len(met_conditions) / len(candidate.conditions)) * 40
        if len(met_conditions) >= 2:
            dates = [pd.Timestamp(c.date) for c in met_conditions if c.date]
            if dates:
                date_range = (max(dates) - min(dates)).days
                if date_range <= 3: score += 30
                elif date_range <= 7: score += 20
                else: score += 10
        volume_cond = candidate.conditions.get(BreakoutCondition.VOLUME_SURGE.value)
        if volume_cond and volume_cond.met and volume_cond.value:
            if volume_cond.value >= 3.0: score += 20
            elif volume_cond.value >= 2.0: score += 15
            else: score += 10
        high_cond = candidate.conditions.get(BreakoutCondition.HIGH_BREAKOUT.value)
        if high_cond and high_cond.met and high_cond.date == candidate.trigger_date: score += 10
        return min(100, score)

    def get_active_candidates(self) -> List[BreakoutCandidate]: return [c for c in self.candidates if c.status == "pending"]
    def get_confirmed_breakouts(self) -> List[BreakoutCandidate]: return [c for c in self.candidates if c.status == "confirmed"]


class StageHistoryManager:
    def __init__(self, ticker: str, data_dir: str = "./stage_history", breakout_window_days: int = 15):
        self.ticker = ticker
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.history_file = self.data_dir / f"{ticker}_stage_history.json"
        self.breakout_tracker = BreakoutTracker(tracking_window_days=breakout_window_days)
        self.history = self._load_history()
        
    def _load_history(self) -> Dict:
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                data = json.load(f)
                if 'breakout_candidates' in data:
                    for cand_dict in data['breakout_candidates']:
                        conditions = {key: ConditionStatus(**cond_dict) for key, cond_dict in cand_dict.get('conditions', {}).items()}
                        cand_dict['conditions'] = conditions
                        self.breakout_tracker.candidates.append(BreakoutCandidate(**cand_dict))
                return data
        return self._initialize_history()
    
    def _initialize_history(self) -> Dict:
        return {'ticker': self.ticker, 'created_at': datetime.now().isoformat(), 'current_stage': None, 'current_substage': None, 'stage_start_date': None, 'stage_transitions': [], 'breakout_candidates': [], 'confirmed_breakouts': [], 'last_updated': None}
    
    def _save_history(self):
        self.history['last_updated'] = datetime.now().isoformat()
        self.history['breakout_candidates'] = [cand.to_dict() for cand in self.breakout_tracker.candidates]
        with open(self.history_file, 'w') as f: json.dump(self.history, f, indent=2, default=str)
    
    def analyze_and_update(self, df: pd.DataFrame, benchmark_df: pd.DataFrame):
        latest_date = df.index[-1]
        if self.history['current_stage'] is None:
            stage_detector = StageDetector(df)
            stage, substage = stage_detector.determine_stage()
            self._initialize_stage(latest_date, stage, substage)
        
        # ステージ1の場合は、ブレイクアウト条件を時系列でチェック
        if self.history['current_stage'] == 1:
            self._check_individual_conditions(df, latest_date)
            confirmed_breakouts = self.breakout_tracker.check_and_update_candidates(latest_date)
            for breakout in confirmed_breakouts:
                self._process_confirmed_breakout(breakout)
        
        # すべてのステージで、後続ステージへの移行をチェック
        self._check_subsequent_transitions(df, latest_date)
        self._save_history()
    
    def _check_individual_conditions(self, df: pd.DataFrame, date: pd.Timestamp):
        if self._check_high_breakout(df, date): self.breakout_tracker.add_or_update_candidate(date, BreakoutCondition.HIGH_BREAKOUT, value=df.loc[date, 'High'])
        volume_ratio = self._check_volume_surge(df, date)
        if volume_ratio and volume_ratio >= 1.5: self.breakout_tracker.add_or_update_candidate(date, BreakoutCondition.VOLUME_SURGE, value=volume_ratio)
        if self._check_ma_alignment(df, date): self.breakout_tracker.add_or_update_candidate(date, BreakoutCondition.MA_ALIGNMENT, details=self._get_ma_details(df, date))
        if self._check_higher_highs_lows(df, date): self.breakout_tracker.add_or_update_candidate(date, BreakoutCondition.HIGHER_HIGHS)
    
    def _check_high_breakout(self, df: pd.DataFrame, date: pd.Timestamp) -> bool:
        lookback = 126
        if len(df) < lookback + 1: return False
        past_high = df['High'].iloc[-(lookback + 1):-1].max()
        return df.loc[date, 'High'] > past_high
    
    def _check_volume_surge(self, df: pd.DataFrame, date: pd.Timestamp) -> Optional[float]:
        latest = df.loc[date]
        if 'Volume_SMA_50' in df.columns:
            avg_vol = latest['Volume_SMA_50']
            return latest['Volume'] / avg_vol if avg_vol > 0 else None
        return None
    
    def _check_ma_alignment(self, df: pd.DataFrame, date: pd.Timestamp) -> bool:
        latest = df.loc[date]
        price, ma150, ma200 = latest['Close'], latest.get('SMA_150'), latest.get('SMA_200')
        if price and ma150 and ma200 and price > ma150 > ma200:
            ma150_trend = self._get_ma_trend(df['SMA_150'])
            return ma150_trend in ['rising', 'flat']
        return False
    
    def _get_ma_trend(self, ma_series: pd.Series, lookback: int = 20) -> str:
        if len(ma_series) < lookback: return 'unknown'
        recent = ma_series.dropna().tail(lookback)
        if len(recent) < 2: return 'unknown'
        slope = np.polyfit(np.arange(len(recent)), recent.values, 1)[0]
        if abs(slope) < 0.005 * np.mean(recent.values): return 'flat'
        return 'rising' if slope > 0 else 'declining'

    def _check_higher_highs_lows(self, df: pd.DataFrame, date: pd.Timestamp) -> bool:
        lookback = 40
        if len(df) < lookback: return False
        recent = df.tail(lookback//2)
        prior = df.iloc[-lookback:-(lookback//2)]
        return recent['High'].max() > prior['High'].max() and recent['Low'].min() > prior['Low'].min()
    
    def _get_ma_details(self, df: pd.DataFrame, date: pd.Timestamp) -> Dict:
        latest = df.loc[date]
        return {k: float(latest.get(k, 0)) for k in ['Close', 'SMA_50', 'SMA_150', 'SMA_200']}
    
    def _process_confirmed_breakout(self, breakout: BreakoutCandidate):
        if self.history['current_stage'] != 2:
            reason = f"Time-series breakout confirmed | Score: {breakout.score:.1f}"
            self._execute_stage_transition(pd.Timestamp(breakout.confirmation_date), 2, '2A', reason)
            self.history['confirmed_breakouts'].append(breakout.to_dict())
    
    def _check_subsequent_transitions(self, df: pd.DataFrame, date: pd.Timestamp):
        """現在のステージに応じて、後続ステージへの移行を網羅的にチェック"""
        current_stage = self.history['current_stage']
        result = None
        if current_stage == 2: result = self._check_stage2_to_3(df, date)
        elif current_stage == 3: result = self._check_stage3_to_4(df, date)
        elif current_stage == 4: result = self._check_stage4_to_1(df, date)
        
        if result and result.get('should_transition'):
            self._execute_stage_transition(date, result['new_stage'], result['new_substage'], result['reason'])

    def _check_stage2_to_3(self, df: pd.DataFrame, date: pd.Timestamp) -> Dict:
        latest = df.loc[date]
        if latest['Close'] < latest['SMA_50'] and self._check_volume_surge(df, date) >= 1.5:
            return {'should_transition': True, 'new_stage': 3, 'new_substage': '3A', 'reason': 'Broke 50-day MA on high volume'}
        ma150_trend = self._get_ma_trend(df['SMA_150'])
        if ma150_trend == 'flat' and df['Close'].rolling(20).std().iloc[-1] > df['Close'].rolling(100).std().iloc[-1]:
            return {'should_transition': True, 'new_stage': 3, 'new_substage': '3', 'reason': 'MA flattening with increased volatility'}
        return {'should_transition': False}

    def _check_stage3_to_4(self, df: pd.DataFrame, date: pd.Timestamp) -> Dict:
        lookback = 50
        if len(df) < lookback: return {'should_transition': False}
        support_level = df['Low'].tail(lookback).min()
        if df.loc[date, 'Close'] < support_level:
            return {'should_transition': True, 'new_stage': 4, 'new_substage': '4A', 'reason': f'Support broken at {support_level:.2f}'}
        return {'should_transition': False}

    def _check_stage4_to_1(self, df: pd.DataFrame, date: pd.Timestamp) -> Dict:
        latest = df.loc[date]
        if latest['Close'] > latest['SMA_150'] and self._get_ma_trend(df['SMA_150']) == 'flat' and latest['Low'] >= df['Low'].tail(50).min():
            return {'should_transition': True, 'new_stage': 1, 'new_substage': '1A', 'reason': 'Price crossed 30W MA, basing begins'}
        return {'should_transition': False}

    def _initialize_stage(self, date: pd.Timestamp, stage: int, substage: str):
        self.history.update({'current_stage': stage, 'current_substage': substage, 'stage_start_date': date.isoformat()})
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
        print(f"Tracking Status: {len(self.breakout_tracker.get_active_candidates())} active candidates, {len(self.history.get('confirmed_breakouts', []))} confirmed breakouts.")
        
        if self.history.get('stage_transitions'):
            print("\n【Stage Transition History】")
            for t in self.history['stage_transitions'][-5:]:
                print(f"  - {t['date']}: Stage {t['from']} -> {t['to']} ({t['reason']})")