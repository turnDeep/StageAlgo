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


class BreakoutCondition(Enum):
    """Stage 1 → Stage 2の条件"""
    HIGH_BREAKOUT = "high_breakout"
    VOLUME_SURGE = "volume_surge"
    MA_ALIGNMENT = "ma_alignment"


class BreakdownCondition(Enum):
    """Stage 3 → Stage 4の条件"""
    LOW_BREAKDOWN = "low_breakdown"
    VOLUME_SURGE = "volume_surge"
    MA_BREAKDOWN = "ma_breakdown"


class ToppingCondition(Enum):
    """Stage 2 → Stage 3の条件"""
    MA_FLATTENING = "ma_flattening"
    PRICE_SIDEWAYS = "price_sideways"
    VOLUME_CHURNING = "volume_churning"


class BasingCondition(Enum):
    """Stage 4 → Stage 1の条件"""
    MA_FLATTENING = "ma_flattening"
    PRICE_STABILIZING = "price_stabilizing"
    VOLUME_DRYING = "volume_drying"


@dataclass
class ConditionStatus:
    """条件の状態"""
    met: bool = False
    date: Optional[str] = None
    value: Optional[float] = None
    details: Optional[Dict] = field(default_factory=dict)


@dataclass
class TransitionCandidate:
    """ステージ移行候補"""
    trigger_date: str
    window_start: str
    window_end: str
    from_stage: int
    to_stage: int
    conditions: Dict[str, ConditionStatus] = field(default_factory=dict)
    status: str = "pending"  # pending, pending_confirmation, confirmed, expired, failed
    confirmation_date: Optional[str] = None
    score: float = 0.0
    breakout_level: Optional[float] = None  # ブレイクアウト価格を記録

    def to_dict(self):
        return asdict(self)


class StageHistoryManager:
    """
    ステージ履歴管理システム（完全改訂版）

    【役割】
    - 時系列でのステージ移行を追跡
    - すべての可能な移行パターンを監視
    - 履歴の保存と統計分析

    【Stage Detectorとの分業】
    - Stage Detector: 現在の状態判定（スナップショット）
    - History Manager: 移行の追跡と履歴管理（時系列）
    """

    def __init__(self, ticker: str, data_dir: str = "./stage_history", **kwargs):
        """
        Args:
            ticker: ティッカーシンボル
            data_dir: 履歴保存ディレクトリ
            **kwargs: 各移行パターンのパラメータ
        """
        self.ticker = ticker
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.history_file = self.data_dir / f"{ticker}_stage_history.json"

        # パラメータ設定
        self.params = {
            'breakout': {  # Stage 1 → 2
                'window_days': 15,
                'min_high_period': 126,
                'volume_multiplier': 1.5,
                **kwargs.get('breakout', {})
            },
            'breakdown': {  # Stage 3 → 4
                'window_days': 15,
                'min_low_period': 126,
                'volume_multiplier': 1.5,
                **kwargs.get('breakdown', {})
            },
            'topping': {  # Stage 2 → 3
                'window_days': 20,
                'ma_type': 'SMA_150',
                **kwargs.get('topping', {})
            },
            'basing': {  # Stage 4 → 1
                'window_days': 20,
                'ma_type': 'SMA_150',
                **kwargs.get('basing', {})
            },
            'continuation_breakout': {  # Stage 3 → 2
                'window_days': 15,
                'min_high_period': 63,
                'volume_multiplier': 1.5,
                **kwargs.get('continuation_breakout', {})
            }
        }
        
        # 移行候補の追跡
        self.transition_candidates: List[TransitionCandidate] = []

        # 履歴の読み込み
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
            'current_substage': None,
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
        データを分析してステージを更新

        【重要な改善】
        - Stage Detectorで正確な初期ステージを判定
        - すべての移行パターンを監視

        Args:
            df: 指標計算済みのDataFrame
            benchmark_df: ベンチマークデータ
        """
        latest_date = df.index[-1]

        # 初回実行時: Stage Detectorで正確に初期化
        if self.history['current_stage'] is None:
            from stage_detector import StageDetector
            detector = StageDetector(df, benchmark_df)
            initial_stage, initial_substage = detector.determine_stage()
            self._initialize_stage(latest_date, initial_stage, initial_substage)
            self._save_history()
            return
        
        # 現在のステージに応じて、可能な移行を監視
        current_stage = self.history['current_stage']

        if current_stage == 1:
            # Stage 1からの可能な移行
            self._check_stage1_to_2(df, latest_date)     # → Stage 2
            self._check_stage1_to_4(df, latest_date)     # → Stage 4 (失敗)

        elif current_stage == 2:
            # Stage 2からの可能な移行
            self._check_stage2_to_3(df, latest_date)     # → Stage 3
            self._check_stage2_continuation(df, latest_date)  # → Stage 2 (再蓄積)

        elif current_stage == 3:
            # Stage 3からの可能な移行
            self._check_stage3_to_4(df, latest_date)     # → Stage 4
            self._check_stage3_to_2(df, latest_date)     # → Stage 2 (継続ブレイクアウト)

        elif current_stage == 4:
            # Stage 4からの可能な移行
            self._check_stage4_to_1(df, latest_date)     # → Stage 1
            self._check_stage4_continuation(df, latest_date)  # → Stage 4 (再分配)

        # 候補の確認と更新
        self._update_candidates(df, latest_date)

        self._save_history()

    # ===== Stage 1からの移行 =====

    def _check_stage1_to_2(self, df: pd.DataFrame, date: pd.Timestamp):
        """
        Stage 1 → Stage 2（最も重要な移行）

        条件:
        1. 高値更新（126日以上）
        2. 出来高急増（1.5倍以上）
        3. MA配列の改善
        """
        params = self.params['breakout']

        # 条件1: 高値更新
        if self._check_high_breakout(df, date, params['min_high_period']):
            self._add_transition_candidate(
                date, 1, 2,
                BreakoutCondition.HIGH_BREAKOUT.value,
                'breakout'
            )

        # 条件2: 出来高急増
        if self._check_volume_surge(df, date, params['volume_multiplier']):
            self._add_transition_candidate(
                date, 1, 2,
                BreakoutCondition.VOLUME_SURGE.value,
                'breakout'
            )

        # 条件3: MA配列
        if self._check_ma_alignment_bullish(df, date):
            self._add_transition_candidate(
                date, 1, 2,
                BreakoutCondition.MA_ALIGNMENT.value,
                'breakout'
            )

    def _check_stage1_to_4(self, df: pd.DataFrame, date: pd.Timestamp):
        """
        Stage 1 → Stage 4（ベース失敗）

        条件:
        1. サポート下抜け
        2. MA配列の悪化
        """
        # 簡易的な下抜け検出
        if self._check_low_breakdown(df, date, 126):
            if self._check_ma_alignment_bearish(df, date):
                self._execute_stage_transition(
                    date, 4, '4A',
                    'Base failure - breakdown from Stage 1'
                )

    # ===== Stage 2からの移行 =====

    def _check_stage2_to_3(self, df: pd.DataFrame, date: pd.Timestamp):
        """
        Stage 2 → Stage 3（天井形成開始）

        条件（Stage 1の逆ロジック）:
        1. MA平坦化
        2. 価格の横ばい
        3. 出来高のChurning
        """
        params = self.params['topping']

        # 条件1: MA平坦化
        if self._check_ma_flattening(df, date, params['ma_type']):
            self._add_transition_candidate(
                date, 2, 3,
                ToppingCondition.MA_FLATTENING.value,
                'topping'
            )

        # 条件2: 価格の横ばい
        if self._check_price_sideways(df, date):
            self._add_transition_candidate(
                date, 2, 3,
                ToppingCondition.PRICE_SIDEWAYS.value,
                'topping'
            )

        # 条件3: 出来高Churning
        if self._check_volume_churning(df, date):
            self._add_transition_candidate(
                date, 2, 3,
                ToppingCondition.VOLUME_CHURNING.value,
                'topping'
            )

    def _check_stage2_continuation(self, df: pd.DataFrame, date: pd.Timestamp):
        """
        Stage 2内の再蓄積（Stage 2 → Stage 2A）

        押し目からの再上昇
        """
        # 10週MA付近での押し目
        latest = df.loc[date]
        ma_50 = latest.get('SMA_50')

        if ma_50 and latest['Close'] > ma_50 * 0.98:
            # MA付近で反発している可能性
            if self._check_volume_surge(df, date, 1.3):
                # 軽度の出来高増加で再上昇
                # これは新しいサブステージへの移行として記録
                pass  # 実装は簡略化

    # ===== Stage 3からの移行 =====

    def _check_stage3_to_4(self, df: pd.DataFrame, date: pd.Timestamp):
        """
        Stage 3 → Stage 4（ブレイクダウン）

        条件:
        1. サポート下抜け
        2. 出来高急増
        3. MA下抜け
        """
        params = self.params['breakdown']

        # 条件1: 安値更新
        if self._check_low_breakdown(df, date, params['min_low_period']):
            self._add_transition_candidate(
                date, 3, 4,
                BreakdownCondition.LOW_BREAKDOWN.value,
                'breakdown'
            )

        # 条件2: 出来高急増
        if self._check_volume_surge(df, date, params['volume_multiplier']):
            self._add_transition_candidate(
                date, 3, 4,
                BreakdownCondition.VOLUME_SURGE.value,
                'breakdown'
            )

        # 条件3: MA下抜け
        if self._check_ma_alignment_bearish(df, date):
            self._add_transition_candidate(
                date, 3, 4,
                BreakdownCondition.MA_BREAKDOWN.value,
                'breakdown'
            )

    def _check_stage3_to_2(self, df: pd.DataFrame, date: pd.Timestamp):
        """
        Stage 3 → Stage 2（継続ブレイクアウト）

        天井形成と思われたが、再び上昇
        """
        params = self.params['continuation_breakout']

        # Stage 3の高値を上抜ける
        if self._check_high_breakout(df, date, params['min_high_period']):
            if self._check_volume_surge(df, date, params['volume_multiplier']):
                self._execute_stage_transition(
                    date, 2, '2A',
                    'Continuation breakout from Stage 3'
                )

    # ===== Stage 4からの移行 =====

    def _check_stage4_to_1(self, df: pd.DataFrame, date: pd.Timestamp):
        """
        Stage 4 → Stage 1（底打ち）

        条件（Stage 2の逆ロジック）:
        1. MA平坦化
        2. 価格の安定化
        3. 出来高の減少
        """
        params = self.params['basing']

        # 条件1: MA平坦化
        if self._check_ma_flattening(df, date, params['ma_type']):
            self._add_transition_candidate(
                date, 4, 1,
                BasingCondition.MA_FLATTENING.value,
                'basing'
            )

        # 条件2: 価格の安定化
        if self._check_price_stabilizing(df, date):
            self._add_transition_candidate(
                date, 4, 1,
                BasingCondition.PRICE_STABILIZING.value,
                'basing'
            )

        # 条件3: 出来高の減少
        if self._check_volume_drying(df, date):
            self._add_transition_candidate(
                date, 4, 1,
                BasingCondition.VOLUME_DRYING.value,
                'basing'
            )

    def _check_stage4_continuation(self, df: pd.DataFrame, date: pd.Timestamp):
        """
        Stage 4内の再分配（Stage 4 → Stage 4A）

        一時的な反発からの再下落
        """
        # 簡略化: 実装省略
        pass

    # ===== 条件判定のヘルパーメソッド =====

    def _check_high_breakout(self, df: pd.DataFrame, date: pd.Timestamp,
                            lookback: int) -> bool:
        """高値更新を確認"""
        if len(df) < lookback + 1:
            return False

        try:
            idx = df.index.get_loc(date)
            if idx < lookback:
                return False

            current_high = df.loc[date, 'High']
            past_high = df['High'].iloc[idx-lookback:idx].max()

            return current_high > past_high
        except:
            return False

    def _check_low_breakdown(self, df: pd.DataFrame, date: pd.Timestamp,
                            lookback: int) -> bool:
        """安値更新を確認"""
        if len(df) < lookback + 1:
            return False

        try:
            idx = df.index.get_loc(date)
            if idx < lookback:
                return False

            current_low = df.loc[date, 'Low']
            past_low = df['Low'].iloc[idx-lookback:idx].min()

            return current_low < past_low
        except:
            return False

    def _check_volume_surge(self, df: pd.DataFrame, date: pd.Timestamp,
                           multiplier: float) -> bool:
        """出来高急増を確認"""
        try:
            latest = df.loc[date]
            if 'Volume_SMA_50' in df.columns and latest['Volume_SMA_50'] > 0:
                return latest['Volume'] > latest['Volume_SMA_50'] * multiplier
        except:
            pass
        return False

    def _check_ma_alignment_bullish(self, df: pd.DataFrame, date: pd.Timestamp) -> bool:
        """強気のMA配列を確認"""
        try:
            latest = df.loc[date]
            price = latest['Close']
            ma_150 = latest.get('SMA_150')
            ma_200 = latest.get('SMA_200')

            if ma_150 and ma_200:
                return price > ma_150 and self._get_ma_trend(df['SMA_150']) in ['rising', 'flat']
        except:
            pass
        return False

    def _check_ma_alignment_bearish(self, df: pd.DataFrame, date: pd.Timestamp) -> bool:
        """弱気のMA配列を確認"""
        try:
            latest = df.loc[date]
            price = latest['Close']
            ma_150 = latest.get('SMA_150')
            ma_200 = latest.get('SMA_200')

            if ma_150 and ma_200:
                return price < ma_150 and self._get_ma_trend(df['SMA_150']) in ['declining', 'flat']
        except:
            pass
        return False

    def _check_ma_flattening(self, df: pd.DataFrame, date: pd.Timestamp,
                            ma_type: str) -> bool:
        """MA平坦化を確認"""
        try:
            trend = self._get_ma_trend(df[ma_type])
            return trend == 'flat'
        except:
            return False

    def _check_price_sideways(self, df: pd.DataFrame, date: pd.Timestamp) -> bool:
        """価格の横ばいを確認"""
        try:
            idx = df.index.get_loc(date)
            if idx < 20:
                return False

            recent = df.iloc[idx-20:idx+1]
            high = recent['High'].max()
            low = recent['Low'].min()
            avg = recent['Close'].mean()

            if avg > 0:
                price_range = (high - low) / avg
                return price_range < 0.10  # Stricter condition
        except:
            pass
        return False

    def _check_volume_churning(self, df: pd.DataFrame, date: pd.Timestamp) -> bool:
        """出来高Churningを確認"""
        try:
            idx = df.index.get_loc(date)
            if idx < 20:
                return False

            recent_volume = df['Volume'].iloc[idx-20:idx+1]
            vol_cv = recent_volume.std() / recent_volume.mean()
            return vol_cv > 0.5
        except:
            pass
        return False

    def _check_price_stabilizing(self, df: pd.DataFrame, date: pd.Timestamp) -> bool:
        """価格の安定化を確認"""
        return self._check_price_sideways(df, date)

    def _check_volume_drying(self, df: pd.DataFrame, date: pd.Timestamp) -> bool:
        """出来高減少を確認"""
        try:
            idx = df.index.get_loc(date)
            if idx < 20:
                return False

            recent_rvol = df['Relative_Volume'].iloc[idx-20:idx+1] if 'Relative_Volume' in df.columns else None
            if recent_rvol is not None:
                return recent_rvol.iloc[-1] < recent_rvol.mean() * 0.7
        except:
            pass
        return False

    def _get_ma_trend(self, ma_series: pd.Series, lookback: int = 20) -> str:
        """MAトレンドを取得"""
        if len(ma_series.dropna()) < lookback:
            return 'unknown'

        recent = ma_series.dropna().tail(lookback)
        slope = np.polyfit(np.arange(len(recent)), recent.values, 1)[0]

        threshold = 0.008 * np.mean(recent.values)
        if abs(slope) < threshold:
            return 'flat'
        return 'rising' if slope > 0 else 'declining'

    # ===== 候補管理 =====

    def _add_transition_candidate(self, date: pd.Timestamp, from_stage: int,
                                  to_stage: int, condition_type: str,
                                  pattern_type: str):
        """移行候補を追加または更新"""
        date_str = date.strftime('%Y-%m-%d')

        # 既存の候補を探す
        active_candidate = self._find_active_candidate(date, from_stage, to_stage)

        if active_candidate:
            # 既存候補を更新
            if not active_candidate.conditions[condition_type].met:
                active_candidate.conditions[condition_type] = ConditionStatus(
                    met=True,
                    date=date_str
                )
        else:
            # 新しい候補を作成
            params = self.params[pattern_type]
            window_days = params.get('window_days', 15)

            # 条件の初期化
            if pattern_type == 'breakout':
                condition_types = [c.value for c in BreakoutCondition]
            elif pattern_type == 'breakdown':
                condition_types = [c.value for c in BreakdownCondition]
            elif pattern_type == 'topping':
                condition_types = [c.value for c in ToppingCondition]
            elif pattern_type == 'basing':
                condition_types = [c.value for c in BasingCondition]
            else:
                condition_types = []

            conditions = {ct: ConditionStatus() for ct in condition_types}
            conditions[condition_type] = ConditionStatus(met=True, date=date_str)

            candidate = TransitionCandidate(
                trigger_date=date_str,
                window_start=date_str,
                window_end=(date + pd.Timedelta(days=window_days)).strftime('%Y-%m-%d'),
                from_stage=from_stage,
                to_stage=to_stage,
                conditions=conditions
            )

            self.transition_candidates.append(candidate)

    def _find_active_candidate(self, date: pd.Timestamp, from_stage: int,
                              to_stage: int) -> Optional[TransitionCandidate]:
        """アクティブな候補を探す"""
        for candidate in self.transition_candidates:
            if (candidate.status == "pending" and
                candidate.from_stage == from_stage and
                candidate.to_stage == to_stage and
                pd.Timestamp(candidate.window_start) <= date <= pd.Timestamp(candidate.window_end)):
                return candidate
        return None

    def _verify_breakout_hold(self, df: pd.DataFrame, candidate: TransitionCandidate,
                              current_date: pd.Timestamp,
                              hold_window: int = 15,
                              hold_tolerance: float = 0.98,
                              hold_days_pct: float = 0.8) -> Tuple[str, str]:
        """ブレイクアウトが一定期間維持されたかを確認。 'confirmed', 'failed', 'pending' のいずれかを返す"""
        if not candidate.breakout_level:
            return 'failed', "No breakout level recorded"

        breakout_date = pd.Timestamp(candidate.trigger_date)
        end_of_window = breakout_date + pd.Timedelta(days=hold_window)

        # 現在日が確認期間を過ぎていたら失敗
        if current_date > end_of_window:
            return 'failed', "Confirmation window expired"

        # 確認期間のデータを取得
        confirmation_period_df = df.loc[breakout_date:current_date]

        # ブレイクアウトレベルを維持した日数をカウント
        hold_level = candidate.breakout_level * hold_tolerance
        days_held = (confirmation_period_df['Close'] >= hold_level).sum()

        # 必要な維持日数を満たしているか
        required_days = len(confirmation_period_df) * hold_days_pct

        # 確認期間中に価格が大幅に下落したら即失敗
        if df.loc[current_date, 'Close'] < candidate.breakout_level * (hold_tolerance - 0.07): # 7%下落
            return 'failed', f"Price dropped significantly below breakout level of {candidate.breakout_level:.2f}"

        if days_held >= required_days:
            # 80%以上の期間で条件を満たしていれば確定
            return 'confirmed', f"Hold confirmed for {days_held}/{len(confirmation_period_df)} days"

        return 'pending', "Confirmation pending"

    def _update_candidates(self, df: pd.DataFrame, current_date: pd.Timestamp):
        """候補を更新して、確定または期限切れに"""
        for candidate in self.transition_candidates:
            # 1. 確認待ちの候補を処理
            if candidate.status == "pending_confirmation":
                if candidate.from_stage == 1 and candidate.to_stage == 2:
                    status, reason = self._verify_breakout_hold(df, candidate, current_date)
                    if status == 'confirmed':
                        candidate.status = "confirmed"
                        candidate.confirmation_date = current_date.strftime('%Y-%m-%d')
                        self._execute_stage_transition(
                            pd.Timestamp(candidate.trigger_date), # 移行日はトリガー日
                            candidate.to_stage, '2A', f"Breakout confirmed: {reason}"
                        )
                    elif status == 'failed':
                        candidate.status = "failed"
                continue

            # 2. ペンディング中の候補を処理
            if candidate.status == "pending":
                met_count = sum(1 for cond in candidate.conditions.values() if cond.met)
                min_conditions = 2  # 最低2条件

                if met_count >= min_conditions:
                    # 特別ケース: Stage 1 -> 2 ブレイクアウトは確認待ちへ
                    if candidate.from_stage == 1 and candidate.to_stage == 2:
                        candidate.status = "pending_confirmation"

                        # ブレイクアウトレベルとトリガー日を記録
                        high_breakout_date_str = None
                        if candidate.conditions[BreakoutCondition.HIGH_BREAKOUT.value].met:
                            high_breakout_date_str = candidate.conditions[BreakoutCondition.HIGH_BREAKOUT.value].date

                        if high_breakout_date_str:
                            trigger_date = pd.Timestamp(high_breakout_date_str)
                            candidate.trigger_date = trigger_date.strftime('%Y-%m-%d')
                            candidate.breakout_level = df.loc[trigger_date, 'High']
                        else: # フォールバック
                            candidate.status = "failed" # 高値更新がない場合は失敗
                        continue

                    # その他の移行は即時確定
                    candidate.status = "confirmed"
                    candidate.confirmation_date = current_date.strftime('%Y-%m-%d')
                    candidate.score = (met_count / len(candidate.conditions)) * 100

                    self._execute_stage_transition(
                        current_date,
                        candidate.to_stage,
                        self._infer_substage(candidate.to_stage),
                        f"Transition confirmed: {met_count}/{len(candidate.conditions)} conditions met"
                    )

                elif current_date > pd.Timestamp(candidate.window_end):
                    candidate.status = "expired"

    def _infer_substage(self, stage: int) -> str:
        """ステージから適切なサブステージを推論"""
        if stage == 1:
            return "1A"
        elif stage == 2:
            return "2A"
        elif stage == 3:
            return "3A"
        elif stage == 4:
            return "4A"
        return str(stage)

    # ===== ステージ管理 =====

    def _initialize_stage(self, date: pd.Timestamp, stage: int, substage: str):
        """初期ステージを設定"""
        self.history.update({
            'current_stage': stage,
            'current_substage': substage,
            'stage_start_date': date.isoformat()
        })

        self.history['stage_transitions'].append({
            'date': date.isoformat(),
            'from': None,
            'from_substage': None,
            'to': stage,
            'to_substage': substage,
            'reason': 'Initial detection'
        })

    def _execute_stage_transition(self, date: pd.Timestamp, new_stage: int,
                                  new_substage: str, reason: str):
        """ステージ移行を実行"""
        old_stage = self.history['current_stage']
        old_substage = self.history['current_substage']

        if old_stage == new_stage and old_substage == new_substage:
            return  # 変化なし

        # 履歴を更新
        self.history.update({
            'current_stage': new_stage,
            'current_substage': new_substage,
            'stage_start_date': date.isoformat()
        })

        # 移行を記録
        transition = {
            'date': date.isoformat(),
            'from': old_stage,
            'from_substage': old_substage,
            'to': new_stage,
            'to_substage': new_substage,
            'reason': reason
        }

        self.history['stage_transitions'].append(transition)

        # 統計を更新
        self._update_statistics(old_stage, new_stage)

    def _update_statistics(self, from_stage: int, to_stage: int):
        """統計情報を更新"""
        stats = self.history['statistics']
        stats['total_transitions'] += 1

        # 主要な移行パターンをカウント
        if from_stage == 1 and to_stage == 2:
            stats['stage1_to_stage2'] += 1
        elif from_stage == 2 and to_stage == 3:
            stats['stage2_to_stage3'] += 1
        elif from_stage == 3 and to_stage == 2:
            stats['stage3_to_stage2'] += 1
        elif from_stage == 3 and to_stage == 4:
            stats['stage3_to_stage4'] += 1
        elif from_stage == 4 and to_stage == 1:
            stats['stage4_to_stage1'] += 1
        else:
            stats['other_transitions'] += 1

    # ===== レポート =====

    def print_summary(self):
        """サマリーを表示"""
        status = self.history
        print("=" * 70)
        print(f"Stage History Summary: {status['ticker']}")
        print("=" * 70)
        print(f"Current Stage: {status.get('current_stage')} ({status.get('current_substage')})")

        if status.get('stage_transitions'):
            print(f"\n【Stage Transition History】")
            for t in status['stage_transitions'][-5:]:
                print(f"  - {t['date']}: Stage {t.get('from')} → {t['to']} ({t['reason']})")

        print(f"\n【Statistics】")
        stats = status['statistics']
        print(f"  Total Transitions: {stats['total_transitions']}")
        print(f"  Stage 1→2: {stats['stage1_to_stage2']}")
        print(f"  Stage 2→3: {stats['stage2_to_stage3']}")
        print(f"  Stage 3→2: {stats['stage3_to_stage2']}")
        print(f"  Stage 3→4: {stats['stage3_to_stage4']}")
        print(f"  Stage 4→1: {stats['stage4_to_stage1']}")
        print(f"  Other: {stats['other_transitions']}")


if __name__ == '__main__':
    # テスト用
    from data_fetcher import fetch_stock_data
    from indicators import calculate_all_basic_indicators

    print("Stage History Manager（完全改訂版）のテストを開始...")

    ticker = 'AAPL'
    print(f"\n{ticker} の履歴分析:")

    stock_df, benchmark_df = fetch_stock_data(ticker, period='2y')

    if stock_df is not None and benchmark_df is not None:
        indicators_df = calculate_all_basic_indicators(stock_df)
        indicators_df = indicators_df.dropna()

        if len(indicators_df) >= 252:
            manager = StageHistoryManager(ticker)

            # 最新データで分析
            manager.analyze_and_update(indicators_df, benchmark_df)
            manager.print_summary()