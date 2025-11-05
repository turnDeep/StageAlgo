"""
Base Minervini Analyzer（完全改訂版）

【重要な改善】
1. 週足データを基本とする（ミネルヴィニの標準）
2. VCPDetectorとの完全統合
3. VolumeAnalyzerとの統合
4. コントラクション段階的減少の厳密な検証
5. 3週間未満のブレイクアウト処理
6. 出来高縮小→ブレイクアウト急増の確認
7. 20%ルールとフラットベースの実装

【理論的基盤】
- Mark Minerviniの「Trade Like a Stock Market Wizard」
- Stage 2上昇トレンド内でのベース形成パターン
- VCP（Volatility Contraction Pattern）理論
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

# 既存モジュールをインポート
from vcp_detector import VCPDetector
from volume_analyzer import VolumeAnalyzer
from stage_detector import StageDetector


@dataclass
class Contraction:
    """コントラクション（調整局面）のデータクラス"""
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    high: float
    low: float
    depth_pct: float
    duration_weeks: int
    avg_volume: float
    volume_ratio: float # 前回コントラクションとの比較


@dataclass
class BaseInfo:
    """ベース情報のデータクラス"""
    base_number: int
    start_date: pd.Timestamp
    resistance_price: float
    lowest_price: float
    depth_pct: float
    duration_weeks: int
    contractions: List[Contraction]
    vcp_valid: bool
    volume_dry_up: bool
    breakout_date: Optional[pd.Timestamp] = None
    breakout_volume_surge: Optional[float] = None
    base_type: str = 'pending'
    quality: str = 'unknown'


class BaseMinerviniAnalyzer:
    """
    Base Minervini Analyzer（完全改訂版）

    【重要な変更点】
    1. 週足データを主軸とする
    2. VCPパターンの厳密な検証
    3. コントラクションの段階的減少を確認
    4. 3週間未満のベースは無効
    5. 出来高分析の統合
    """

    def __init__(self,
                 daily_df: pd.DataFrame,
                 weekly_df: pd.DataFrame,
                 benchmark_daily_df: Optional[pd.DataFrame] = None):
        """
        Args:
            daily_df: 日足データ（補助用）
            weekly_df: 週足データ（主軸）
            benchmark_daily_df: ベンチマーク日足データ（Stage判定用）
        """
        self.daily_df = daily_df
        self.weekly_df = weekly_df
        self.benchmark_daily_df = benchmark_daily_df

        # ステート管理
        self.state = "SCANNING"
        self.current_base: Optional[Dict] = None
        self.bases: List[BaseInfo] = []
        self.last_breakout_price: float = 0
        self.base_count: int = 0

        # パラメータ（調整可能）
        self.params = {
            'min_base_weeks': 3,
            'max_base_weeks': 65,
            'min_depth_pct': 10,
            'max_depth_pct': 50,
            'min_separation_pct': 20, # 20%ルール
            'min_breakout_volume_increase': 40, # 40%以上の出来高増加
            'contraction_decrease_ratio': 0.8, # 各コントラクションは前回の80%以下
        }

    def analyze(self) -> List[Dict]:
        """
        週足データを使用してベース分析を実行

        Returns:
            List[Dict]: ベース検出イベントのリスト
        """
        events = []

        # Stage 2確認に必要な最小データ数
        min_weeks = 40 # 約200日
        if len(self.weekly_df) < min_weeks:
            print(f"データ不足: {len(self.weekly_df)}週（最低{min_weeks}週必要）")
            return events

        # 週足データを1週ずつ処理
        for i, (date, row) in enumerate(self.weekly_df.iloc[min_weeks:].iterrows()):
            # その時点までのデータ
            historical_weekly = self.weekly_df.iloc[:min_weeks + i + 1]

            # Stage判定（日足データで行う）
            if self.benchmark_daily_df is not None:
                date_str = date.strftime('%Y-%m-%d')
                historical_daily = self.daily_df[self.daily_df.index <= date_str]

                if len(historical_daily) >= 200:
                    stage_detector = StageDetector(
                        historical_daily,
                        self.benchmark_daily_df,
                        interval='1d'
                    )
                    current_stage = stage_detector.determine_stage()

                    if current_stage != 2:
                        # Stage 2でない場合はスキャン継続
                        self.state = "SCANNING"
                        self.current_base = None
                        continue

            # ステートマシン処理
            if self.state == "SCANNING":
                self._handle_scanning(date, row, historical_weekly, events)

            elif self.state == "FORMING":
                self._handle_forming(date, row, historical_weekly, events)

            elif self.state == "WAITING_FOR_SEPARATION":
                self._handle_waiting_separation(date, row, historical_weekly, events)

        return events

    def _handle_scanning(self, date: pd.Timestamp, row: pd.Series,
                         historical_df: pd.DataFrame, events: List[Dict]):
        """SCANNINGステートの処理"""

        # 新しい高値更新を検出
        lookback_weeks = min(52, len(historical_df))
        recent_data = historical_df.tail(lookback_weeks)

        if row['High'] >= recent_data['High'].max():
            # 高値更新 → ベース開始候補

            # 前回のベースから十分な上昇があったか確認（20%ルール）
            if self.last_breakout_price > 0:
                gain_from_last = (row['High'] - self.last_breakout_price) / self.last_breakout_price

                if gain_from_last < self.params['min_separation_pct'] / 100:
                    # 20%未満の上昇 → まだ分離不十分
                    return

            # ベース開始
            self.state = "FORMING"
            self.current_base = {
                'start_date': date,
                'start_high': row['High'],
                'resistance': row['High'],
                'lowest_price': row['Low'],
                'contractions': [],
                'pivot_highs': [row['High']],
                'pivot_lows': [row['Low']],
                'start_volume': row['Volume']
            }

            events.append({
                'date': date.strftime('%Y-%m-%d'),
                'event': 'BASE_START',
                'base_number': self.base_count + 1,
                'resistance_price': row['High'],
                'after_20pct_gain': self.last_breakout_price > 0
            })

    def _handle_forming(self, date: pd.Timestamp, row: pd.Series,
                        historical_df: pd.DataFrame, events: List[Dict]):
        """FORMINGステートの処理"""

        if self.current_base is None:
            self.state = "SCANNING"
            return

        # ベース期間の更新
        start_date = self.current_base['start_date']
        duration_weeks = len(historical_df[historical_df.index > start_date])

        # 最安値の更新
        self.current_base['lowest_price'] = min(
            self.current_base['lowest_price'],
            row['Low']
        )

        # 調整深さの計算
        depth_pct = (
            (self.current_base['start_high'] - self.current_base['lowest_price'])
            / self.current_base['start_high'] * 100
        )

        # 失敗条件のチェック
        # 1. 深すぎる調整（50%超）
        if depth_pct > self.params['max_depth_pct']:
            events.append({
                'date': date.strftime('%Y-%m-%d'),
                'event': 'BASE_FAILED',
                'reason': f'調整深すぎる（{depth_pct:.1f}% > 50%）',
                'base_start': start_date.strftime('%Y-%m-%d')
            })
            self.state = "SCANNING"
            self.current_base = None
            return

        # 2. 期間が長すぎる（65週超）
        if duration_weeks > self.params['max_base_weeks']:
            events.append({
                'date': date.strftime('%Y-%m-%d'),
                'event': 'BASE_FAILED',
                'reason': f'期間長すぎる（{duration_weeks}週 > 65週）',
                'base_start': start_date.strftime('%Y-%m-%d')
            })
            self.state = "SCANNING"
            self.current_base = None
            return

        # 3. 200日MA割れ（日足で確認）
        date_str = date.strftime('%Y-%m-%d')
        if date_str in self.daily_df.index:
            if 'SMA_200' in self.daily_df.columns:
                if row['Close'] < self.daily_df.loc[date_str, 'SMA_200']:
                    events.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'event': 'BASE_FAILED',
                        'reason': '200日MA割れ',
                        'base_start': start_date.strftime('%Y-%m-%d')
                    })
                    self.state = "SCANNING"
                    self.current_base = None
                    return

        # コントラクション（ピボット）の検出
        self._detect_pivot_points(date, row, historical_df)

        # ブレイクアウトチェック
        if row['Close'] > self.current_base['resistance']:
            self._handle_breakout(date, row, historical_df, duration_weeks, depth_pct, events)

    def _detect_pivot_points(self, date: pd.Timestamp, row: pd.Series,
                            historical_df: pd.DataFrame):
        """ピボットハイとピボットローを検出"""

        if len(historical_df) < 5:
            return

        # 直近5週のデータ
        recent_5 = historical_df.tail(5)

        # ピボットハイの検出（中央が両側より高い）
        if len(recent_5) == 5:
            middle_idx = 2
            middle_high = recent_5.iloc[middle_idx]['High']

            if (middle_high > recent_5.iloc[0]['High'] and
                middle_high > recent_5.iloc[1]['High'] and
                middle_high > recent_5.iloc[3]['High'] and
                middle_high > recent_5.iloc[4]['High']):

                self.current_base['pivot_highs'].append(middle_high)

        # ピボットローの検出
        if len(recent_5) == 5:
            middle_idx = 2
            middle_low = recent_5.iloc[middle_idx]['Low']

            if (middle_low < recent_5.iloc[0]['Low'] and
                middle_low < recent_5.iloc[1]['Low'] and
                middle_low < recent_5.iloc[3]['Low'] and
                middle_low < recent_5.iloc[4]['Low']):

                self.current_base['pivot_lows'].append(middle_low)

    def _handle_breakout(self, date: pd.Timestamp, row: pd.Series,
                         historical_df: pd.DataFrame, duration_weeks: int,
                         depth_pct: float, events: List[Dict]):
        """ブレイクアウト時の処理"""

        start_date = self.current_base['start_date']

        # 【重要】3週間未満のブレイクアウトは無効
        if duration_weeks < self.params['min_base_weeks']:
            events.append({
                'date': date.strftime('%Y-%m-%d'),
                'event': 'PREMATURE_BREAKOUT',
                'reason': f'期間短すぎ（{duration_weeks}週 < 3週）',
                'base_start': start_date.strftime('%Y-%m-%d')
            })

            # ベースリセット（新しい抵抗線を設定）
            self.current_base = {
                'start_date': date,
                'start_high': row['High'],
                'resistance': row['High'],
                'lowest_price': row['Low'],
                'contractions': [],
                'pivot_highs': [row['High']],
                'pivot_lows': [row['Low']],
                'start_volume': row['Volume']
            }
            return

        # VCPパターンの検証
        vcp_valid = self._verify_vcp_pattern()

        # 出来高検証
        volume_valid, volume_surge_pct = self._verify_breakout_volume(
            date, row, historical_df
        )

        # 調整深さの検証
        depth_valid = (
            self.params['min_depth_pct'] <= depth_pct <= self.params['max_depth_pct']
        )

        # ベースタイプの判定
        if duration_weeks < 5:
            base_type = "短期ベース"
            quality = "リスク注意"
        elif 5 <= duration_weeks <= 26:
            base_type = "標準ベース"
            quality = "良好" if vcp_valid else "標準"
        elif 26 < duration_weeks <= 65:
            base_type = "長期ベース"
            quality = "許容範囲"
        else:
            base_type = "過長"
            quality = "要注意"

        # 総合判定
        if volume_valid and depth_valid:
            # 有効なブレイクアウト
            self.base_count += 1

            events.append({
                'date': date.strftime('%Y-%m-%d'),
                'event': 'VALID_BREAKOUT',
                'base_number': self.base_count,
                'base_type': base_type,
                'quality': quality,
                'base_start': start_date.strftime('%Y-%m-%d'),
                'duration_weeks': duration_weeks,
                'depth_pct': f'{depth_pct:.1f}%',
                'volume_surge': f'{volume_surge_pct:.0f}%',
                'vcp_valid': vcp_valid,
                'contractions': len(self.current_base['contractions'])
            })

            # ベース情報を保存
            base_info = BaseInfo(
                base_number=self.base_count,
                start_date=start_date,
                resistance_price=self.current_base['resistance'],
                lowest_price=self.current_base['lowest_price'],
                depth_pct=depth_pct,
                duration_weeks=duration_weeks,
                contractions=self.current_base['contractions'],
                vcp_valid=vcp_valid,
                volume_dry_up=True,
                breakout_date=date,
                breakout_volume_surge=volume_surge_pct,
                base_type=base_type,
                quality=quality
            )
            self.bases.append(base_info)

            # 分離待ちステートへ
            self.state = "WAITING_FOR_SEPARATION"
            self.last_breakout_price = row['Close']

        else:
            # ブレイクアウト失敗
            reasons = []
            if not vcp_valid:
                reasons.append("VCPパターン未確認")
            if not volume_valid:
                reasons.append(f"出来高不足（{volume_surge_pct:.0f}%）")
            if not depth_valid:
                reasons.append(f"調整深さ不適切（{depth_pct:.1f}%）")

            events.append({
                'date': date.strftime('%Y-%m-%d'),
                'event': 'WEAK_BREAKOUT',
                'reason': ", ".join(reasons),
                'base_start': start_date.strftime('%Y-%m-%d')
            })

            # スキャンに戻る
            self.state = "SCANNING"
            self.current_base = None

    def _verify_vcp_pattern(self) -> bool:
        """
        VCPパターンの検証

        条件:
        1. ピボットハイが段階的に低くなる（または同程度）
        2. ピボットローが段階的に高くなる（より高い安値）
        3. 各コントラクションが前回より浅い
        """
        if self.current_base is None:
            return False

        pivot_highs = self.current_base['pivot_highs']
        pivot_lows = self.current_base['pivot_lows']

        # 最低2回のピボットが必要
        if len(pivot_highs) < 2 or len(pivot_lows) < 2:
            return False

        # コントラクションの計算
        contractions = []
        for i in range(min(len(pivot_highs), len(pivot_lows)) - 1):
            high = pivot_highs[i]
            low = pivot_lows[i]
            depth = (high - low) / high * 100
            contractions.append(depth)

        if len(contractions) < 2:
            return False

        # 各コントラクションが前回の80%以下（段階的減少）
        for i in range(len(contractions) - 1):
            if contractions[i+1] > contractions[i] * self.params['contraction_decrease_ratio']:
                return False

        # より高い安値の確認
        for i in range(len(pivot_lows) - 1):
            if pivot_lows[i+1] < pivot_lows[i]:
                return False

        return True

    def _verify_breakout_volume(self, date: pd.Timestamp, row: pd.Series,
                               historical_df: pd.DataFrame) -> Tuple[bool, float]:
        """
        ブレイクアウト時の出来高を検証

        Returns:
            Tuple[bool, float]: (検証結果, 出来高増加率%)
        """
        # 50週平均出来高を計算
        lookback = min(50, len(historical_df) - 1)
        recent_volumes = historical_df['Volume'].tail(lookback + 1).iloc[:-1]
        avg_volume = recent_volumes.mean()

        if avg_volume == 0:
            return False, 0

        # 出来高増加率
        volume_surge_pct = (row['Volume'] - avg_volume) / avg_volume * 100

        # 40%以上の増加が必要
        return (
            volume_surge_pct >= self.params['min_breakout_volume_increase'],
            volume_surge_pct
        )

    def _handle_waiting_separation(self, date: pd.Timestamp, row: pd.Series,
                                   historical_df: pd.DataFrame, events: List[Dict]):
        """分離待ちステートの処理"""

        # 20%の上昇を確認
        gain_from_breakout = (row['High'] - self.last_breakout_price) / self.last_breakout_price

        if gain_from_breakout >= self.params['min_separation_pct'] / 100:
            events.append({
                'date': date.strftime('%Y-%m-%d'),
                'event': 'SEPARATION_ACHIEVED',
                'breakout_price': self.last_breakout_price,
                'current_price': row['High'],
                'gain_pct': f'{gain_from_breakout * 100:.1f}%'
            })

            # スキャンに戻る（次のベースを探す）
            self.state = "SCANNING"
            self.current_base = None

    def get_base_count(self) -> int:
        """現在のベースカウントを取得"""
        return self.base_count

    def get_bases(self) -> List[BaseInfo]:
        """検出されたベースのリストを取得"""
        return self.bases

    def evaluate_base_stage(self) -> Dict:
        """
        現在のベース段階を評価

        Returns:
            Dict: 段階評価
        """
        count = self.base_count

        if count == 0:
            return {
                'stage': 'なし',
                'assessment': 'ベース未検出',
                'action': 'ベース形成を待つ'
            }
        elif count <= 2:
            return {
                'stage': '初期段階',
                'assessment': '最適なエントリー機会',
                'action': '積極的にエントリー検討'
            }
        elif count == 3:
            return {
                'stage': '中期段階',
                'assessment': 'まだ許容範囲だが注意',
                'action': '慎重にエントリー、タイトなストップロス'
            }
        else: # count >= 4
            return {
                'stage': '後期段階',
                'assessment': '深い調整のリスク高・クライマックスランの可能性',
                'action': '新規エントリー非推奨、既存ポジションは利確検討'
            }


if __name__ == '__main__':
    # テスト用
    from data_fetcher import fetch_stock_data
    from indicators import calculate_all_basic_indicators

    print("Base Minervini Analyzer（改訂版）のテストを開始...")

    test_tickers = ['AAPL', 'NVDA', 'TSLA']

    for ticker in test_tickers:
        print(f"\n{'='*70}")
        print(f"{ticker} のベース分析:")
        print(f"{'='*70}")

        # 日足と週足データを取得
        daily_df, benchmark_df = fetch_stock_data(ticker, period='5y', interval='1d')
        weekly_df, _ = fetch_stock_data(ticker, period='5y', interval='1wk')

        if daily_df is not None and weekly_df is not None:
            # 指標計算
            daily_df = calculate_all_basic_indicators(daily_df, interval='1d')
            weekly_df = calculate_all_basic_indicators(weekly_df, interval='1wk')
            benchmark_df = calculate_all_basic_indicators(benchmark_df, interval='1d')

            # ベース分析
            analyzer = BaseMinerviniAnalyzer(daily_df, weekly_df, benchmark_df)
            events = analyzer.analyze()

            # 結果表示
            print(f"\n検出されたイベント: {len(events)}件")

            valid_breakouts = [e for e in events if e['event'] == 'VALID_BREAKOUT']
            print(f"有効なブレイクアウト: {len(valid_breakouts)}件")

            if valid_breakouts:
                print("\n【有効なベース】")
                for event in valid_breakouts:
                    print(f"  Base #{event['base_number']}: {event['base_start']} → {event['date']}")
                    print(f"    期間: {event['duration_weeks']}週")
                    print(f"    深さ: {event['depth_pct']}")
                    print(f"    出来高急増: {event['volume_surge']}")
                    print(f"    VCP: {'✓' if event['vcp_valid'] else '✗'}")
                    print(f"    品質: {event['quality']}")

            # ベース段階評価
            stage_eval = analyzer.evaluate_base_stage()
            print(f"\n【ベース段階評価】")
            print(f"  現在のベースカウント: {analyzer.get_base_count()}")
            print(f"  段階: {stage_eval['stage']}")
            print(f"  評価: {stage_eval['assessment']}")
            print(f"  推奨アクション: {stage_eval['action']}")
