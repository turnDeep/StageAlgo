import pandas as pd
from indicators import calculate_all_indicators
from data_fetcher import fetch_stock_data

class StageAnalysisSystem:
    """
    Stan Weinsteinのステージ分析理論に基づき、株式のステージを分析し、
    ステージ移行のスコアリングを行うシステム。
    """
    def __init__(self, indicators_df: pd.DataFrame, ticker: str, benchmark_indicators_df: pd.DataFrame):
        """
        Args:
            indicators_df (pd.DataFrame): テクニカル指標が計算済みの株価データ。
            ticker (str): 分析対象のティッカーシンボル。
            benchmark_indicators_df (pd.DataFrame): ベンチマークの指標計算済みデータ。
        """
        if indicators_df.empty:
            raise ValueError("指標データフレームが空です。分析を実行できません。")

        self.indicators_df = indicators_df
        self.ticker = ticker
        self.benchmark_indicators_df = benchmark_indicators_df

        self.latest_data = indicators_df.iloc[-1]
        self.latest_benchmark_data = benchmark_indicators_df.iloc[-1]
        self.analysis_date = self.latest_data.name.strftime('%Y-%m-%d')

    def _determine_current_stage(self) -> int:
        """
        Geminiの分析に基づき、歴史的背景を考慮して現在の市場ステージを判断します。
        1. MAの傾きで主要トレンド（ステージ2 or 4）を判断。
        2. MAが横ばいなら、価格の歴史的位置でステージ1と3を区別する。
        """
        price = self.latest_data['Close']
        ma50_slope = self.latest_data['ma50_slope']
        slope_threshold = 0.002 # 傾きが「横ばい」かどうかを判断する閾値

        # ステップ1: MAの傾きで主要トレンドを判断
        if ma50_slope > slope_threshold:
            # 傾きが明確な上昇 → ステージ2の可能性が高い
            return 2

        if ma50_slope < -slope_threshold:
            # 傾きが明確な下降 → ステージ4の可能性が高い
            return 4

        # ステップ2: MAが横ばいの場合、価格の歴史的背景で判断（改善版）
        if abs(ma50_slope) <= slope_threshold:
            # 長期（1年）と中期（150日）のデータを取得
            history_1y = self.indicators_df.iloc[-252:-1]
            history_150d = self.indicators_df.iloc[-151:-1]

            if history_1y.empty or len(history_1y) < 200:
                return 1 # データ不足時はデフォルトでステージ1

            high_1y = history_1y['High'].max()
            high_150d = history_150d['High'].max() if not history_150d.empty else high_1y

            # 1. 長期的な底値圏か？ (1年高値から40%以上下落)
            is_long_term_base = price < high_1y * 0.6
            if is_long_term_base:
                return 1 # 長期的な下落後の底固めと判断

            # 2. 中期的な天井圏か？ (150日高値から30%未満の下落)
            is_medium_term_top = price >= high_150d * 0.7
            if is_medium_term_top:
                return 3 # 高値圏での横ばいと判断

            # 上記の中間的な状態はステージ1（底固め）と見なす
            return 1

        # 上記のいずれにも当てはまらない場合のデフォルト
        return 1

    def _score_stage1_to_2_improved(self) -> dict:
        """ステージ1→2（上昇期への移行）のスコアを計算します（改善版ロジック）。"""
        score = 0
        details = {}

        # 1. 出来高
        volume_ratio = self.latest_data['Volume'] / self.latest_data['volume_ma50']
        if volume_ratio >= 2.5:
            score += 25; details['出来高'] = f"A評価 ({volume_ratio:.1f}倍, 25点)"
        elif volume_ratio >= 2.0:
            score += 20; details['出来高'] = f"B評価 ({volume_ratio:.1f}倍, 20点)"
        else:
            details['出来高'] = f"C評価 ({volume_ratio:.1f}倍, 0点)"

        # 2. 価格ブレイク
        price_50day_high = self.indicators_df['Close'].tail(51).iloc[:-1].max()
        current_close = self.latest_data['Close']
        if current_close > price_50day_high * 1.03:
            score += 25; details['価格ブレイク'] = "A評価 (50日高値+3%超, 25点)"
        elif current_close > price_50day_high:
            score += 20; details['価格ブレイク'] = "B評価 (50日高値更新, 20点)"
        else:
            details['価格ブレイク'] = "C評価 (0点)"

        # 3. MA転換
        price_over_ma50 = self.latest_data['Close'] > self.latest_data['ma50']
        ma50_slope_up = self.latest_data['ma50_slope'] > 0.002 # 少し傾きがあることを確認
        if price_over_ma50 and ma50_slope_up:
            score += 20; details['MA転換'] = "A評価 (価格>MA50 & 傾きが正, 20点)"
        elif price_over_ma50 or ma50_slope_up:
            score += 10; details['MA転換'] = "B評価 (どちらか一方, 10点)"
        else:
            details['MA転換'] = "C評価 (0点)"

        # 4. RS Rating
        rs = self.latest_data['rs_rating']
        if rs >= 85:
            score += 15; details['RS Rating'] = f"A評価 ({rs:.0f} >= 85, 15点)"
        elif rs >= 70:
            score += 10; details['RS Rating'] = f"B評価 (70 <= {rs:.0f} < 85, 10点)"
        else:
            details['RS Rating'] = f"C評価 ({rs:.0f} < 70, 0点)"

        # 5. 市場環境
        is_bull_market = self.latest_benchmark_data['Close'] > self.latest_benchmark_data['ma200']
        if is_bull_market:
            score += 10; details['市場環境'] = "強気市場 (+10点)"
        else:
            details['市場環境'] = "弱気/中立市場 (+0点)"

        # 6. 確認メカニズム
        is_confirmed = False
        confirmation_status = "未確認"
        for i in range(1, 6):
            if len(self.indicators_df) < (50 + i + 1): continue

            breakout_day_index = -(i)
            breakout_day_data = self.indicators_df.iloc[breakout_day_index]
            historical_data = self.indicators_df.iloc[breakout_day_index - 50 : breakout_day_index]
            if historical_data.empty: continue

            price_50d_high_before = historical_data['Close'].max()

            if breakout_day_data['Close'] > price_50d_high_before:
                days_since = i - 1
                if days_since >= 2:
                    days_to_confirm_df = self.indicators_df.iloc[-days_since:]
                    if (days_to_confirm_df['Close'] > price_50d_high_before).all():
                        is_confirmed = True
                        confirmation_status = f"確認済み (ブレイクアウト後{days_since}日維持)"
                        break
                else:
                    confirmation_status = f"未確認 (ブレイクアウト後{days_since}日)"
                    break
        details['確認'] = confirmation_status

        # 最終判定
        if is_confirmed and score >= 90:
            level = "A判定 (強力な移行シグナル)"
            action = "自信を持ってエントリーを検討するべき理想的なブレイクアウト。"
        elif is_confirmed and score >= 75:
            level = "B判定 (移行シグナル)"
            action = "エントリーを検討。リスク管理のためポジションサイズ調整も考慮。"
        else:
            level = "C判定 (準備段階)"
            if score >= 75 and not is_confirmed:
                action = f"ブレイクアウトの可能性あり(スコア: {score})。ただし、{confirmation_status}のため、判定は見送り。"
            else:
                action = f"エントリーは見送り、全ての条件が揃うのを待つ (スコア: {score})。"

        return {"score": score, "level": level, "action": action, "details": details}

    def _score_stage2_to_3(self) -> dict:
        """ステージ2→3（天井圏への移行）のスコアを計算します。"""
        score = 0
        recent_data = self.indicators_df.tail(20)
        down_days_high_volume = recent_data[(recent_data['Close'] < recent_data['Close'].shift(1)) & (recent_data['Volume'] > recent_data['volume_ma50'] * 1.5)]
        if len(down_days_high_volume) >= 2: score += 30
        upper_wick = self.indicators_df['High'] - self.indicators_df[['Open', 'Close']].max(axis=1)
        if (upper_wick.tail(5) > (self.indicators_df['High'] - self.indicators_df['Low']).tail(5) * 0.5).any(): score += 25
        if abs(self.latest_data['ma50_slope']) < 0.001: score += 20
        if self.latest_data['rs_rating'] < self.latest_data['rs_rating_ma10']: score += 15
        if self.latest_data['Close'] < self.latest_data['vwap']: score += 10
        if score >= 75: level, action = "危険 (ステージ3移行が濃厚)", "ポジションの大部分の利益確定を強く推奨。"
        elif score >= 50: level, action = "警告 (トレンド鈍化)", "新規の買いは見送り、一部を利益確定。"
        else: level, action = "安全 (ステージ2継続)", "ポジションを維持し、トレンドの継続を期待する。"
        return {"score": score, "level": level, "action": action}

    def _score_stage3_to_4(self) -> dict:
        """ステージ3→4（下降期への移行）のスコアを計算します。"""
        score = 0
        if self.latest_data['Close'] < self.indicators_df['Low'].tail(50).min(): score += 30
        if self.latest_data['Volume'] >= self.latest_data['volume_ma50'] * 2.0: score += 30
        if self.latest_data['Close'] < self.latest_data['ma50'] and self.latest_data['ma50_slope'] < 0: score += 25
        if self.latest_data['rs_rating'] <= 40: score += 15
        if score >= 75: level, action = "危険 (ステージ4確定)", "全てのポジションを決済し、損失を限定する。"
        else: level, action = "警告 (ステージ3継続)", "レンジ内での推移。保有は避け、ブレイクダウンに最大限警戒する。"
        return {"score": score, "level": level, "action": action}

    def _score_stage4_to_1(self) -> dict:
        """ステージ4→1（底固め期への移行）のスコアを計算します。"""
        score = 0
        if abs(self.latest_data['ma50_slope']) < 0.001 and not (self.latest_data['Low'] < self.indicators_df['Low'].tail(20).min()): score += 35
        if self.latest_data['Volume'] < self.latest_data['volume_ma50'] * 0.7: score += 30
        volatility = self.indicators_df['Close'].rolling(20).std() / self.indicators_df['Close'].rolling(20).mean()
        if volatility.iloc[-1] < volatility.rolling(100).mean().iloc[-1]: score += 20
        if self.latest_data['rs_rating'] > self.latest_data['rs_rating_ma10']: score += 15
        if score >= 70: level, action = "底打ち完了 (ステージ1移行)", "監視リストに追加し、将来のステージ2への移行を待つ。"
        else: level, action = "下降継続 (ステージ4継続)", "底打ちの条件は未達。"
        return {"score": score, "level": level, "action": action}

    def analyze(self) -> dict:
        """株式のステージ分析を実行し、結果を返します。"""
        current_stage = self._determine_current_stage()
        transition_analysis = {}
        if current_stage == 1:
            transition_analysis = self._score_stage1_to_2_improved()
            transition_analysis['target_transition'] = "ステージ1 → 2 (改善版)"
        elif current_stage == 2:
            transition_analysis = self._score_stage2_to_3()
            transition_analysis['target_transition'] = "ステージ2 → 3"
        elif current_stage == 3:
            transition_analysis = self._score_stage3_to_4()
            transition_analysis['target_transition'] = "ステージ3 → 4"
        elif current_stage == 4:
            transition_analysis = self._score_stage4_to_1()
            transition_analysis['target_transition'] = "ステージ4 → 1"
        return {"ticker": self.ticker, "analysis_date": self.analysis_date, "current_stage": f"ステージ{current_stage}", "transition_analysis": transition_analysis}

if __name__ == '__main__':
    test_ticker = 'CAN'
    print(f"--- {test_ticker} のステージ分析を開始 ---")
    stock_df, benchmark_df = fetch_stock_data(test_ticker)
    if stock_df is not None and benchmark_df is not None:
        stock_indicators_df = calculate_all_indicators(stock_df, benchmark_df)
        # RS Rating計算のために、ベンチマークの'Close'のみを持つDataFrameを渡す
        benchmark_indicators_df = calculate_all_indicators(benchmark_df, benchmark_df.copy())

        if not stock_indicators_df.empty and not benchmark_indicators_df.empty:
            analyzer = StageAnalysisSystem(stock_indicators_df, test_ticker, benchmark_indicators_df)
            result = analyzer.analyze()

            print(f"ティッカー: {result['ticker']}")
            print(f"分析日: {result['analysis_date']}")
            print(f"現在の推定ステージ: {result['current_stage']}")
            print("\n--- 移行分析 ---")
            analysis = result['transition_analysis']
            print(f"分析対象: {analysis['target_transition']}")
            print(f"スコア: {analysis['score']} / 100")
            print(f"判定: {analysis['level']}")
            print(f"推奨アクション: {analysis['action']}")
            if 'details' in analysis:
                print("\n[スコア詳細]")
                for key, value in analysis['details'].items():
                    print(f"  - {key}: {value}")
        else:
            print("指標計算後にデータが空になりました。")
    else:
        print(f"{test_ticker} のデータ取得に失敗しました。")