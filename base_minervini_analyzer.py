import pandas as pd
import json
import numpy as np
import os

class BaseMinerviniAnalyzer:
    def __init__(self, df):
        self.df = df
        self.json_events = []
        self.state = "SCANNING"
        self.base_anchor = {}
        self.ma_short = 50
        self.ma_mid = 150
        self.ma_long = 200

    def analyze(self):
        """
        1日ずつデータを処理し、状態（ステートマシン）に基づいてベース形成イベントを検出する。
        """
        start_index = 252
        if len(self.df) < start_index:
            print("データが不足しています（最低252日分必要です）。")
            return []

        for date, row in self.df.iloc[start_index:].iterrows():
            if self.state == "SCANNING":
                # ステージ判定は外部で行うため、ここでは単純に High_52W をチェック
                if row['High'] == row['High_52W']:
                    self.state = "FORMING"
                    self.base_anchor = {
                        "start_date": date,
                        "start_high": row['High'],
                        "current_low": row['Low'],
                        "duration": 0
                    }
                    self.json_events.append({
                        "date": date.strftime('%Y-%m-%d'),
                        "event": "BASE_START",
                        "resistance_price": self.base_anchor["start_high"]
                    })
            elif self.state == "FORMING":
                self.base_anchor["duration"] += 1
                self.base_anchor["current_low"] = min(self.base_anchor["current_low"], row['Low'])

                resistance_price = self.base_anchor["start_high"]
                current_low = self.base_anchor["current_low"]
                duration_days = self.base_anchor["duration"]
                drawdown = (resistance_price - current_low) / resistance_price

                if drawdown > 0.50:
                    self.json_events.append({
                        "date": date.strftime('%Y-%m-%d'), "event": "BASE_FAILED",
                        "reason": "Too deep (>50%)", "drawdown_pct": f"{drawdown:.2%}",
                        "base_start": self.base_anchor["start_date"].strftime('%Y-%m-%d')
                    })
                    self.state = "SCANNING"
                    continue

                if row['Close'] < row[f'SMA_{self.ma_long}']:
                    self.json_events.append({
                        "date": date.strftime('%Y-%m-%d'), "event": "BASE_FAILED",
                        "reason": "Broke MA200",
                        "base_start": self.base_anchor["start_date"].strftime('%Y-%m-%d')
                    })
                    self.state = "SCANNING"
                    continue

                if row['Close'] > resistance_price:
                    MIN_DURATION_FLAT = 25
                    MIN_DURATION_CUP = 35

                    if duration_days < MIN_DURATION_FLAT:
                        self.json_events.append({
                            "date": date.strftime('%Y-%m-%d'), "event": "PREMATURE_BREAKOUT",
                            "reason": f"Too short ({duration_days} < {MIN_DURATION_FLAT} days)",
                            "base_start": self.base_anchor["start_date"].strftime('%Y-%m-%d')
                        })
                        self.state = "SCANNING"
                        continue

                    is_flat_base = (drawdown < 0.15) and (duration_days >= MIN_DURATION_FLAT)
                    is_cup_base = (0.15 <= drawdown <= 0.50) and (duration_days >= MIN_DURATION_CUP)

                    if is_flat_base or is_cup_base:
                        today_volume = row['Volume']
                        vol_50ma = row[f'Volume_SMA_{self.ma_short}']
                        volume_ratio = today_volume / vol_50ma if vol_50ma > 0 else 0

                        if volume_ratio > 1.5:
                            self.json_events.append({
                                "date": date.strftime('%Y-%m-%d'), "event": "VALID_BREAKOUT",
                                "base_type": "flat_base" if is_flat_base else "cup_base",
                                "base_start": self.base_anchor["start_date"].strftime('%Y-%m-%d'),
                                "duration_days": duration_days,
                                "drawdown_pct": f"{drawdown:.2%}",
                                "volume_ratio": f"{volume_ratio:.2f}x"
                            })
                            self.state = "SCANNING"
                        else:
                            self.json_events.append({
                                "date": date.strftime('%Y-%m-%d'), "event": "WEAK_BREAKOUT",
                                "reason": f"Low volume (Ratio: {volume_ratio:.2f}x)",
                                "base_start": self.base_anchor["start_date"].strftime('%Y-%m-%d')
                            })
                    else:
                        self.json_events.append({
                            "date": date.strftime('%Y-%m-%d'), "event": "IMMATURE_BREAKOUT",
                            "reason": "Duration/Depth mismatch",
                            "duration_days": duration_days,
                            "drawdown_pct": f"{drawdown:.2%}",
                            "base_start": self.base_anchor["start_date"].strftime('%Y-%m-%d')
                        })
                        self.state = "SCANNING"
        return self.json_events
