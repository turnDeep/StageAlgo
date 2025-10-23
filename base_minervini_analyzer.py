import pandas as pd
import json
import numpy as np
import os

class BaseMinerviniAnalyzer:
  """
  Base Minervini Analyzer（修正版）

  【重要な修正点】
  1. 52週高値更新だけではベースとしてカウントしない
  2. ブレイクアウト後に20%以上の分離（separation）を確認
  3. 分離後、再び統合期間に入った場合のみ次のベースとしてカウント

  【ベースの正しい定義】
  - ベース = 統合期間（consolidation）
  - 価格が横ばいになり、出来高が減少する期間
  - その後、高値をブレイクアウトして次の上昇が始まる
  """

  def __init__(self, df):
    self.df = df
    self.json_events = []
    self.state = "SCANNING"
    self.base_anchor = {}
    self.breakout_price = 0
    self.separation_high = 0 # 分離後の高値
    self.last_base_resistance = 0 # 前回のベースのレジスタンス
    self.ma_short = 50
    self.ma_mid = 150
    self.ma_long = 200
    self.min_separation_pct = 0.20 # 20%の分離が必要
    self.min_pullback_pct = 0.08 # 8%以上の調整でベース形成開始

  def analyze(self):
    """
    1日ずつデータを処理し、状態（ステートマシン）に基づいてベース形成イベントを検出する。

    【状態遷移】
    SCANNING → FORMING → WAITING_FOR_SEPARATION → SEPARATION_ACHIEVED → SCANNING

    【重要な変更】
    - SEPARATION_ACHIEVED後に再び調整局面に入った場合のみ、次のベースとしてカウント
    """
    start_index = 252
    if len(self.df) < start_index:
      print("データが不足しています（最低252日分必要です）。")
      return []

    for date, row in self.df.iloc[start_index:].iterrows():
      if self.state == "SCANNING":
        # 新しいベース形成の開始を検出
        # 条件: 前回のベースから20%以上上昇した後、8%以上の調整に入った

        if self.last_base_resistance > 0:
          # 前回のベースから十分に上昇したか確認
          price_gain_from_last_base = (row['High'] - self.last_base_resistance) / self.last_base_resistance

          if price_gain_from_last_base >= self.min_separation_pct:
            # 十分な上昇を確認、次に調整局面を待つ
            self.separation_high = row['High']
            self.state = "WAITING_FOR_PULLBACK"
        else:
          # 初回のベース検出: 52週高値をチェック
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
              "base_number": len([e for e in self.json_events if e['event'] == 'VALID_BREAKOUT']) + 1,
              "resistance_price": self.base_anchor["start_high"]
            })

      elif self.state == "WAITING_FOR_PULLBACK":
        # 分離後の調整を待つ
        pullback_pct = (self.separation_high - row['Low']) / self.separation_high

        if pullback_pct >= self.min_pullback_pct:
          # 8%以上の調整を確認 → 新しいベース形成開始
          self.state = "FORMING"
          self.base_anchor = {
            "start_date": date,
            "start_high": self.separation_high,
            "current_low": row['Low'],
            "duration": 0
          }
          self.json_events.append({
            "date": date.strftime('%Y-%m-%d'),
            "event": "BASE_START",
            "base_number": len([e for e in self.json_events if e['event'] == 'VALID_BREAKOUT']) + 1,
            "resistance_price": self.base_anchor["start_high"],
            "after_separation": True
          })
        elif row['High'] > self.separation_high * 1.02:
          # さらに上昇継続（2%以上）
          self.separation_high = row['High']

      elif self.state == "FORMING":
        self.base_anchor["duration"] += 1
        self.base_anchor["current_low"] = min(self.base_anchor["current_low"], row['Low'])

        resistance_price = self.base_anchor["start_high"]
        current_low = self.base_anchor["current_low"]
        duration_days = self.base_anchor["duration"]
        drawdown = (resistance_price - current_low) / resistance_price

        # 失敗条件のチェック
        if drawdown > 0.50:
          self.json_events.append({
            "date": date.strftime('%Y-%m-%d'),
            "event": "BASE_FAILED",
            "reason": "Too deep (>50%)",
            "drawdown_pct": f"{drawdown:.2%}",
            "base_start": self.base_anchor["start_date"].strftime('%Y-%m-%d')
          })
          self.state = "SCANNING"
          continue

        if row['Close'] < row[f'SMA_{self.ma_long}']:
          self.json_events.append({
            "date": date.strftime('%Y-%m-%d'),
            "event": "BASE_FAILED",
            "reason": "Broke MA200",
            "base_start": self.base_anchor["start_date"].strftime('%Y-%m-%d')
          })
          self.state = "SCANNING"
          continue

        # ブレイクアウトチェック
        if row['Close'] > resistance_price:
          MIN_DURATION_FLAT = 25
          MIN_DURATION_CUP = 35

          if duration_days < MIN_DURATION_FLAT:
            self.json_events.append({
              "date": date.strftime('%Y-%m-%d'),
              "event": "PREMATURE_BREAKOUT",
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
              base_number = len([e for e in self.json_events if e['event'] == 'VALID_BREAKOUT']) + 1
              self.json_events.append({
                "date": date.strftime('%Y-%m-%d'),
                "event": "VALID_BREAKOUT",
                "base_number": base_number,
                "base_type": "flat_base" if is_flat_base else "cup_base",
                "base_start": self.base_anchor["start_date"].strftime('%Y-%m-%d'),
                "duration_days": duration_days,
                "drawdown_pct": f"{drawdown:.2%}",
                "volume_ratio": f"{volume_ratio:.2f}x"
              })
              self.state = "WAITING_FOR_SEPARATION"
              self.breakout_price = resistance_price
              self.last_base_resistance = resistance_price
            else:
              self.json_events.append({
                "date": date.strftime('%Y-%m-%d'),
                "event": "WEAK_BREAKOUT",
                "reason": f"Low volume (Ratio: {volume_ratio:.2f}x)",
                "base_start": self.base_anchor["start_date"].strftime('%Y-%m-%d')
              })
              self.state = "SCANNING"
          else:
            self.json_events.append({
              "date": date.strftime('%Y-%m-%d'),
              "event": "IMMATURE_BREAKOUT",
              "reason": "Duration/Depth mismatch",
              "duration_days": duration_days,
              "drawdown_pct": f"{drawdown:.2%}",
              "base_start": self.base_anchor["start_date"].strftime('%Y-%m-%d')
            })
            self.state = "SCANNING"

      elif self.state == "WAITING_FOR_SEPARATION":
        # 20%の上昇（分離）を待つ
        if row['High'] >= self.breakout_price * (1 + self.min_separation_pct):
          self.json_events.append({
            "date": date.strftime('%Y-%m-%d'),
            "event": "SEPARATION_ACHIEVED",
            "breakout_price": self.breakout_price,
            "separation_price": row['High'],
            "gain_pct": f"{((row['High'] - self.breakout_price) / self.breakout_price * 100):.1f}%"
          })
          self.separation_high = row['High']
          self.state = "WAITING_FOR_PULLBACK"

        # 200日線を下回ったらリセット
        elif row['Close'] < row[f'SMA_{self.ma_long}']:
          self.json_events.append({
            "date": date.strftime('%Y-%m-%d'),
            "event": "SEPARATION_FAILED",
            "reason": "Broke MA200 before 20% gain",
            "breakout_price": self.breakout_price
          })
          self.state = "SCANNING"

    return self.json_events
