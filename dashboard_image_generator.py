# dashboard_image_generator.py
"""
Market Dashboard Image Generator
JSONデータから画像（PNG）を生成
"""

import json
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
import pandas as pd
from typing import Dict, List, Tuple
import os


class DashboardImageGenerator:
    """
    Market DashboardをPNG画像として生成
    """

    def __init__(self, width: int = 2400, height: int = 2000):
        """
        Args:
            width: 画像の幅
            height: 画像の高さ
        """
        self.width = width
        self.height = height
        self.bg_color = (248, 249, 250)
        self.text_color = (33, 37, 41)
        self.border_color = (222, 226, 230)
        self.positive_color = (40, 167, 69)
        self.negative_color = (220, 53, 69)
        self.neutral_color = (108, 117, 125)
        self.bar_color = (0, 123, 255)

        # フォント設定
        try:
            self.font_title = ImageFont.truetype("Arial.ttf", 28)
            self.font_header = ImageFont.truetype("Arial.ttf", 18)
            self.font_normal = ImageFont.truetype("Arial.ttf", 13)
            self.font_small = ImageFont.truetype("Arial.ttf", 11)
            self.font_tiny = ImageFont.truetype("Arial.ttf", 9)
        except:
            self.font_title = ImageFont.load_default()
            self.font_header = ImageFont.load_default()
            self.font_normal = ImageFont.load_default()
            self.font_small = ImageFont.load_default()
            self.font_tiny = ImageFont.load_default()

    def load_json_data(self, json_file: str) -> Dict:
        """JSONファイルからデータを読み込む"""
        with open(json_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def generate_dashboard_image(self, json_file: str, output_file: str = 'market_dashboard.png'):
        """
        ダッシュボード画像を生成
        """
        data = self.load_json_data(json_file)

        image = Image.new('RGB', (self.width, self.height), self.bg_color)
        draw = ImageDraw.Draw(image)

        y_offset = 20

        # ヘッダー
        y_offset = self._draw_header(draw, data, y_offset)

        # Market Exposure
        y_offset = self._draw_market_exposure(draw, data['exposure'], y_offset)

        # メインコンテンツ
        left_width = int(self.width * 0.7)
        right_x = left_width + 20

        # 左側: テーブル
        left_y = y_offset
        left_y = self._draw_performance_table(
            draw, data['market_performance'],
            "Market Performance Overview",
            20, left_y, left_width - 40
        )

        left_y = self._draw_performance_table(
            draw, data['sectors_performance'],
            "Sectors Performance",
            20, left_y + 30, left_width - 40
        )

        left_y = self._draw_performance_table(
            draw, data['macro_performance'],
            "Macro Performance",
            20, left_y + 30, left_width - 40
        )

        # 右側: Screeners
        if data.get('screener_results'):
            self._draw_screener_results(
                draw, data['screener_results'],
                right_x, y_offset, self.width - right_x - 20
            )

        image.save(output_file, 'PNG')
        print(f"✓ Dashboard image saved to: {output_file}")

    def _draw_header(self, draw: ImageDraw, data: Dict, y: int) -> int:
        """ヘッダーを描画"""
        title = "Market Dashboard"
        date_str = data.get('generated_at', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        draw.text((20, y), title, fill=self.text_color, font=self.font_title)
        draw.text((20, y + 35), f"Generated: {date_str}",
                 fill=self.neutral_color, font=self.font_small)

        return y + 70

    def _draw_market_exposure(self, draw: ImageDraw, exposure: Dict, y: int) -> int:
        """Market Exposureセクションを描画"""
        box_height = 140
        box_width = 450
        x = 20

        draw.rectangle(
            [x, y, x + box_width, y + box_height],
            fill=(255, 255, 255),
            outline=self.border_color,
            width=2
        )

        draw.text((x + 15, y + 15), "Market Exposure",
                 fill=self.text_color, font=self.font_header)

        score = exposure.get('score', 0)
        level = exposure.get('level', 'N/A')

        score_text = f"{score:.1f}%"
        draw.text((x + 15, y + 50), score_text,
                 fill=self.text_color, font=self.font_title)

        level_color = self._get_level_color(level)
        draw.text((x + 15, y + 90), level,
                 fill=level_color, font=self.font_header)

        arrow_x = x + box_width - 80
        arrow_y = y + 60
        self._draw_level_arrow(draw, level, arrow_x, arrow_y)

        return y + box_height + 30

    def _get_level_color(self, level: str) -> Tuple[int, int, int]:
        """レベルに応じた色を取得"""
        if level == 'Bullish':
            return self.positive_color
        elif level == 'Positive':
            return (144, 238, 144)
        elif level == 'Neutral':
            return (255, 193, 7)
        elif level == 'Negative':
            return (255, 140, 0)
        elif level == 'Bearish':
            return self.negative_color
        else:
            return self.neutral_color

    def _draw_level_arrow(self, draw: ImageDraw, level: str, x: int, y: int):
        """レベルに応じた矢印を描画"""
        arrow_size = 40

        if level in ['Bullish', 'Positive']:
            points = [
                (x, y + arrow_size),
                (x + arrow_size // 2, y),
                (x + arrow_size, y + arrow_size)
            ]
            draw.polygon(points, fill=self.positive_color)
        elif level in ['Bearish', 'Negative']:
            points = [
                (x, y),
                (x + arrow_size // 2, y + arrow_size),
                (x + arrow_size, y)
            ]
            draw.polygon(points, fill=self.negative_color)
        else:
            points = [
                (x, y + arrow_size // 2),
                (x + arrow_size, y + arrow_size // 2)
            ]
            draw.line(points, fill=self.neutral_color, width=4)

    def _draw_performance_table(self, draw: ImageDraw, data: Dict,
                                title: str, x: int, y: int, width: int) -> int:
        """パフォーマンステーブルを描画"""
        if not data or 'data' not in data:
            return y

        df_data = data['data']
        if not df_data:
            return y

        # タイトル
        draw.text((x, y), title, fill=self.text_color, font=self.font_header)
        y += 35

        # テーブル設定
        row_height = 28
        col_widths = {
            'ticker': 90,
            'price': 70,
            '% 1D': 60,
            'RS': 120,  # RS Historyグラフ用
            'RS STS %': 70,
            '% YTD': 60,
            '% 1W': 55,
            '% 1M': 55,
            '% 1Y': 55,
            '% From 52W': 65,
            # Trend Indicators
            '10MA': 40,
            '20MA': 40,
            '50MA': 40,
            '200MA': 40,
            '20>50': 40,
            '50>200': 45,
        }

        # ヘッダー行
        current_x = x
        header_y = y

        draw.rectangle(
            [x, header_y, x + width, header_y + row_height * 2],
            fill=(240, 240, 240),
            outline=self.border_color
        )

        # ヘッダーテキスト（2行構造）
        # 1行目
        headers_row1 = {
            'Ticker': col_widths['ticker'],
            'Price': col_widths['price'],
            '% 1D': col_widths['% 1D'],
            'Relative\nStrength': col_widths['RS'],
            'RS STS\n%': col_widths['RS STS %'],
        }

        current_x = x
        for header, col_width in headers_row1.items():
            lines = header.split('\n')
            for i, line in enumerate(lines):
                draw.text((current_x + 3, header_y + 3 + i * 12), line,
                         fill=self.text_color, font=self.font_tiny)
            current_x += col_width

        # Performance列の結合ヘッダー
        perf_start_x = current_x
        perf_width = sum([col_widths['% YTD'], col_widths['% 1W'],
                         col_widths['% 1M'], col_widths['% 1Y']])
        draw.text((perf_start_x + perf_width // 2 - 30, header_y + 3),
                 "Performance", fill=self.text_color, font=self.font_tiny)

        # Highs列
        highs_x = perf_start_x + perf_width
        draw.text((highs_x + 3, header_y + 3), "Highs",
                 fill=self.text_color, font=self.font_tiny)

        # Trend Indicators列の結合ヘッダー
        trend_start_x = highs_x + col_widths['% From 52W']
        trend_width = sum([col_widths['10MA'], col_widths['20MA'],
                          col_widths['50MA'], col_widths['200MA'],
                          col_widths['20>50'], col_widths['50>200']])
        draw.text((trend_start_x + trend_width // 2 - 50, header_y + 3),
                 "Trend Indicators (MAs)", fill=self.text_color, font=self.font_tiny)

        # 2行目のヘッダー
        headers_row2_x = perf_start_x
        for header in ['YTD', '1W', '1M', '1Y']:
            col_width = col_widths[f'% {header}']
            draw.text((headers_row2_x + 3, header_y + row_height + 3),
                     header, fill=self.text_color, font=self.font_tiny)
            headers_row2_x += col_width

        # Highsの詳細
        draw.text((headers_row2_x + 3, header_y + row_height + 3),
                 "52W", fill=self.text_color, font=self.font_tiny)
        headers_row2_x += col_widths['% From 52W']

        # Trend Indicatorsの詳細
        for header in ['10MA', '20MA', '50MA', '200MA', '20>50', '50>200']:
            col_width = col_widths[header]
            draw.text((headers_row2_x + 3, header_y + row_height + 3),
                     header, fill=self.text_color, font=self.font_tiny)
            headers_row2_x += col_width

        y += row_height * 2

        # データ行
        for i, row in enumerate(df_data[:10]):
            current_x = x
            row_y = y + i * row_height

            bg_color = (255, 255, 255) if i % 2 == 0 else (250, 250, 250)
            draw.rectangle(
                [x, row_y, x + width, row_y + row_height],
                fill=bg_color,
                outline=self.border_color
            )

            # Ticker
            ticker = row.get('ticker', '')
            draw.text((current_x + 3, row_y + 8), ticker,
                     fill=self.text_color, font=self.font_small)
            current_x += col_widths['ticker']

            # Price
            price = row.get('price', 0)
            draw.text((current_x + 3, row_y + 8), f"${price:.2f}",
                     fill=self.text_color, font=self.font_small)
            current_x += col_widths['price']

            # % 1D
            pct_1d = row.get('% 1D', 0)
            color = self.positive_color if pct_1d > 0 else self.negative_color
            draw.text((current_x + 3, row_y + 8), f"{pct_1d:+.2f}%",
                     fill=color, font=self.font_small)
            current_x += col_widths['% 1D']

            # RS History グラフ（20日分の縦棒）
            rs_history = row.get('RS History', [0] * 20)
            self._draw_rs_history_chart(draw, rs_history,
                                        current_x + 3, row_y + 3,
                                        col_widths['RS'] - 6, row_height - 6)
            current_x += col_widths['RS']

            # RS STS %
            rs_sts = row.get('RS STS %', 0)
            bar_width = int((rs_sts / 100) * (col_widths['RS STS %'] - 10))
            draw.rectangle(
                [current_x + 3, row_y + 10, current_x + 3 + bar_width, row_y + row_height - 10],
                fill=self.bar_color,
                outline=None
            )
            draw.text((current_x + 8, row_y + 8), f"{rs_sts:.0f}",
                     fill=(255, 255, 255) if bar_width > 20 else self.text_color,
                     font=self.font_tiny)
            current_x += col_widths['RS STS %']

            # Performance列
            for col in ['% YTD', '% 1W', '% 1M', '% 1Y']:
                pct_val = row.get(col, 0)
                color = self.positive_color if pct_val > 0 else self.negative_color
                draw.text((current_x + 3, row_y + 8), f"{pct_val:+.1f}%",
                         fill=color, font=self.font_tiny)
                current_x += col_widths[col]

            # % From 52W High
            from_high = row.get('% From 52W High', 0)
            color = self.positive_color if from_high > -10 else self.negative_color
            draw.text((current_x + 3, row_y + 8), f"{from_high:.1f}%",
                     fill=color, font=self.font_tiny)
            current_x += col_widths['% From 52W']

            # Trend Indicators (▲/▼)
            for col in ['10MA', '20MA', '50MA', '200MA', '20>50MA', '50>200MA']:
                is_true = row.get(col, False)
                symbol = "▲" if is_true else "▼"
                color = self.positive_color if is_true else self.negative_color
                draw.text((current_x + 10, row_y + 8), symbol,
                         fill=color, font=self.font_normal)
                col_key = col if col in col_widths else col.replace('MA', '')
                current_x += col_widths.get(col_key, 40)

        return y + len(df_data[:10]) * row_height + 20

    def _draw_rs_history_chart(self, draw: ImageDraw, rs_history: List[float],
                               x: int, y: int, width: int, height: int):
        """
        RS Historyの縦棒グラフを描画（20日分）

        Args:
            draw: ImageDrawオブジェクト
            rs_history: 20日分のRS値リスト
            x, y: グラフの左上座標
            width, height: グラフのサイズ
        """
        if not rs_history or len(rs_history) == 0:
            return

        # 20日分に調整
        rs_history = rs_history[-20:] if len(rs_history) > 20 else rs_history

        # 最大値・最小値を取得
        max_rs = max(rs_history) if rs_history else 1
        min_rs = min(rs_history) if rs_history else -1
        rs_range = max(abs(max_rs), abs(min_rs), 1)  # ゼロ除算防止

        # 各バーの幅
        bar_width = max(1, width // len(rs_history))

        # ゼロ線の位置
        zero_y = y + height // 2

        # 各バーを描画
        for i, rs_val in enumerate(rs_history):
            bar_x = x + i * bar_width

            # バーの高さ（正規化）
            bar_height = int((rs_val / rs_range) * (height // 2))

            if rs_val >= 0:
                # 正の値：上向き（緑）
                bar_y1 = zero_y - bar_height
                bar_y2 = zero_y
                color = self.positive_color
            else:
                # 負の値：下向き（赤）
                bar_y1 = zero_y
                bar_y2 = zero_y - bar_height
                color = self.negative_color

            # バーを描画
            draw.rectangle(
                [bar_x, min(bar_y1, bar_y2), bar_x + bar_width - 1, max(bar_y1, bar_y2)],
                fill=color,
                outline=None
            )

        # ゼロ線を描画
        draw.line([(x, zero_y), (x + width, zero_y)],
                 fill=self.neutral_color, width=1)

    def _draw_screener_results(self, draw: ImageDraw, screener_results: Dict,
                               x: int, y: int, width: int):
        """スクリーナー結果を描画"""
        screener_names = {
            'momentum_97': 'Momentum 97',
            'explosive_eps': 'Explosive EPS Growth',
            'healthy_chart': 'Healthy Chart',
            'up_on_volume': 'Up on Volume',
            'top_2_rs': 'Top 2% RS Rating',
            'bullish_4pct': '4% Bullish Yesterday'
        }

        current_y = y
        box_height = 110

        for key, name in screener_names.items():
            tickers_data = screener_results.get(key, {}).get('data', [])
            tickers = [t['Ticker'] for t in tickers_data[:15]] if tickers_data else []

            draw.rectangle(
                [x, current_y, x + width, current_y + box_height],
                fill=(255, 255, 255),
                outline=self.border_color,
                width=1
            )

            draw.text((x + 10, current_y + 10), name,
                     fill=self.text_color, font=self.font_normal)

            ticker_text = ', '.join(tickers) if tickers else 'No stocks found'

            max_width = width - 20
            wrapped_text = self._wrap_text(ticker_text, self.font_small, max_width)

            text_y = current_y + 40
            for line in wrapped_text[:3]:
                draw.text((x + 10, text_y), line,
                         fill=self.text_color, font=self.font_small)
                text_y += 18

            current_y += box_height + 10

    def _wrap_text(self, text: str, font: ImageFont, max_width: int) -> List[str]:
        """テキストを指定幅で折り返す"""
        words = text.split(' ')
        lines = []
        current_line = []

        for word in words:
            test_line = ' '.join(current_line + [word])
            try:
                bbox = font.getbbox(test_line)
                text_width = bbox[2] - bbox[0]
            except:
                text_width = len(test_line) * 6

            if text_width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]

        if current_line:
            lines.append(' '.join(current_line))

        return lines


def generate_dashboard_image_from_json(
    json_file: str = 'market_dashboard_data.json',
    output_file: str = 'market_dashboard.png'
):
    """
    JSONファイルからダッシュボード画像を生成
    """
    generator = DashboardImageGenerator()
    generator.generate_dashboard_image(json_file, output_file)


if __name__ == '__main__':
    generate_dashboard_image_from_json()
