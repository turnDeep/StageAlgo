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

    def __init__(self, width: int = 1920, height: int = 1200):
        """
        Args:
            width: 画像の幅
            height: 画像の高さ
        """
        self.width = width
        self.height = height
        self.bg_color = (248, 249, 250)  # #f8f9fa
        self.text_color = (33, 37, 41)    # #212529
        self.border_color = (222, 226, 230)  # #dee2e6
        self.positive_color = (40, 167, 69)  # #28a745
        self.negative_color = (220, 53, 69)  # #dc3545
        self.neutral_color = (108, 117, 125)  # #6c757d

        # フォント設定（システムにインストールされているフォントを使用）
        try:
            self.font_title = ImageFont.truetype("Arial.ttf", 24)
            self.font_header = ImageFont.truetype("Arial.ttf", 16)
            self.font_normal = ImageFont.truetype("Arial.ttf", 12)
            self.font_small = ImageFont.truetype("Arial.ttf", 10)
        except:
            # フォントが見つからない場合はデフォルトフォント
            self.font_title = ImageFont.load_default()
            self.font_header = ImageFont.load_default()
            self.font_normal = ImageFont.load_default()
            self.font_small = ImageFont.load_default()

    def load_json_data(self, json_file: str) -> Dict:
        """JSONファイルからデータを読み込む"""
        with open(json_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def generate_dashboard_image(self, json_file: str, output_file: str = 'market_dashboard.png'):
        """
        ダッシュボード画像を生成

        Args:
            json_file: 入力JSONファイルパス
            output_file: 出力PNGファイルパス
        """
        # データ読み込み
        data = self.load_json_data(json_file)

        # 画像とDrawオブジェクトを作成
        image = Image.new('RGB', (self.width, self.height), self.bg_color)
        draw = ImageDraw.Draw(image)

        # 各セクションを描画
        y_offset = 20

        # ヘッダー
        y_offset = self._draw_header(draw, data, y_offset)

        # Market Exposure
        y_offset = self._draw_market_exposure(draw, data['exposure'], y_offset)

        # メインコンテンツエリア（左側と右側）
        left_width = int(self.width * 0.7)
        right_x = left_width + 20

        # 左側: Market Performance, Sectors, Macro
        left_y = y_offset
        left_y = self._draw_performance_table(
            draw, data['market_performance'],
            "Market Performance Overview",
            20, left_y, left_width - 40
        )

        left_y = self._draw_performance_table(
            draw, data['sectors_performance'],
            "Sectors Performance",
            20, left_y + 20, left_width - 40
        )

        left_y = self._draw_performance_table(
            draw, data['macro_performance'],
            "Macro Performance",
            20, left_y + 20, left_width - 40
        )

        # 右側: Screener Results
        if data.get('screener_results'):
            self._draw_screener_results(
                draw, data['screener_results'],
                right_x, y_offset, self.width - right_x - 20
            )

        # 画像を保存
        image.save(output_file, 'PNG')
        print(f"✓ Dashboard image saved to: {output_file}")

    def _draw_header(self, draw: ImageDraw, data: Dict, y: int) -> int:
        """ヘッダーを描画"""
        title = "Market Dashboard"
        date_str = data.get('generated_at', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        # タイトル
        draw.text((20, y), title, fill=self.text_color, font=self.font_title)

        # 日付
        draw.text((20, y + 30), f"Generated: {date_str}",
                 fill=self.neutral_color, font=self.font_small)

        return y + 60

    def _draw_market_exposure(self, draw: ImageDraw, exposure: Dict, y: int) -> int:
        """Market Exposureセクションを描画"""
        box_height = 120
        box_width = 400
        x = 20

        # 背景ボックス
        draw.rectangle(
            [x, y, x + box_width, y + box_height],
            fill=(255, 255, 255),
            outline=self.border_color,
            width=2
        )

        # タイトル
        draw.text((x + 10, y + 10), "Market Exposure",
                 fill=self.text_color, font=self.font_header)

        # スコア
        score = exposure.get('score', 0)
        level = exposure.get('level', 'N/A')

        score_text = f"{score:.1f}%"
        draw.text((x + 10, y + 40), score_text,
                 fill=self.text_color, font=self.font_title)

        # レベル
        level_color = self._get_level_color(level)
        draw.text((x + 10, y + 70), level,
                 fill=level_color, font=self.font_header)

        # 矢印（レベルに応じて）
        arrow_x = x + box_width - 60
        arrow_y = y + 50
        self._draw_level_arrow(draw, level, arrow_x, arrow_y)

        return y + box_height + 20

    def _get_level_color(self, level: str) -> Tuple[int, int, int]:
        """レベルに応じた色を取得"""
        if level == 'Bullish':
            return self.positive_color
        elif level == 'Positive':
            return (144, 238, 144)  # ライトグリーン
        elif level == 'Neutral':
            return (255, 193, 7)  # イエロー
        elif level == 'Negative':
            return (255, 140, 0)  # オレンジ
        elif level == 'Bearish':
            return self.negative_color
        else:
            return self.neutral_color

    def _draw_level_arrow(self, draw: ImageDraw, level: str, x: int, y: int):
        """レベルに応じた矢印を描画"""
        arrow_size = 30

        if level in ['Bullish', 'Positive']:
            # 上向き矢印
            points = [
                (x, y + arrow_size),
                (x + arrow_size // 2, y),
                (x + arrow_size, y + arrow_size)
            ]
            draw.polygon(points, fill=self.positive_color)
        elif level in ['Bearish', 'Negative']:
            # 下向き矢印
            points = [
                (x, y),
                (x + arrow_size // 2, y + arrow_size),
                (x + arrow_size, y)
            ]
            draw.polygon(points, fill=self.negative_color)
        else:
            # 横向き矢印（Neutral）
            points = [
                (x, y + arrow_size // 2),
                (x + arrow_size, y + arrow_size // 2)
            ]
            draw.line(points, fill=self.neutral_color, width=3)

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
        y += 30

        # テーブルの設定
        row_height = 25
        col_widths = {
            'ticker': 80,
            'price': 60,
            '% 1D': 60,
            'RS STS %': 70,
            '% YTD': 60,
            '% 1W': 60,
            '% 1M': 60,
            '% 1Y': 60,
        }

        # ヘッダー行
        current_x = x
        header_y = y

        # 背景
        draw.rectangle(
            [x, header_y, x + width, header_y + row_height],
            fill=(240, 240, 240),
            outline=self.border_color
        )

        # ヘッダーテキスト
        for col, col_width in col_widths.items():
            draw.text((current_x + 5, header_y + 5), col,
                     fill=self.text_color, font=self.font_small)
            current_x += col_width

        y += row_height

        # データ行
        for i, row in enumerate(df_data[:10]):  # 最大10行
            current_x = x
            row_y = y + i * row_height

            # 背景（交互に色を変える）
            bg_color = (255, 255, 255) if i % 2 == 0 else (250, 250, 250)
            draw.rectangle(
                [x, row_y, x + width, row_y + row_height],
                fill=bg_color,
                outline=self.border_color
            )

            # Ticker
            ticker = row.get('ticker', '')
            draw.text((current_x + 5, row_y + 5), ticker,
                     fill=self.text_color, font=self.font_small)
            current_x += col_widths['ticker']

            # Price
            price = row.get('price', 0)
            draw.text((current_x + 5, row_y + 5), f"${price:.2f}",
                     fill=self.text_color, font=self.font_small)
            current_x += col_widths['price']

            # % 1D
            pct_1d = row.get('% 1D', 0)
            color = self.positive_color if pct_1d > 0 else self.negative_color
            draw.text((current_x + 5, row_y + 5), f"{pct_1d:+.2f}%",
                     fill=color, font=self.font_small)
            current_x += col_widths['% 1D']

            # RS STS %
            rs_sts = row.get('RS STS %', 0)
            # バーグラフを描画
            bar_width = int((rs_sts / 100) * (col_widths['RS STS %'] - 10))
            draw.rectangle(
                [current_x + 5, row_y + 8, current_x + 5 + bar_width, row_y + row_height - 8],
                fill=(0, 123, 255),
                outline=None
            )
            draw.text((current_x + 10, row_y + 5), f"{rs_sts:.0f}",
                     fill=(255, 255, 255), font=self.font_small)
            current_x += col_widths['RS STS %']

            # その他のパーセンテージ列
            for col in ['% YTD', '% 1W', '% 1M', '% 1Y']:
                pct_val = row.get(col, 0)
                color = self.positive_color if pct_val > 0 else self.negative_color
                draw.text((current_x + 5, row_y + 5), f"{pct_val:+.1f}%",
                         fill=color, font=self.font_small)
                current_x += col_widths[col]

        return y + len(df_data[:10]) * row_height + 10

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
        box_height = 100

        for key, name in screener_names.items():
            tickers_data = screener_results.get(key, {}).get('data', [])
            tickers = [t['Ticker'] for t in tickers_data[:10]] if tickers_data else []

            # ボックス描画
            draw.rectangle(
                [x, current_y, x + width, current_y + box_height],
                fill=(255, 255, 255),
                outline=self.border_color,
                width=1
            )

            # タイトル
            draw.text((x + 10, current_y + 10), name,
                     fill=self.text_color, font=self.font_normal)

            # ティッカーリスト
            ticker_text = ', '.join(tickers) if tickers else 'No stocks found'

            # テキストを折り返し
            max_width = width - 20
            wrapped_text = self._wrap_text(ticker_text, self.font_small, max_width)

            text_y = current_y + 35
            for line in wrapped_text[:3]:  # 最大3行
                draw.text((x + 10, text_y), line,
                         fill=self.text_color, font=self.font_small)
                text_y += 15

            current_y += box_height + 10

    def _wrap_text(self, text: str, font: ImageFont, max_width: int) -> List[str]:
        """テキストを指定幅で折り返す"""
        words = text.split(' ')
        lines = []
        current_line = []

        for word in words:
            test_line = ' '.join(current_line + [word])
            # テキスト幅を取得（Pillow 10.0.0以降）
            try:
                bbox = font.getbbox(test_line)
                text_width = bbox[2] - bbox[0]
            except:
                # 古いバージョンのPillowの場合
                text_width = len(test_line) * 6  # 概算

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

    Args:
        json_file: 入力JSONファイル
        output_file: 出力PNG画像ファイル
    """
    generator = DashboardImageGenerator()
    generator.generate_dashboard_image(json_file, output_file)


if __name__ == '__main__':
    # テスト実行
    generate_dashboard_image_from_json()
