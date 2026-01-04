---
title: "PC画面キャプチャとGemini 2.0 Flashによる全自動日報生成: 実装と備忘録"
date: 2026-01-05
slug: daily_screenshot_activity_summary
draft: false
math: false
authors:
    - yonesuke
categories:
    - Python
    - AI
---

PCの画面を1分おきにキャプチャし、Gemini 2.0 Flashを利用して1日の活動内容を自動で要約するツールを作成した際のメモ。

!!! warning
    この記事は Google Antigravity を使用して作成されました。
    あくまで私自身の勉強した結果の備忘録としてのメモと思っていただければと思います。
    （正確性はかならずしも担保されません。）
    作成過程で知らないことが多くあり、非常に勉強になりました。

<!-- more -->

## 概要
一定間隔でスクリーンショットを取得・保存し、それらをまとめてGemini APIに送信して活動レポート（Markdown）を生成する。

## 環境・構成要素
-   **OS**: Windows
-   **言語**: Python 3.12
-   **管理ツール**: `uv`
-   **主なライブラリ**:
    -   `mss`: スクリーンショット撮影
    -   `schedule`: 定期実行
    -   `google-genai`: 画像分析・要約 (Gemini 2.0 Flash)
    -   `Pillow`: 画像読み込み

## 実装内容

### 1. スクリーンショット撮影 (`capture.py`)
`mss` を使用して全モニターのスクリーンショットを取得する。`schedule` で1分間隔で実行。
保存先は日付ごとのディレクトリ (`YYYY-MM-DD`)。

```python
def take_screenshot():
    """全モニターをキャプチャして日付フォルダに保存"""
    with mss.mss() as sct:
        folder = get_daily_folder()
        filename = datetime.now().strftime("%H-%M-%S.png")
        filepath = os.path.join(folder, filename)
        sct.shot(mon=-1, output=filepath)

def start_capture_loop():
    schedule.every(1).minutes.do(take_screenshot)
    while True:
        schedule.run_pending()
        time.sleep(1)
```

### 2. 画像分析・レポート生成 (`analyze.py`)
保存された画像を読み込み、Gemini 2.0 Flash APIに送信する。
画像枚数が多い場合（API制限やトークン節約のため）、必要に応じてサンプリングする処理を含める。

```python
    # 画像リスト取得
    image_paths = sorted(glob.glob(os.path.join(folder, "*.png")))
    
    # 枚数が多い場合は間引き (例: 最大300枚)
    if len(image_paths) > MAX_IMAGES:
         selected_paths = image_paths[::step]
    
    # プロンプト構成
    prompt = (
        "以下の画像は、あるユーザーのPC画面を1分ごとに記録した時系列のスクリーンショットです。"
        "これらの画像を見て、ユーザーが今日どのような作業や活動を行っていたかを詳細に分析し、"
        "その日の活動レポートとして日本語でまとめてください。"
        "時系列に沿った活動の流れと、主な作業内容（プログラミング、ブラウジング、動画視聴など）を含めてください。"
    )

    # API送信
    client = genai.Client(api_key=API_KEY)
    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=[prompt, *images]
    )
```

## 今後の課題
-   タスクトレイ常駐アプリ化 (GUIでの操作)
-   無操作時（アイドル時）の自動停止処理の実装
-   保存日数の管理・古いデータの自動削除
