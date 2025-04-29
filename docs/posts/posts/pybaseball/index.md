---
title: "MLBデータの可視化"
date: 2025-04-29
slug: pybaseball
draft: false
slug: pybaseball
categories:
    - Baseball
    - Python
authors:
    - yonesuke
---

メジャーリーグを見ていると、Google Cloud提供のStatcastというデータを使った解析がよく行われている。Statcastは、MLBの試合中に選手やボールの動きを追跡するためのシステムで、打球速度や打球角度、投球速度などの様々なデータを収集しているらしい。
そしてStatcastのデータはBaseball Savantというサイトで全世界に公開されている。
データ分析を生業にするものとして、そして一野球ファンとして、ぜひともこのデータを触ってみたいと思い、Pythonの`pybaseball`というライブラリを使ってStatcastのデータを取得・可視化を行ってみた。
内容としてはほとんど`pybaseball`の使い方みたいな感じではある。

<!-- more -->

## ライブラリの準備

`pybaseball`は`pip install`でインストールできる。

```bash
pip install pybaseball
```

分析を始めるにあたって、必要なライブラリをインポートしておく。

```python
import pybaseball
from pybaseball import statcast
from pybaseball import playerid_lookup
import plotly.graph_objects as go
import polars as pl

pybaseball.cache.enable() # cacheを有効にしておこう。
```

実際にStatcastのデータを取得してみる。
ここでは2024年のレギュラーシーズンのデータを取得することにする。

```python
# 2024年レギュラーシーズンを取得
_df = statcast(start_dt="2024-03-01", end_dt="2024-09-30")
df = (
    pl.DataFrame(_df)
    .filter(pl.col("game_type") == "R") # filter for regular season games
    .with_columns(pl.col("game_date").cast(pl.Date)) # cast game_date to Date type
    .with_columns((pl.col("game_date").dt.strftime("%Y-%m-%d")+", "+pl.col("away_team")+" vs "+pl.col("home_team")+", "+pl.col("des")).alias("des_with_date")) # combine game_date and des into a new column
)
```

|    | pitch_type   | game_date           |   release_speed |   release_pos_x |   release_pos_z | player_name   |   batter |   pitcher | events    | description     |   spin_dir |   spin_rate_deprecated |   break_angle_deprecated |   break_length_deprecated |   zone | des                                                                                   | game_type   | stand   | p_throws   | home_team   | away_team   | type   |   hit_location | bb_type     |   balls |   strikes |   game_year |   pfx_x |   pfx_z |   plate_x |   plate_z |   on_3b |   on_2b |   on_1b |   outs_when_up |   inning | inning_topbot   |   hc_x |   hc_y |   tfs_deprecated |   tfs_zulu_deprecated |   umpire |   sv_id |     vx0 |      vy0 |      vz0 |         ax |      ay |       az |   sz_top |   sz_bot |   hit_distance_sc |   launch_speed |   launch_angle |   effective_speed |   release_spin_rate |   release_extension |   game_pk |   fielder_2 |   fielder_3 |   fielder_4 |   fielder_5 |   fielder_6 |   fielder_7 |   fielder_8 |   fielder_9 |   release_pos_y |   estimated_ba_using_speedangle |   estimated_woba_using_speedangle |   woba_value |   woba_denom |   babip_value |   iso_value |   launch_speed_angle |   at_bat_number |   pitch_number | pitch_name      |   home_score |   away_score |   bat_score |   fld_score |   post_away_score |   post_home_score |   post_bat_score |   post_fld_score | if_fielding_alignment   | of_fielding_alignment   |   spin_axis |   delta_home_win_exp |   delta_run_exp |   bat_speed |   swing_length |   estimated_slg_using_speedangle |   delta_pitcher_run_exp |   hyper_speed |   home_score_diff |   bat_score_diff |   home_win_exp |   bat_win_exp |   age_pit_legacy |   age_bat_legacy |   age_pit |   age_bat |   n_thruorder_pitcher |   n_priorpa_thisgame_player_at_bat |   pitcher_days_since_prev_game |   batter_days_since_prev_game |   pitcher_days_until_next_game |   batter_days_until_next_game |   api_break_z_with_gravity |   api_break_x_arm |   api_break_x_batter_in |   arm_angle | des_with_date                                                                                                 |
|---:|:-------------|:--------------------|----------------:|----------------:|----------------:|:--------------|---------:|----------:|:----------|:----------------|-----------:|-----------------------:|-------------------------:|--------------------------:|-------:|:--------------------------------------------------------------------------------------|:------------|:--------|:-----------|:------------|:------------|:-------|---------------:|:------------|--------:|----------:|------------:|--------:|--------:|----------:|----------:|--------:|--------:|--------:|---------------:|---------:|:----------------|-------:|-------:|-----------------:|----------------------:|---------:|--------:|--------:|---------:|---------:|-----------:|--------:|---------:|---------:|---------:|------------------:|---------------:|---------------:|------------------:|--------------------:|--------------------:|----------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|------------:|----------------:|--------------------------------:|----------------------------------:|-------------:|-------------:|--------------:|------------:|---------------------:|----------------:|---------------:|:----------------|-------------:|-------------:|------------:|------------:|------------------:|------------------:|-----------------:|-----------------:|:------------------------|:------------------------|------------:|---------------------:|----------------:|------------:|---------------:|---------------------------------:|------------------------:|--------------:|------------------:|-----------------:|---------------:|--------------:|-----------------:|-----------------:|----------:|----------:|----------------------:|-----------------------------------:|-------------------------------:|------------------------------:|-------------------------------:|------------------------------:|---------------------------:|------------------:|------------------------:|------------:|:--------------------------------------------------------------------------------------------------------------|
|  0 | FF           | 2024-09-30 00:00:00 |            97.4 |           -2.1  |            4.88 | Díaz, Edwin   |   518595 |    621242 | field_out | hit_into_play   |        nan |                    nan |                      nan |                       nan |      3 | Travis d'Arnaud grounds out, shortstop Francisco Lindor to first baseman Pete Alonso. | R           | R       | R          | ATL         | NYM         | X      |              6 | ground_ball |       2 |         2 |        2024 |   -0.96 |    0.99 |      0.67 |      3    |     nan |  642201 |     nan |              2 |        9 | Bot             | 112.78 | 146.97 |              nan |                   nan |      nan |     nan | 9.89341 | -141.549 | -1.85711 | -15.1483   | 30.4239 | -18.5447 |     3.43 |     1.54 |                 5 |           87.6 |            -30 |              99.9 |                2196 |                 7.6 |    747139 |      682626 |      624413 |      657193 |      578428 |      596019 |      607043 |      621438 |      516782 |           52.91 |                           0.049 |                             0.052 |            0 |            1 |             0 |           0 |                    2 |              82 |              5 | 4-Seam Fastball |            7 |            8 |           7 |           8 |                 8 |                 7 |                7 |                8 | Standard                | Strategic               |         232 |               -0.142 |          -0.248 |        68.8 |            7.3 |                            0.059 |                   0.248 |            88 |                -1 |               -1 |          0.142 |         0.142 |               30 |               35 |        30 |        35 |                     1 |                                  4 |                              1 |                             1 |                              3 |                             1 |                       1.4  |              0.96 |                    0.96 |        17.6 | 2024-09-30, NYM vs ATL, Travis d'Arnaud grounds out, shortstop Francisco Lindor to first baseman Pete Alonso. |
|  1 | SL           | 2024-09-30 00:00:00 |            90.7 |           -2.14 |            5.06 | Díaz, Edwin   |   518595 |    621242 |           | ball            |        nan |                    nan |                      nan |                       nan |     14 | Travis d'Arnaud grounds out, shortstop Francisco Lindor to first baseman Pete Alonso. | R           | R       | R          | ATL         | NYM         | B      |            nan |             |       1 |         2 |        2024 |    0.2  |    0.61 |      0.75 |      1.2  |     nan |  642201 |     nan |              2 |        9 | Bot             | nan    | nan    |              nan |                   nan |      nan |     nan | 6.95286 | -131.971 | -5.15984 |   1.01261  | 25.1038 | -24.2244 |     3.47 |     1.52 |               nan |          nan   |            nan |              92.9 |                2209 |                 7.3 |    747139 |      682626 |      624413 |      657193 |      578428 |      596019 |      607043 |      621438 |      516782 |           53.15 |                         nan     |                           nan     |          nan |          nan |           nan |         nan |                  nan |              82 |              4 | Slider          |            7 |            8 |           7 |           8 |                 8 |                 7 |                7 |                8 | Standard                | Strategic               |         201 |                0     |           0.037 |       nan   |          nan   |                          nan     |                  -0.037 |           nan |                -1 |               -1 |          0.142 |         0.142 |               30 |               35 |        30 |        35 |                     1 |                                  4 |                              1 |                             1 |                              3 |                             1 |                       2.14 |             -0.2  |                   -0.2  |        23.1 | 2024-09-30, NYM vs ATL, Travis d'Arnaud grounds out, shortstop Francisco Lindor to first baseman Pete Alonso. |
|  2 | SL           | 2024-09-30 00:00:00 |            91.1 |           -2.07 |            5.14 | Díaz, Edwin   |   518595 |    621242 |           | swinging_strike |        nan |                    nan |                      nan |                       nan |      9 | Travis d'Arnaud grounds out, shortstop Francisco Lindor to first baseman Pete Alonso. | R           | R       | R          | ATL         | NYM         | S      |            nan |             |       1 |         1 |        2024 |    0.12 |    0.35 |      0.66 |      1.61 |     nan |  642201 |     nan |              2 |        9 | Bot             | nan    | nan    |              nan |                   nan |      nan |     nan | 6.7576  | -132.671 | -3.85718 |   0.114032 | 25.1083 | -27.4434 |     3.43 |     1.54 |               nan |          nan   |            nan |              93.5 |                2302 |                 7.4 |    747139 |      682626 |      624413 |      657193 |      578428 |      596019 |      607043 |      621438 |      516782 |           53.08 |                         nan     |                           nan     |          nan |          nan |           nan |         nan |                  nan |              82 |              3 | Slider          |            7 |            8 |           7 |           8 |                 8 |                 7 |                7 |                8 | Standard                | Strategic               |         210 |                0     |          -0.06  |        71.2 |            8.9 |                          nan     |                   0.06  |           nan |                -1 |               -1 |          0.142 |         0.142 |               30 |               35 |        30 |        35 |                     1 |                                  4 |                              1 |                             1 |                              3 |                             1 |                       2.37 |             -0.12 |                   -0.12 |        22.4 | 2024-09-30, NYM vs ATL, Travis d'Arnaud grounds out, shortstop Francisco Lindor to first baseman Pete Alonso. |
|  3 | SL           | 2024-09-30 00:00:00 |            91.3 |           -2.05 |            5.07 | Díaz, Edwin   |   518595 |    621242 |           | ball            |        nan |                    nan |                      nan |                       nan |     14 | Travis d'Arnaud grounds out, shortstop Francisco Lindor to first baseman Pete Alonso. | R           | R       | R          | ATL         | NYM         | B      |            nan |             |       0 |         1 |        2024 |    0.21 |    0.63 |      0.61 |      1.18 |     nan |  642201 |     nan |              2 |        9 | Bot             | nan    | nan    |              nan |                   nan |      nan |     nan | 6.38551 | -132.827 | -5.39777 |   1.18125  | 26.3297 | -23.8177 |     3.47 |     1.54 |               nan |          nan   |            nan |              93.5 |                2227 |                 7.4 |    747139 |      682626 |      624413 |      657193 |      578428 |      596019 |      607043 |      621438 |      516782 |           53.07 |                         nan     |                           nan     |          nan |          nan |           nan |         nan |                  nan |              82 |              2 | Slider          |            7 |            8 |           7 |           8 |                 8 |                 7 |                7 |                8 | Standard                | Strategic               |         212 |                0     |           0.012 |       nan   |          nan   |                          nan     |                  -0.012 |           nan |                -1 |               -1 |          0.142 |         0.142 |               30 |               35 |        30 |        35 |                     1 |                                  4 |                              1 |                             1 |                              3 |                             1 |                       2.09 |             -0.21 |                   -0.21 |        22.4 | 2024-09-30, NYM vs ATL, Travis d'Arnaud grounds out, shortstop Francisco Lindor to first baseman Pete Alonso. |
|  4 | SL           | 2024-09-30 00:00:00 |            89.1 |           -2.13 |            5.15 | Díaz, Edwin   |   518595 |    621242 |           | swinging_strike |        nan |                    nan |                      nan |                       nan |     14 | Travis d'Arnaud grounds out, shortstop Francisco Lindor to first baseman Pete Alonso. | R           | R       | R          | ATL         | NYM         | S      |            nan |             |       0 |         0 |        2024 |    0.17 |    0.66 |      1.36 |      1.78 |     nan |  642201 |     nan |              2 |        9 | Bot             | nan    | nan    |              nan |                   nan |      nan |     nan | 8.39601 | -129.602 | -3.73422 |   0.31946  | 25.3824 | -24.1466 |     3.43 |     1.54 |               nan |          nan   |            nan |              91.2 |                2160 |                 7.4 |    747139 |      682626 |      624413 |      657193 |      578428 |      596019 |      607043 |      621438 |      516782 |           53.08 |                         nan     |                           nan     |          nan |          nan |           nan |         nan |                  nan |              82 |              1 | Slider          |            7 |            8 |           7 |           8 |                 8 |                 7 |                7 |                8 | Standard                | Standard                |         216 |                0     |          -0.04  |        71.7 |            9   |                          nan     |                   0.04  |           nan |                -1 |               -1 |          0.142 |         0.142 |               30 |               35 |        30 |        35 |                     1 |                                  4 |                              1 |                             1 |                              3 |                             1 |                       2.2  |             -0.17 |                   -0.17 |        20.2 | 2024-09-30, NYM vs ATL, Travis d'Arnaud grounds out, shortstop Francisco Lindor to first baseman Pete Alonso. |

## プレイヤーの情報抽出

`playerid_lookup`を使って選手の情報を取得することができる。選手の名前やチーム名、ポジションなどの情報を取得することができる。

```python
dict_player_id = {
    "Shohei Ohtani": playerid_lookup("ohtani", "shohei")["key_mlbam"].values[0],
    "Aaron Judge": playerid_lookup("judge", "aaron")["key_mlbam"].values[0],
}
```

例えば2024年シーズンの大谷選手のホームラン数は次のように計算できる。
```python
# 2024年ホームラン数
len(df.filter(pl.col("batter") == dict_player_id["Shohei Ohtani"], pl.col("events") == "home_run")) # 54
```

メジャー全体ではジャッジ選手のホームランが一番多かった。ジャッジ選手と大谷選手のホームラン数のシーズン内での推移を見たければ次のようにすればよい。

```python
fig = go.Figure()
for player_name, player_id in dict_player_id.items():
    df_player = (
        df.filter(pl.col("batter") == player_id, pl.col("events") == "home_run")
        .group_by("game_date").agg(pl.len(), pl.col("des_with_date").first())
        .sort("game_date")
        .with_columns(pl.col("len").cum_sum())
        .select(pl.col("game_date").append(pl.lit("2024-03-20")), pl.col("len").append(pl.lit(0)), pl.col("des_with_date").append(None))
        .select(pl.col("game_date").append(pl.lit("2024-09-30")), pl.col("len").append(pl.col("len").max()), pl.col("des_with_date").append(None))
        .sort("game_date")
    )
    fig.add_trace(
        go.Scatter(
            x=df_player["game_date"],
            y=df_player["len"],
            mode='lines',
            line={"shape": 'hv'},
            name=player_name,
            text=df_player["des_with_date"],
        )
    )
fig.update_layout(
    title='Ohtani vs Judge HR (2024)',
    xaxis_title='Date',
    yaxis_title='Cumulative Home Runs',
    xaxis_range=["2024-03-20", "2024-09-30"],
    yaxis_range=[0, 60],
)
fig.show()
```
<iframe src="posts/posts/pybaseball/ohtani_vs_judge_hr.html"></iframe>

## Barrel Zoneについて

メジャーの中継を聞いているとBarrel Zoneという言葉をよく耳にする。Barrel Zoneとは、打球の角度と速度の組み合わせで、打球がホームランになる確率が高いゾーンのことを指す。
Barrel Zoneに関するデータもStatcast内に含まれているので、ホームランとの関係性についても調べてみることにした。

はじめに2024年シーズンのすべての打席について打球速度・打球角度とそれがBarrel Zoneに入っているかどうかを可視化しよう。

```python
# https://baseballsavant.mlb.com/csv-docs
dict_launch_speed_angle = {1: "Weak", 2: "Topped", 3: "Under", 4: "Flare/Burner", 5: "Solid Contact", 6: "Barrel"}

# 打球速度と打球角度の分布、そしてその区分について
fig = go.Figure()
for launch_speed_angle_id, launch_speed_angle_name in dict_launch_speed_angle.items():
    df_launch_speed_angle = (
        df.filter(pl.col("launch_speed_angle") == launch_speed_angle_id)
        .group_by("launch_speed", "launch_angle").agg(pl.len(), pl.col("des_with_date").first())
    )
    fig.add_trace(
        go.Scattergl(
            x=df_launch_speed_angle["launch_speed"],
            y=df_launch_speed_angle["launch_angle"],
            mode='markers',
            name=launch_speed_angle_name,
            marker=dict(size=3),
            text=df_launch_speed_angle["des_with_date"],
        )
    )
fig.update_layout(
    title='Launch Speed vs Launch Angle (2024)',
    xaxis_title='Launch Speed (mph)',
    yaxis_title='Launch Angle (degree)',
)
fig.show()
```

```plotly
{"file_path": "posts/posts/pybaseball/launch_speed_angle.json"}
```

次に、Barrel Zoneに入っている打球のうち、ホームランになったものとならなかったものを分けて可視化してみる。

```python
# Barrel zone vs Home Run比較
# Barrel and Home Run, Barrel and Non-Home Run, Non-Barrel and Home Run, Non-Barrel and Non-Home Runのプロットをする
fig = go.Figure()
# Barrel and Home Run
df_barrel_home_run = (
    df.filter(pl.col("events") == "home_run", pl.col("launch_speed_angle") == 6)
    .group_by("launch_speed", "launch_angle").agg(pl.len(), pl.col("des_with_date").first())
)
fig.add_trace(
    go.Scattergl(
        x=df_barrel_home_run["launch_speed"],
        y=df_barrel_home_run["launch_angle"],
        mode='markers',
        name='Barrel and Home Run',
        marker=dict(size=3),
        text=df_barrel_home_run["des_with_date"],
    )
)
# Barrel and Non-Home Run
df_barrel_non_home_run = (
    df.filter(pl.col("events") != "home_run", pl.col("launch_speed_angle") == 6)
    .group_by("launch_speed", "launch_angle").agg(pl.len(), pl.col("des_with_date").first())
)
fig.add_trace(
    go.Scattergl(
        x=df_barrel_non_home_run["launch_speed"],
        y=df_barrel_non_home_run["launch_angle"],
        mode='markers',
        name='Barrel and Non-Home Run',
        marker=dict(size=3),
        text=df_barrel_non_home_run["des_with_date"],
    )
)
# Non-Barrel and Home Run
df_non_barrel_home_run = (
    df.filter(pl.col("events") == "home_run", pl.col("launch_speed_angle") != 6)
    .group_by("launch_speed", "launch_angle").agg(pl.len(), pl.col("des_with_date").first())
)
fig.add_trace(
    go.Scattergl(
        x=df_non_barrel_home_run["launch_speed"],
        y=df_non_barrel_home_run["launch_angle"],
        mode='markers',
        name='Non-Barrel and Home Run',
        marker=dict(size=3),
        text=df_non_barrel_home_run["des_with_date"],
    )
)
# Non-Barrel and Non-Home Run
df_non_barrel_non_home_run = (
    df.filter(pl.col("events") != "home_run", pl.col("launch_speed_angle") != 6)
    .group_by("launch_speed", "launch_angle").agg(pl.len(), pl.col("des_with_date").first())
)
fig.add_trace(
    go.Scattergl(
        x=df_non_barrel_non_home_run["launch_speed"],
        y=df_non_barrel_non_home_run["launch_angle"],
        mode='markers',
        name='Non-Barrel and Non-Home Run',
        marker=dict(size=1, color="gray", opacity=0.5),
        text=df_non_barrel_non_home_run["des_with_date"],
    )
)
fig.update_layout(
    title='Barrel Zone and Home Run Relation (2024)',
    xaxis_title='Launch Speed (mph)',
    yaxis_title='Launch Angle (degree)',
)

fig.show()
```

```plotly
{"file_path": "posts/posts/pybaseball/barrel_zone_home_run.json"}
```

これを見ていると、Barrel Zoneに入っている打球はホームランになる確率が高いことがわかる。また、Barrel Zoneの中でも特に打球速度が大きいものを打った選手を見ていると、ジャッジ選手と大谷選手が目立つ。さすがと言わざるを得ない。
また、Barrel Zoneに入っていて、しかもかなり打球速度・角度ともに良いところにあるのにも関わらずホームランになっていないものも見受けられる。その一つが、

- 2024年8月8日、PHIvsLADでの大谷選手の打球

である。この打球を実際に見てみるとこんな感じ。少し上がり過ぎたか風に押し戻されたのかも。滞空時間はかなり長いことが分かる。

<iframe src="https://streamable.com/m/aaron-nola-in-play-run-s-to-shohei-ohtani?partnerId=web_video-playback-page_video-share" width="560" height="315"></iframe>
