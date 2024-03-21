---
title: "『推薦システム実践入門』5章1節2節"
date: 2024-03-21
slug: recommendation-systems_setup
draft: false
authors:
  - yonesuke
categories:
  - 推薦システム実践入門
  - Recommendation Systems
  - Python
---

<img src="https://www.oreilly.co.jp/books/images/picture_large978-4-87311-966-3.jpeg" width=300>

[『推薦システム実践入門』](https://www.oreilly.co.jp/books/9784873119663/)を読み始めたので、実装をまとめてみます。
Pythonによる実装自体は[GitHub](https://github.com/oreilly-japan/RecommenderSystems)にアップされています。
データ処理には定番の[pandas](https://pandas.pydata.org/)が使われているのですが、
最近自分が[polars](https://pola.rs/)を使い始めているのと、devcontainer上でpandasを回しているとすぐにメモリがあふれてしまうという問題にぶち当たったので、
ここではpolarsを使った実装をまとめていこうと思います。
このブログで紹介した実装は自分の[GitHub](https://github.com/yonesuke/recommend-systems)にも適宜アップロードしていく予定です。

<!-- more -->

この本の5章では推薦アルゴリズムの詳細と題して、[Movielens](https://grouplens.org/datasets/movielens/)のデータセットを題材に複数の推薦アルゴリズムを実装・評価していきます。
この5章で紹介される推薦アルゴリズムとして、以下のものが挙げられています。

| アルゴリズム名 | 概要 | ブログリンク |
| --- | --- | --- |
| ランダム推薦 | ランダムにアイテムを推薦する。ベースラインとして利用されることがある |
| 統計情報や特定のルールに基づく推薦（人気度推薦など） | ベースラインとしてよく利用される |

## データセットのダウンロード

以下のコマンドでダウンロードします。

```bash
wget -nc --no-check-certificate https://files.grouplens.org/datasets/movielens/ml-10m.zip -P data
unzip data/ml-10m.zip -d data/
```

### フォルダ構成

以下の形でファイルを配置しています。
```
yonesuke/recommend-systems
├── data/ml-10M100K
|   ├── data_loader.py
|   ├── metric_calculator.py
|   └── models.py
├── algorithms
│   ├── base_recommender.py
│   ├── popularity_recommender.py
│   ├── random_recommender.py
│   └── ...
└── utils
    ├── data_loader.py
    ├── metric_calculator.py
    └── models.py
```

- `utils`以下にはデータのロードや評価指標の計算などの共通の処理をまとめています。
- `algorithms`以下には各推薦アルゴリズムの実装をまとめています。
- `data`以下にはデータセットを配置しています。

## データセット・結果出力関連データクラス
データセットをまとめたデータクラスを`utils/models.py`に実装します。

```python
import dataclasses
import polars as pl

@dataclasses.dataclass(frozen=True)
class Dataset:
    """Dataset for recommendation system
    
    Args:
    train (pl.DataFrame): Training data
    test (pl.DataFrame): Test data
    test_user2items (dict[int, list[int]]): Test data for each user
    item_content (pl.DataFrame): Item content data
    """
    train: pl.DataFrame
    test: pl.DataFrame
    test_user2items: dict[int, list[int]]
    item_content: pl.DataFrame
    
@dataclasses.dataclass(frozen=True)
class RecommendResult:
    """Recommend result
    
    Args:
    rating (pl.DataFrame): Rating data (expected header: user_id, item_id, pred_rating)
    user2items (dict[int, list[int]]): Recommended items for each user
    """
    rating: pl.DataFrame
    user2items: dict[int, list[int]]
    
@dataclasses.dataclass(frozen=True)
class Metrics:
    """Metrics for recommendation system
    
    Args:
    rsme (float): RSME
    precision_at_k (float): Precision@K
    recall_at_k (float): Recall@K
    """
    rsme: float
    precision_at_k: float
    recall_at_k: float
    
    def __repr__(self) -> str:
        return f'RSME: {self.rsme:.4f}, Precision@K: {self.precision_at_k:.4f}, Recall@K: {self.recall_at_k:.4f}'
```

## データの読み込み
polarsを使ったデータの読み込み(`utils/data_loader.py`)を実装します。

```python
import polars as pl
import os
from utils.models import Dataset

class DataLoader:
    """Data loader for recommendation system
    
    Args:
    n_user (int): Number of users
    n_test_items (int): Number of test items
    data_path (str): Path to the data
    """
    def __init__(
        self, n_user: int = 1000, n_test_items: int = 5, data_path: str = '../data/ml-10M100K'
    ) -> None:
        self.n_user = n_user
        self.n_test_items = n_test_items
        self.data_path = data_path
        
    def load(self) -> Dataset:
        """Load the dataset
        
        Returns:
            Dataset: Dataset for recommendation system
        """
        movielens, movie_content = self._load()
        movielens_train, movielens_test = self._split_data(movielens)
        # ranking用の評価データは、各ユーザーの評価値が4以上の映画だけを正解とする
        # key: user_id, value: list of item_id
        user2items = (
            movielens_test
            .filter(pl.col('rating') >= 4.0)
            .group_by('user_id').agg(pl.col('movie_id'))
        )
        movielens_test_user2items = {col[0]: col[1] for col in user2items.iter_rows()}
        return Dataset(movielens_train, movielens_test, movielens_test_user2items, movie_content)
    
    def _split_data(self, df_movielens: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
        df_movielens = (
            df_movielens.with_columns(
                pl.col('timestamp').rank(method='ordinal', descending=True)
                .over('user_id')
                .alias('rating_order')
            )
        )
        df_train = df_movielens.filter(pl.col('rating_order') > self.n_test_items)
        df_test = df_movielens.filter(pl.col('rating_order') <= self.n_test_items)
        return df_train, df_test
    
    def _load(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        # 映画の情報の読み込み（10197作品）
        df_movies = (
            pl.read_csv(
                os.path.join(self.data_path, 'movies.dat'),
                has_header=False,
                truncate_ragged_lines=True
            )
            .with_columns(pl.col('column_1').str.split('::'))
            .with_columns(
                pl.col('column_1').list[0].alias('movie_id'),
                pl.col('column_1').list[1].alias('title'),
                (
                    pl.when(pl.col('column_1').list.len() > 2)
                    .then(pl.col('column_1').list[2].str.split('|'))
                    .otherwise(pl.lit([]))
                    .alias('genres')
                )
            )
            .drop('column_1')
        )
        
        # ユーザーが付与した映画のタグ情報の読み込み
        df_tags = (
            pl.read_csv(
                os.path.join(self.data_path, 'tags.dat'),
                has_header=False,
            )
            .with_columns(pl.col('column_1').str.split('::'))
            .with_columns(
                pl.col('column_1').list[0].alias('user_id'),
                pl.col('column_1').list[1].alias('movie_id'),
                pl.col('column_1').list[2].str.to_lowercase().alias('tag'),
                pl.from_epoch(pl.col('column_1').list[3].cast(pl.Int32)).alias('timestamp')
            )
            .drop('column_1')
        )
        
        # tag情報を結合
        df_movies = df_movies.join(
            df_tags.group_by('movie_id').agg(pl.col('tag')),
            on='movie_id', how='left'
        )
        
        # 評価データの読み込み
        df_ratings = (
            pl.read_csv(
                os.path.join(self.data_path, 'ratings.dat'),
                has_header=False,
            )
            .with_columns(pl.col('column_1').str.split('::'))
            .with_columns(
                pl.col('column_1').list[0].alias('user_id'),
                pl.col('column_1').list[1].alias('movie_id'),
                pl.col('column_1').list[2].cast(pl.Float64).alias('rating'),
                pl.from_epoch(pl.col('column_1').list[3].cast(pl.Int32)).alias('timestamp')
            )
            .drop('column_1')
        )
        
        # user数をn_userに制限
        valid_user_ids = (
            df_ratings.get_column('user_id')
            .unique(maintain_order=True)
            .to_list()
            [:self.n_user]
        )
        df_ratings = df_ratings.filter(pl.col('user_id').is_in(valid_user_ids))
        
        # 上記のデータを結合
        df_movielens = df_ratings.join(df_movies, on='movie_id', how='left')
        
        return df_movielens, df_movies
```

## 評価指標の計算

評価指標の計算(`utils/metric_calculator.py`)を実装します。

```python
import numpy as np
from sklearn.metrics import mean_squared_error
from utils.models import Metrics

class MetricCalculator:
    def calc(
        self,
        true_rating: list[float],
        pred_rating: list[float],
        true_user2items: dict[int, list[int]],
        pred_user2items: dict[int, list[int]],
        k: int
    ) -> Metrics:
        rsme = self._calc_rmse(true_rating, pred_rating)
        precision_at_k = self._calc_precision_at_k(true_user2items, pred_user2items, k)
        recall_at_k = self._calc_recall_at_k(true_user2items, pred_user2items, k)
        return Metrics(rsme, precision_at_k, recall_at_k)
    
    def _precision_at_k(self, true_item: list[int], pred_item: list[int], k: int) -> float:
        if k == 0:
            return 0.0
        
        return len(set(true_item) & set(pred_item[:k])) / k
    
    def _recall_at_k(self, true_item: list[int], pred_item: list[int], k: int) -> float:
        if k == 0:
            return 0.0
        
        return len(set(true_item) & set(pred_item[:k])) / len(true_item)
    
    def _calc_rmse(self, true_rating: list[float], pred_rating: list[float]) -> float:
        return np.sqrt(mean_squared_error(true_rating, pred_rating))
    
    def _calc_precision_at_k(
        self, true_user2items: dict[int, list[int]], pred_user2items: dict[int, list[int]], k: int
    ) -> float:
        scores = []
        for user_id in true_user2items.keys():
            true_item = true_user2items[user_id]
            pred_item = pred_user2items[user_id]
            scores.append(self._precision_at_k(true_item, pred_item, k))
        return np.mean(scores)
    
    def _calc_recall_at_k(
        self, true_user2items: dict[int, list[int]], pred_user2items: dict[int, list[int]], k: int
    ) -> float:
        scores = []
        for user_id in true_user2items.keys():
            true_item = true_user2items[user_id]
            pred_item = pred_user2items[user_id]
            scores.append(self._recall_at_k(true_item, pred_item, k))
        return np.mean(scores)
```

## まとめ
ここでは『推薦システム実践入門』5章の1節2節で紹介された実装をpolarsを使って実装してみた。
次回以降はランダム推薦と人気度推薦の実装を行っていく予定です。