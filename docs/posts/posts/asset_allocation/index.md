---
title: "投資家のポートフォリオ最適化問題"
date: 2024-01-07
slug: asset_allocation
draft: false
slug: asset_allocation
categories:
    - Finance
    - Python
authors:
    - yonesuke
---

投資家のポートフォリオ最適化問題の概要と理論について説明し、数値計算による確認を行う。

!!! warning
    本記事に含まれる情報に基づいて被ったいかなる損害についても一切責任を負いません。

<!-- more -->

## 投資家のポートフォリオ最適化問題とは
!!! note
    この説明はこのブログを読んでもらうより、[Wikipedia](https://ja.wikipedia.org/wiki/%E7%8F%BE%E4%BB%A3%E3%83%9D%E3%83%BC%E3%83%88%E3%83%95%E3%82%A9%E3%83%AA%E3%82%AA%E7%90%86%E8%AB%96)や他の記事を見てもらったほうがよっぽど詳しく書いてある。

投資家が持つ資産を最大限に活用するために、その資産をどのように配分するかを決定する問題を**資産配分問題**という。
一般に投資家は、同じ期待収益をもつのであれば、より小さいリスクで投資を行いたいと考える。
このような投資家の要望を満たすために、資産配分問題においては、投資家が保有する資産の期待収益とリスクを表すパラメータを用いて、投資家の要望を満たすような資産の配分比率を決定する。
このように、与えられた期待リターンの中で最もリスクが小さいポートフォリオを求める問題を**ポートフォリオ最適化問題**という。

資産$i$の期待収益を$r_i$、資産$i$と資産$j$の共分散を行列$\Sigma$の$(i,j)$成分とする（すなわちは$\Sigma$は資産間共分散行列）。
このとき、各資産の配分比率を$w_i$とすると、ポートフォリオの期待収益は$w^{\top}r$となる。また、ポートフォリオの分散をリスクと考えると、それは$w^{\top}\Sigma w$となる。ここで、$w = (w_1, \dots, w_n)^{\top}$は投資家が保有する資産の配分比率を表すベクトル、$r = (r_1, \dots, r_n)^{\top}$は資産の期待収益を表すベクトルである。

以上をもとに最適化問題としての定式化を行うと、以下のようになる。
$$
\begin{aligned}
& \min w^{\top} \Sigma w \\\\
& \text{s.t. } w^{\top}r = \mu,\ \sum_{i=1}^n w_i = 1
\end{aligned}
$$
ただし、$\mu$は投資家が期待するポートフォリオの期待収益である。

上の定式化において$w$は$n$次元のベクトルであるが、正負の値をとることができるようになっている。これは空売りを許可していることを意味する。空売りを許可しない場合は、$w_i \geq 0$という制約を追加する。

## `cvxopt`を用いた数値計算
`cvxopt`はPythonで凸最適化問題を解くためのライブラリである。

はじめに資産の期待収益とリスクを定めるクラスを定義する。
```python
import dataclasses

@dataclasses.dataclass
class Asset:
    ticker: str
    exp_rtn: float
    risk: float
```

これをもとにポートフォリオ最適化問題を解くクラスを定義する。
```python
import cvxopt
import numpy as np
from typing import List, Tuple

@dataclasses.dataclass
class Portfolio:
    assets: List[Asset]
    cov_mat: np.ndarray = dataclasses.field(repr=False)
    n_asset: int = dataclasses.field(init=False)
    tickers: List[str] = dataclasses.field(init=False, repr=False)
    exp_rtns: List[float] = dataclasses.field(init=False, repr=False)
    risks: List[float] = dataclasses.field(init=False, repr=False)
    
    def __post_init__(self):
        self.n_asset = len(self.assets)
        assert self.n_asset == self.cov_mat.shape[0] == self.cov_mat.shape[1], 'cov matrix size mismatch'
        self.tickers = [asset.ticker for asset in self.assets]
        self.exp_rtns = np.array([asset.exp_rtn for asset in self.assets])
        self.risks = np.array([asset.risk for asset in self.assets])

    def generate_weights(self, n_sample: int) -> np.ndarray:
        return np.random.dirichlet(np.ones(self.n_asset), size=n_sample)
    
    def calc_exp_rtn_risk(self, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        assert np.isclose(weights.sum(axis=-1), 1.0).all(), 'weights must sum to 1'
        assert weights.shape[-1] == self.n_asset, 'weights size mismatch'
        exp_rtns = weights @ self.exp_rtns
        if weights.ndim == 1:
            risks = np.sqrt(np.dot(weights, (self.cov_mat @ weights)))
        else:
            risks = np.sqrt([np.dot(weight, (self.cov_mat @ weight)) for weight in weights])
        return exp_rtns, risks
    
    def calc_min_risk(self, mu: float, allow_short_sell: bool = True) -> Tuple[float, np.ndarray]:
        P = cvxopt.matrix(self.cov_mat)
        q = cvxopt.matrix(np.zeros(self.n_asset, dtype=float))
        if allow_short_sell:
            G = None
            h = None
        else:
            G = cvxopt.matrix(-np.identity(self.n_asset))
            h = cvxopt.matrix(np.zeros(self.n_asset, dtype=float))
        A = cvxopt.matrix(np.vstack((self.exp_rtns, np.ones(self.n_asset, dtype=float))))
        b = cvxopt.matrix([mu, 1.0])
        sol = cvxopt.solvers.qp(P, q, G, h, A, b, options=dict(show_progress=False))
        risk = np.sqrt(sol['primal objective'] * 2)
        weight = np.array(sol['x'])
        return risk, weight
```

`calc_min_risk`メソッドは、ポートフォリオの期待収益が$\mu$となるような最小リスクのポートフォリオを求めるメソッドである。`allow_short_sell`が`True`のときは空売りを許可する。`False`のときは空売りを許可しない。

以上をもとに、ポートフォリオ最適化問題を解く。次の5銘柄の株式を考える。ただし、簡単のために、各銘柄の共分散は0とする。
```python
asset_1 = Asset(ticker='銘柄1', exp_rtn=0.06, risk=0.12)
asset_2 = Asset(ticker='銘柄2', exp_rtn=0.12, risk=0.18)
asset_3 = Asset(ticker='銘柄3', exp_rtn=0.03, risk=0.12)
asset_4 = Asset(ticker='銘柄4', exp_rtn=0.08, risk=0.15)
asset_5 = Asset(ticker='銘柄5', exp_rtn=0.06, risk=0.17)

portfolio = Portfolio(
    assets=[asset_1, asset_2, asset_3, asset_4, asset_5],
    cov_mat=np.diag([asset_1.risk**2, asset_2.risk**2, asset_3.risk**2, asset_4.risk**2, asset_5.risk**2])
)
```

期待リターンを動かしたときに、リスクがどのように変化するかを計算する。
ここでは、空売りを許可する場合と許可しない場合、また、ランダムに配分比率を生成した場合の結果を比較する。
```python
mus = np.linspace(portfolio.exp_rtns.min(), portfolio.exp_rtns.max(), 100)
# allow short sell
risks_with_short = [portfolio.calc_min_risk(mu, True)[0] for mu in mus]
# not allow short sell
risks_without_short = [portfolio.calc_min_risk(mu, False)[0] for mu in mus]
# random weights
weights = portfolio.generate_weights(50000)
exp_rtns, risks = portfolio.calc_exp_rtn_risk(weights)
```

以上をプロットしよう。
```python
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(x=risks_with_short, y=mus, name='allow short sell'))
fig.add_trace(go.Scatter(x=risks_without_short, y=mus, name='not allow short sell'))
fig.add_trace(
    go.Scattergl(
        x=risks, y=exp_rtns, mode='markers', opacity=0.5, name='random weights',
        marker=dict(size=1)
    )
)
for asset in portfolio.assets:
    fig.add_trace(
        go.Scatter(
            x=[asset.risk], y=[asset.exp_rtn], mode='markers', name=asset.ticker,
            marker=dict(size=12, line=dict(width=2)), marker_symbol='x')
        )
fig.update_layout(
    xaxis_title='Risk', yaxis_title='Expected Return',
    xaxis_tickformat = '.2%', yaxis_tickformat = '.2%',
    xaxis_range=[0, 0.2], yaxis_range=[0, 0.14],
    autosize=False, width=800, height=600
)
fig.show()
```

```plotly
{"file_path": "posts/posts/asset_allocation/plotly.json"}
```

## まとめ
plotlyきれい。今後も使っていきたい。
あと、今回は最適化問題を解くのにライブラリに丸投げしたが、自力でそれのアルゴリズムも実装してみたい。