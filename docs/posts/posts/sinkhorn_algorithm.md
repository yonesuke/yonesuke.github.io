---
title: "SinkhornアルゴリズムとJAXによる実装: 最適輸送の高速化"
date: 2026-01-11
slug: sinkhorn_algorithm
draft: false
math: true
authors:
    - yonesuke
categories:
    - Python
    - AI
    - JAX
---

最適輸送問題に以前から興味があり、最近のDeepSeekのMHC (Manifold Constrained Hyper Connections) 論文でも言及されていたため、その中心となるSinkhornアルゴリズムについて詳細を調べ、JAXで実装してみた際のメモ。

!!! warning
    この記事は Google Antigravity を使用して作成されました。
    あくまで私自身の勉強した結果の備忘録としてのメモと思っていただければと思います。
    （正確性はかならずしも担保されません。）
    作成過程で知らないことが多くあり、非常に勉強になりました。

<!-- more -->

## 1. モチベーション: なぜ Sinkhorn なのか？

最適輸送（Optimal Transport, OT）は、ある確率分布を別の確率分布へ最小のコストで輸送する方法を見つける問題である。近年、機械学習の分野で分布間の距離（Wasserstein距離など）を測るツールとして急速に注目を集めている。

しかし、通常の最適輸送問題は線形計画問題（Linear Programming, LP）として定式化されるため、データ点数 $N$ に対して計算コストが大きく（一般に $O(N^3 \log N)$ 程度）、大規模なデータセットへの適用が困難であった。
そこで登場したのが **Sinkhorn アルゴリズム** である。エントロピー正則化を導入することで、問題を凸最適化問題の容易なクラスに緩和し、単純な行列スケーリングの繰り返しによって $O(N^2)$ で高速に解くことを可能にした。

本記事では、この Sinkhorn アルゴリズムのモチベーション、背後にある理論、そして JAX を用いた実装について詳細に解説する。

### 1.1 問題設定：割り当て問題とコスト行列
ここでは、$N$ 個の要素（ソース）を $N$ 個の要素（ターゲット）に1対1で対応させる「割り当て問題（Assignment Problem）」や、それに関連するマッチング問題を考える。
入力として与えられるのは、$i$ 番目のソースと $j$ 番目のターゲットをマッチさせたときのコストを表す行列 $C \in \mathbb{R}^{N \times N}$ である。

目標は、総コストを最小化するような置換行列（Permutation Matrix）、あるいはより一般に、行和と列和がすべて $1$ になるような二重確率行列（Doubly Stochastic Matrix） $P$ を求めることである。

$$
\min_{P \in \mathcal{DS}_N} \langle P, C \rangle = \sum_{i,j} P_{ij} C_{ij}
$$

ここで $\mathcal{DS}_N$ は以下の条件を満たす行列の集合である：
$$
\sum_{j=1}^N P_{ij} = 1, \quad \sum_{i=1}^N P_{ij} = 1, \quad P_{ij} \ge 0
$$

DeepSeek が最近発表した **Manifold Constrained Hyper Connections (MHC)** などの文脈でも、このようなコスト行列 $C$ を入力として、対応する確率的なマッチング $P$ を得たい場面が頻出する。
MHCでは、エキスパート間のルーティングや情報の集約において、制約付きの割り当て問題を効率的に解く必要があり、まさに Sinkhorn アルゴリズムの高速性が活きる場面の一つである。

しかし、これを厳密な線形計画法で解くのは計算コストが高く ($O(N^3)$)、微分可能でもない。

### 1.2 エントロピー正則化の導入
Marco Cuturi (2013) は、元のコスト関数に「エントロピー正則化項」を加えることを提案した。これにより、問題は以下のようになる。

$$
\min_{P \in \mathcal{DS}_N} \langle P, C \rangle - \epsilon H(P)
$$

ここで $H(P) = -\sum_{i,j} P_{ij} (\log P_{ij} - 1)$ はシャノンエントロピー、$\epsilon > 0$ は正則化の強さを決めるハイパーパラメータである。

この正則化には2つの大きな利点がある：
1.  **計算効率**: 行列スケーリングアルゴリズム（Sinkhorn-Knopp アルゴリズム）を用いて、行列とベクトルの積だけで解けるようになり、並列化（GPU化）が極めて容易になる。
2.  **微分可能性**: 解が一意に定まり、入力に対する勾配を解析的に計算できるため、ニューラルネットワークの損失関数として組み込める。

## 2. 理論的導出

なぜエントロピーを入れると簡単になるのか。ラグランジュの未定乗数法を用いて導出する。

### 2.1 ラグランジュ関数
制約条件は、行和と列和がともに $1$ になることである。

1. $\sum_{j} P_{ij} = 1$
2. $\sum_{i} P_{ij} = 1$

これらに対するラグランジュ乗数を導入し、エントロピー正則化項を加えた目的関数を最小化する。導出の過程は一般の場合と同様だが、最終的な更新式において、ターゲットとする周辺分布が「全て1のベクトル」になる点が特徴である。

$$
\mathcal{L}(P, \alpha, \beta) = \sum_{i,j} P_{ij} C_{ij} + \epsilon \sum_{i,j} P_{ij} (\log P_{ij} - 1) - \sum_{i} \alpha_i (\sum_{j} P_{ij} - 1) - \sum_{j} \beta_j (\sum_{i} P_{ij} - 1)
$$

### 2.2 最適解の形式
$P_{ij}$ で偏微分して 0 と置く。

$$
\frac{\partial \mathcal{L}}{\partial P_{ij}} = C_{ij} + \epsilon \log P_{ij} - \alpha_i - \beta_j = 0
$$

これを $P_{ij}$ について解くと：

$$
\log P_{ij} = \frac{\alpha_i + \beta_j - C_{ij}}{\epsilon}
$$

$$
P_{ij} = \exp\left( \frac{\alpha_i}{\epsilon} \right) \exp\left( -\frac{C_{ij}}{\epsilon} \right) \exp\left( \frac{\beta_j}{\epsilon} \right)
$$

ここで、$u_i = \exp(\alpha_i / \epsilon)$, $v_j = \exp(\beta_j / \epsilon)$, $K_{ij} = \exp(-C_{ij} / \epsilon)$ と置くと、最適解 $P^*$ は以下の形式を持つことがわかる。

$$
P^*_{ij} = u_i K_{ij} v_j
$$

あるいは行列形式で書くと：
$$
P^* = \text{diag}(u) K \text{diag}(v)
$$

これは、最適輸送計画 $P^*$ が、ギブス核 $K$ を対角行列で左右からスケーリングしたものになることを示している。

### 2.3 Sinkhorn アルゴリズム
未知数はベクトル $u$ と $v$ である。これらは制約条件を満たす必要がある。

1. $P \mathbf{1} = \mathbf{1} \implies \text{diag}(u) K \text{diag}(v) \mathbf{1} = \mathbf{1} \implies u \odot (K v) = \mathbf{1} \implies u = \frac{1}{K v}$
2. $P^T \mathbf{1} = \mathbf{1} \implies \text{diag}(v) K^T \text{diag}(u) \mathbf{1} = \mathbf{1} \implies v \odot (K^T u) = \mathbf{1} \implies v = \frac{1}{K^T u}$

ここで $\odot$ は要素ごとの積、除算も要素ごとである。
この2つの式を交互に更新することで、解に収束させることができる。これが **Sinkhorn-Knopp アルゴリズム** である。

**アルゴリズム**:
1. $K = \exp(-C / \epsilon)$ を計算。
2. $u, v$ を適当に初期化（例: 全て1）。
3. 収束するまで以下を繰り返す：
    - $u \leftarrow 1 / (K v)$
    - $v \leftarrow 1 / (K^T u)$
4. 最終的な輸送計画は $P = \text{diag}(u) K \text{diag}(v)$。

このアルゴリズムは行列ベクトル積のみで構成されているため、GPU上で非常に高速に動作する。

### 2.4 Log-Domain での安定化
$\epsilon$ が小さい場合、$K = \exp(-C/\epsilon)$ の要素は非常に小さくなり、ゼロアンダーフローを起こす可能性がある。これを防ぐために、対数領域（Log-domain）で計算を行うのが一般的である。

$f_i = \epsilon \log u_i, g_j = \epsilon \log v_j$ と変数を置き換え、Log-Sum-Exp (LSE) 演算を用いて更新式を書き換える。

$$
f_i \leftarrow \min_{\epsilon} (C_{i \cdot} - g) \text{ (のような形式)}
$$

## 3. JAX による実装

それでは、JAXを使ってこれを実装する。JAXを使うことで、自動微分（Auto-diff）やJITコンパイルによる高速化の恩恵を簡単に受けることができる。

```python
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "jax",
#     "numpy",
# ]
# ///

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import numpy as np

def sinkhorn_knopp(C, epsilon=1e-2, max_iter=1000, a=None, b=None):
    """
    Standard Sinkhorn-Knopp algorithm for N x N cost matrix.
    
    Args:
        C: (N, N) Cost matrix.
        epsilon: Regularization parameter.
        max_iter: Max iterations.
        a: (Optional) Source marginals (N,). Defaults to ones.
        b: (Optional) Target marginals (N,). Defaults to ones.
    """
    N = C.shape[0]
    if a is None:
        a = jnp.ones(N)
    if b is None:
        b = jnp.ones(N)
        
    K = jnp.exp(-C / epsilon)
    
    def body_fn(val):
        u, v, i = val
        v = b / (jnp.dot(K.T, u) + 1e-10)
        u = a / (jnp.dot(K, v) + 1e-10)
        return u, v, i + 1

    def cond_fn(val):
        _, _, i = val
        return i < max_iter

    u = jnp.ones_like(a)
    v = jnp.ones_like(b)
    
    u, v, _ = jax.lax.while_loop(cond_fn, body_fn, (u, v, 0))
    
    P = jnp.diag(u) @ K @ jnp.diag(v)
    return P

def sinkhorn_log(C, epsilon=1e-2, max_iter=1000, a=None, b=None):
    """
    Log-domain Sinkhorn algorithm for N x N cost matrix.
    Stable for small epsilon.
    """
    N = C.shape[0]
    if a is None:
        a = jnp.ones(N)
    if b is None:
        b = jnp.ones(N)

    def body_fn(val):
        f, g, i = val
        # g update in log domain
        M_g = (f[:, None] - C) / epsilon
        lse_g = logsumexp(M_g, axis=0)
        g = epsilon * jnp.log(b) - epsilon * lse_g
        
        # f update in log domain
        M_f = (g[None, :] - C) / epsilon
        lse_f = logsumexp(M_f, axis=1)
        f = epsilon * jnp.log(a) - epsilon * lse_f
        
        return f, g, i + 1

    def cond_fn(val):
        _, _, i = val
        return i < max_iter

    f = jnp.zeros_like(a)
    g = jnp.zeros_like(b)
    
    f, g, _ = jax.lax.while_loop(cond_fn, body_fn, (f, g, 0))
    
    P_log = (f[:, None] + g[None, :] - C) / epsilon
    P = jnp.exp(P_log)
    return P
```

### 3.1 実行方法について
このコードは `uv` を用いて依存関係を管理し、即座に実行可能な形式（PEP 723）になっている。
以下のコマンドで簡単に試すことができる：

```bash
uv run sinkhorn_jax.py
```

### 3.2 結果の検証
ランダムな $N \times N$ コスト行列対して実行し、得られた輸送計画 $P$ の行和・列和がそれぞれ $1$ に一致しているかを確認する。

```python
# 検証用コードの一部
C = jax.random.uniform(key, (N, N))

P_std = sinkhorn_knopp(C, epsilon=0.1)
# 誤差確認: 行和・列和と 1.0 との差分ノルム
# Standard Marginals Error: ...

P_log = sinkhorn_log(C, epsilon=0.01)
# 誤差確認
# Log-Domain Marginals Error: ...
```

Log-domain の実装を用いることで、より小さな $\epsilon$ （より鮮明な輸送計画）に対しても数値的に安定して解を求めることができる。
