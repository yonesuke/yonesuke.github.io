---
title: "『Pythonによるはじめての数値流体力学』第2章"
date: 2025-12-20
slug: begginer_cfd_python_sec2
draft: false
math: true
authors:
    - yonesuke
categories:
    - Physics
    - Computational Fluid Dynamics
    - Python
---

『Pythonによるはじめての数値流体力学』という本を読み始めました。
この本は数値流体力学(CFD)の基礎的な内容をPythonで実装しながら学んでいくことを目的とした本です。
第2章まで読み進めたので、その内容を簡単にまとめておきます。
なお、本ではNumpyをベースとした実装が行われていますが、ここではJAXを用いた実装を紹介します。

<!-- more -->

## Jacobi法とは

- **定義**: 連立一次方程式 $Ax=b$ に対する反復法の一つで、対角成分で正規化して各変数を独立に更新します。
- **反復式**: 行列 $A$ を対角行列 $D$ と残差部分 $R(=A-D)$ に分解すると、$Ax=b$の解$x^*$は
  $$ x^{*} = D^{-1}(b - Rx^{*}) $$
  を満たします。これを反復式として用いると、$k+1$回目の反復での近似解$x^{(k+1)}$は
  $$ x^{(k+1)} = D^{-1}(b - Rx^{(k)}) $$
  と表されます。これがJacobi法の反復式です。また、$R=A-D$を用いると
    $$ x^{(k+1)} - x^{(k)} = D^{-1}(b-Ax^{(k)}) $$
    とも書けます。
- **収束条件**: Jacobi法はすべての$i$について
    $$ |a_{ii}| > \sum_{j \neq i} |a_{ij}| $$
    を満たす行列$A$に対して収束します。
    このことを行列$A$が**厳密に対角優位**であると言うそうです。
    ただし、あとで見るようにこれは十分条件であり、必須条件ではありません。

## JAXによる実装例

それでは実際にJAXを用いてJacobi法を実装してみます。

- **前処理**: 反復式において、$D^{-1} b$ と $D^{-1} A$ は事前に計算が可能なので、これらを計算しておきます。
- **収束判定**: 反復ごとに $|x^{(k+1)} - x^{(k)}|$ の和を計算し、これがある閾値以下になったら収束と見なします。
- **浮動小数点精度**: 収束判定や差分の計算で高精度な計算が必要になる場合があるため、`float64`を用います。JAXではデフォルトが`float32`なので、`jax.config.update("jax_enable_x64", True)`で`float64`を有効化します。
- **JAXの高速化テクニック**: 反復処理全体を`jax.lax.while_loop`でカプセル化し、XLAでコンパイル可能にします。また、`jax.debug.print`を用いて収束過程をモニタリングします。

```python
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import dataclasses

# Enable float64 precision in JAX
jax.config.update("jax_enable_x64", True)

@dataclasses.dataclass
class SolutionResult:
    success: bool
    solution: jax.Array
    num_iteration: int
    residual: float

def jacobi_solve(A: jax.Array, b: jax.Array, x: jax.Array, max_iteration: int, residual_tol: float, monitor: int = 100) -> SolutionResult:
    # normalization
    A_normalized = A / jnp.diag(A).reshape(-1, 1)
    b_normalized = b / jnp.diag(A)

    def cond_fun(state):
        k, _, residual = state
        return jnp.logical_and(k < max_iteration, residual >= residual_tol)

    def body_fun(state):
        k, x, _ = state
        dx = b_normalized - A_normalized @ x
        x = x + dx
        residual = jnp.sum(jnp.abs(dx))
        jax.lax.cond(
            k % monitor == 0,
            lambda _: jax.debug.print("iteration: {k}, residual: {residual}", k=k, residual=residual),
            lambda _: None,
            operand=None
        )
        return k + 1, x, residual

    initial_residual = jnp.sum(jnp.abs(b_normalized - A_normalized @ x))
    initial_state = (0, x, initial_residual)
    final_state = jax.lax.while_loop(cond_fun, body_fun, initial_state)
    k, x, residual = final_state

    success = residual.__float__() < residual_tol
    return SolutionResult(success, x, k.__int__(), residual.__float__())
```

## 数値実験

以下の行列$A$とベクトル$b$に対してJacobi法を適用してみます。

$$A = \begin{bmatrix} 4 & 0.8 & 2 \\ 0.4 & 1 & 0.6 \\ 0.5 & 3.5 & 5 \end{bmatrix}, \quad b = \begin{bmatrix} 1 \\ 0.5 \\ 4 \end{bmatrix}$$

```python
A = jnp.array([[4, 0.8, 2], [0.4, 1, 0.6], [0.5, 3.5, 5]])
b = jnp.array([1, 0.5, 4])
x = jnp.zeros_like(b)
solution = jacobi_solve(A, b, x, 10000, 1e-10)
print(solution)
```

実行結果は以下のようになります。(SolutionResultはきれいにフォーマットしています。)

```
iteration: 0, residual: 1.55
iteration: 100, residual: 2.1321799356655013e-07
SolutionResult(
    success=True,
    solution=Array([-0.13953488,  0.11627907,  0.73255814], dtype=float64),
    num_iteration=150,
    residual=8.97331098315135e-11
)
```

このように、150回の反復で収束したことが確認できます。
実は行列$A$は2行目が対角優位でないため、厳密には収束条件を満たしていませんが、収束しています。
これは厳密な収束条件が十分条件であり、必須条件ではないためです。

## まとめ

第2章では、Jacobi法の紹介とJAXを用いた実装例、そして数値実験を通じてその動作を確認しました。
第3章では拡散方程式の数値解法について学ぶ予定です。
