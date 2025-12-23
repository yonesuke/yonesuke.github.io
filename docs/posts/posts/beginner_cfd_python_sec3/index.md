---
title: "『Pythonによるはじめての数値流体力学』第3章"
date: 2025-12-21
slug: beginner_cfd_python_sec3
draft: false
math: true
authors:
    - yonesuke
categories:
    - Physics
    - Computational Fluid Dynamics
    - Python
---

引き続き『Pythonによるはじめての数値流体力学』という本を読み進めています。
第3章では、拡散方程式の数値解法について学びました。
理論の解説とJAXによる実装を行います。

<!-- more -->

## 拡散方程式と離散化手法

拡散方程式は、物質やエネルギーが空間内でどのように広がるかを記述する偏微分方程式です。1次元の拡散方程式は物理量$f(t,x)$に関する以下の式で表されます：

$$
\frac{\partial f}{\partial t} = \gamma \frac{\partial^2 f}{\partial x^2} + c
$$

$\gamma$ は拡散係数、$c$ はソースの項です。時刻は$t\in [0, \infty)$、空間は$x\in [0, 1]$とします。

- **境界条件**: 空間方向の境界条件について、任意の時刻$t$に対して$f(t, 0) = f(t, 1) = 0$とします。
- **初期条件**: 時刻$t=0$において、$f(0, x) = 0$とします。

実はこの条件のもとで十分時間が経過したときの定常状態として、$f(t, x) = \frac{c}{2\gamma} x(1-x)$が成り立ちます。
のちの数値計算で定常解と数値解の比較を行います。

## 離散化
この問題を解くために、空間方向について$x=0$から$x=1$に$N$個の格子点を配置します。
点の間隔は

$$
\Delta x = \frac{1}{N-1}
$$

です。

### オイラー陽解法

オイラー陽解法は、時刻$m\Delta t$における$f$の離散値$f^{[m]}_i$をもとに次の時刻$(m+1)\Delta t$における$f$の値を計算する手法です。

$$
\frac{f^{[m+1]}_i - f^{[m]}_i}{\Delta t} = \gamma \frac{f^{[m]}_{i+1} - 2f^{[m]}_i + f^{[m]}_{i-1}}{(\Delta x)^2} + c.
$$

これを整理すると、

$$
f^{[m+1]}_i = f^{[m]}_i + \frac{\gamma \Delta t}{(\Delta x)^2} (f^{[m]}_{i+1} - 2f^{[m]}_i + f^{[m]}_{i-1}) + c \Delta t
$$

となります。

- **収束条件**: オイラー陽解法が収束するためには次が必要です。

$$
\frac{\gamma \Delta t}{(\Delta x)^2} \leq \frac{1}{2}.
$$





### オイラー陰解法

オイラー陰解法は、次の時刻$(m+1)\Delta t$における$f$の離散値$f^{[m+1]}_i$をもとに計算を行う手法です。

$$
\frac{f^{[m+1]}_i - f^{[m]}_i}{\Delta t} = \gamma \frac{f^{[m+1]}_{i+1} - 2f^{[m+1]}_i + f^{[m+1]}_{i-1}}{(\Delta x)^2} + c
$$

これを整理すると$f^{[m+1]}$は次の連立一次方程式を満たします。

$$
-\frac{\gamma \Delta t}{(\Delta x)^2} f^{[m+1]}_{i-1} + \left(1 + \frac{2\gamma \Delta t}{(\Delta x)^2}\right) f^{[m+1]}_i - \frac{\gamma \Delta t}{(\Delta x)^2} f^{[m+1]}_{i+1} = f^{[m]}_i + c \Delta t
$$

行列形式で表すと、

$$
A f^{[m+1]} = f^{[m]} + c \Delta t,\\
a_{i-1, i}=a_{i, i+1} = -\frac{\gamma \Delta t}{(\Delta x)^2},\ a_{i, i} = 1 + \frac{2\gamma \Delta t}{(\Delta x)^2}.
$$

陽解法では$f^{[m]}$から直接次のステップの計算が可能でしたが、その点陰解法では都度上の連立一次方程式を解く必要があります。

## JAXによる実装

- **陰解法**: 解くべき連立一次方程式は疎な行列になります。空間グリッド数$n$に対して要素が$(n-3)\times 2 + n-2=3n-8$個しかないような行列です。ここではJAXの`sparse`モジュールを用いて疎行列を扱い、Jacobi法での行列積の高速化を図ります。
- **JAXによる高速化**: 前回に引き続き、反復処理全体を`jax.lax.fori_loop`でカプセル化し、XLAでコンパイル可能にします。

```python
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import dataclasses

jax.config.update("jax_enable_x64", True)

@dataclasses.dataclass
class SolutionResult:
    success: bool
    solution: jax.Array
    num_iteration: int
    residual: float

def jacobi_solve(A: jax.Array, b: jax.Array, x: jax.Array, max_iteration: int, residual_tol: float, monitor: int = 100) -> SolutionResult:
    
    def cond_fun(state):
        k, _, residual = state
        return jnp.logical_and(k < max_iteration, residual >= residual_tol)

    def body_fun(state):
        k, x, _ = state
        dx = b - A @ x
        x = x + dx
        residual = jnp.sum(jnp.abs(dx))
        jax.lax.cond(
            k % monitor == 0,
            lambda _: jax.debug.print("iteration: {k}, residual: {residual}", k=k, residual=residual),
            lambda _: None,
            operand=None
        )
        return k + 1, x, residual

    initial_residual = jnp.sum(jnp.abs(b - A @ x))
    initial_state = (0, x, initial_residual)
    final_state = jax.lax.while_loop(cond_fun, body_fun, initial_state)
    k, x, residual = final_state

    success = residual < residual_tol
    return SolutionResult(success, x, k, residual)

def solve_1d_diffusion_explicit(num_grid_x: int, time_step: float, num_steps: int, gamma: float, c: float) -> jax.Array:
    dx = 1.0 / (num_grid_x - 1)
    r = gamma * time_step / dx ** 2
    
    u = jnp.zeros((num_grid_x,))

    def body_fun(n, u):
        du = r * (jnp.diff(jnp.diff(u))) + c * time_step
        return u.at[1:-1].add(du)

    u = jax.lax.fori_loop(0, num_steps, body_fun, u)

    return u

def solve_1d_diffusion_implicit(num_grid_x: int, time_step: float, num_steps: int, gamma: float, c: float, max_iteration: int = 1000, residual_tol: float = 1e-6) -> jax.Array:
    dx = 1.0 / (num_grid_x - 1)
    r = gamma * time_step / dx ** 2

    indices_lower = [[i, i-1] for i in range(1, num_grid_x - 2)]
    values_lower = [ -r for _ in range(1, num_grid_x - 2)]
    indices_diag = [[i, i] for i in range(num_grid_x - 2)]
    values_diag = [1.0 + 2.0 * r for _ in range(num_grid_x - 2)]
    indices_upper = [[i, i+1] for i in range(num_grid_x - 3)]
    values_upper = [ -r for _ in range(num_grid_x - 3)]
    indices = jnp.array(indices_lower + indices_diag + indices_upper)
    values = jnp.array(values_lower + values_diag + values_upper)
    A = sparse.BCOO((values, indices), shape=(num_grid_x - 2, num_grid_x - 2))

    # normalize A
    A = A / (1.0 + 2.0 * r)

    u = jnp.zeros((num_grid_x,))
    
    def body_fun(n, u):
        b = u[1:-1] + c * time_step
        b = b / (1.0 + 2.0 * r) # normalize b
        result = jacobi_solve(A, b, u[1:-1], max_iteration, residual_tol) # initial guess: u[1:-1]
        return u.at[1:-1].set(result.solution)

    u = jax.lax.fori_loop(0, num_steps, body_fun, u)

    return u
```

## 数値実験

- オイラー陽解法

パラメータは$\gamma=1.0, c=4.0$とし、$\Delta t = 1.0\times 10^{-3}$として、1000時間ステップの計算を行います。グリッド数を変えていったときの定常解との誤差を調べます。

```python
gamma = 1.0
c = 4.0
xs_true = jnp.linspace(0.0, 1.0, 100)
ys_true = (c / (2.0 * gamma)) * xs_true * (1.0 - xs_true)
plt.plot(xs_true, ys_true, label="Analytical Solution", color="gray", linestyle="dashed")

time_step = 1e-3
num_steps = 1000

nums_grid_x = [15, 20, 25, 30, 35, 40]
for num_grid_x in nums_grid_x:
    xs = jnp.linspace(0.0, 1.0, num_grid_x)
    solution = solve_1d_diffusion_explicit(num_grid_x, time_step, num_steps, gamma, c)
    plt.scatter(xs, solution, label=f"Explicit: {num_grid_x} grid points")

plt.xlabel("x")
plt.ylabel("u")
plt.ylim(-0.5, 1.5)
plt.legend()
plt.title("1D Diffusion Equation: Explicit Method")
plt.show()
```

![alt text](euler_explicit.png)

グリッド数をあげていくと$N=25$以降で解が不安定になる様子が確認されます。
実際陽解法の安定性と照らし合わせると、$\gamma=1.0, \Delta t=1.0\times 10^{-3}$のもとで$\Delta x \geq 0.044...$が安定性のために必要です。グリッド数としては$N\leq23.3...$となり、実際に数値実験結果とも一致していることがわかります。

- オイラー陰解法

パラメータは$\gamma=1.0, c=4.0$とし、$\Delta t = 1.0\times 10^{-2}$として、1000時間ステップの計算を行います。グリッド数を変えていったときの定常解との誤差を調べます。

```python
plt.plot(xs_true, ys_true, label="Analytical Solution", color="gray", linestyle="dashed")

for num_grid_x in nums_grid_x:
    xs = jnp.linspace(0.0, 1.0, num_grid_x)
    solution = solve_1d_diffusion_implicit(num_grid_x, time_step, num_steps, gamma, c)
    plt.scatter(xs, solution, label=f"Implicit: {num_grid_x} grid points")

plt.xlabel("x")
plt.ylabel("u")
plt.ylim(-0.5, 1.5)
plt.legend()
plt.title("1D Diffusion Equation: Implicit Method (Jacobi)")
plt.show()
```

![alt text](euler_implicit.png)

いずれのグリッド数においても解が求められており、解析解とも一致することがわかります。


## Thomasのアルゴリズム

オイラー陰解法に登場する行列$A$は3重対角行列です。実は3重対角行列はThomasのアルゴリズムによって容易に解くことが可能です。反復的な行列積を利用するJacobi法と異なるため非常に高速に動作することも特徴です。JAXでは`jax.lax.linalg.tridiagonal_solve`を利用すると良いです。

```python
def solve_1d_diffusion_thomas(num_grid_x: int, time_step: float, num_steps: int, gamma: float, c: float) -> jax.Array:
    dx = 1.0 / (num_grid_x - 1)
    r = gamma * time_step / dx ** 2

    A_lower = jnp.full((num_grid_x - 2,), -r).at[0].set(0.0)
    A_diag = jnp.full((num_grid_x - 2,), 1.0 + 2.0 * r)
    A_upper = jnp.full((num_grid_x - 2,), -r).at[-1].set(0.0)
    u = jnp.zeros((num_grid_x,))

    def body_fun(n, u):
        b = u[1:-1] + c * time_step
        new_u = jax.lax.linalg.tridiagonal_solve(A_lower, A_diag, A_upper, b.reshape(-1, 1)).reshape(-1)
        return u.at[1:-1].set(new_u)
    
    u = jax.lax.fori_loop(0, num_steps, body_fun, u)

    return u
```

実際に速度比較を行ってみます。少し大きめのグリッド数での実験を行いました。

```python
%%timeit
_ = solve_1d_diffusion_implicit(500, time_step, 5000, gamma, c)

%%timeit
_ = solve_1d_diffusion_thomas(500, time_step, 5000, gamma, c)
```

Jacobi法を利用した陰解法の実行時間が5.95 s ± 879 ms per loopで、Thomasのアルゴリズムを利用した実行時間が181 ms ± 23.7 ms per loopとなりました。
実に**33倍の高速化**です！

## まとめ

第3章では、拡散方程式の数値計算手法としてオイラー陽解法・陰解法の紹介とJAXを用いた実装例、そして数値実験を通じてその動作を確認しました。また、陰解法についてはJacobi法からThomasのアルゴリズムへの切り替えによる高速化を確認しました。
第4章では対流方程式の数値解法について学ぶ予定です。
