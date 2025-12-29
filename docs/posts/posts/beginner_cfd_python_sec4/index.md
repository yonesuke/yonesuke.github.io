---
title: "『Pythonによるはじめての数値流体力学』第4章"
date: 2025-12-30
slug: beginner_cfd_python_sec4
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
第4章では、対流方程式の数値解法について学びました。
理論の解説とJAXによる実装を行います。

<!-- more -->

## 対流方程式と数値的安定性

対流方程式（移流方程式）は、ある物理量 $f$ が速度 $u$ の流れに乗って運ばれる様子を記述します。

$$
\frac{\partial f}{\partial t} + u \frac{\partial f}{\partial x} = 0
$$

### 数値安定性の鍵：CFL条件
対流方程式を陽解法で解く際、時間刻み $\Delta t$ と格子間隔 $\Delta x$ は **CFL条件(Courant-Friedrichs-Lewy Condition)** を満たす必要があります。

$$
C = \frac{u \Delta t}{\Delta x} \leq 1
$$

このクーラン数 $C$ が1を超えると、物理現象が計算網（格子）を追い越してしまい、計算が即座に発散します。物理的な情報伝達速度よりも計算速度が遅くなってはいけない、という極めて重要な制約です。

## 離散化手法：1次風上差分（Upwind Scheme）

物理量が上流側から流れてくるという物理的な直感に基づいた手法が**風上差分**です。

* **$u > 0$（左から右への流れ）のとき**:
  $$\frac{f_i^{[m+1]} - f_i^{[m]}}{\Delta t} + u \frac{f_i^{[m]} - f_{i-1}^{[m]}}{\Delta x} = 0$$
* **$u < 0$（右から左への流れ）のとき**:
  $$\frac{f_i^{[m+1]} - f_i^{[m]}}{\Delta t} + u \frac{f_{i+1}^{[m]} - f_i^{[m]}}{\Delta x} = 0$$


1次風上差分は非常に安定していますが、テイラー展開を用いて誤差項を分析すると、意図しない**拡散項**が計算に含まれてしまいます。

$$
\frac{\partial f}{\partial t} + u \frac{\partial f}{\partial x} \approx \frac{u \Delta x}{2} \frac{\partial^2 f}{\partial x^2}
$$

右辺が拡散方程式の拡散項と同じ形をしています。これが**数値粘性**と呼ばれるもので、本来くっきりしているはずの波が、時間の経過とともになだらかになってしまいます。

## 高次風上差分

1次風上差分は安定的である一方で、数値粘性がかなり強いです。
そこで、安定かつ高精度な手法を求めて高次の差分近似式を用いる手法が提案されています。
ここではQUICK法とCIP法を紹介します。
なお、この二つ手法はともに風上差分をベースとしています。

### QUICK法
QUICK法では空間方向を$\pm \Delta x/2$幅で中心差分をとります。

$$\frac{f_i^{[m+1]} - f_i^{[m]}}{\Delta t} + u \frac{f_{i+1/2}^{[m]} - f_{i-1/2}^{[m]}}{\Delta x} = 0$$

$u>0$として具体的に$f_{i\pm1/2}^{[m]}$を求めていきます。
- $f_{i+1/2}$について、3点 $(x_{i-1}, f_{i-1}), (x_i, f_i), (x_{i+1}, f_{i+1})$ を通る2次関数 $g(x)$ を仮定し、$x_{i+1/2}$ での値を求めます。
    $$f_{i+1/2} = \frac{1}{8} (3f_{i+1} + 6f_i - f_{i-1})$$
    <details>
    <summary>具体的な計算</summary>

    $g(x)=a(x-x_i)^2+b(x-x_i)+c$とすると、$f_{i-1}=g(x_{i-1})=a\Delta x^{2}-b\Delta x+c,f_i=g(x_i)=c,f_{i+1}=g(x_{i+1})=a\Delta x^{2}+b\Delta x+c$です。これを連立して、
    $$
    a=\frac{f_{i+1}-2f_i+f_{i-1}}{2\Delta x^2},\quad b=\frac{f_{i+1}-f_{i-1}}{2\Delta x},\quad c=f_{i}
    $$
    が得られます。$f_{i+1/2}=g(x_i+\Delta x/2)$を計算すると、上記式が求まります。

    </details>
- 同様に、もう一方$f_{i-1/2}$を求めます。使用する点は $(x_{i-2}, f_{i-2}), (x_{i-1}, f_{i-1}), (x_i, f_i)$ となり、これらを用いて $x_{i-1/2}$ での値を近似します。上の式のインデックスをすべて $-1$ することで求められます。
    $$f_{i-1/2} = \frac{1}{8} (3f_{i} + 6f_{i-1} - f_{i-2})$$

元の式に代入して整理すると、
$$f_i^{[m+1]} = f_i^{[m]} - \frac{u \Delta t}{8 \Delta x} (3f_{i+1} + 3f_i - 7f_{i-1} + f_{i-2})$$
となります。

### CIP法
物理量$f$とその勾配$g$も数値的に求めていく方法がCIP法です。
$u>0$で$x_i$の上流側の物理量$f$が3次関数$F_i^n(x)$を通るとします。
$$
F_i^n(x)=a_i (x-x_i)^3+b_i (x-x_i)^2+c_i(x-x_i)+d_i.
$$
物理量とその勾配がこの関数を通るとすると、$F_i^n(x_i)=f_i^n,\frac{dF_i^n}{dx}(x_i)=g_i^n,F_i^n(x_{i-1})=f_{i-1}^n,\frac{dF_i^n}{dx}(x_{i-1})=g_{i-1}^n$となるので、ここから$a_i,b_i,c_i,d_i$が求まります。

時刻ステップが一つ進むと対応して$F_i^{n+1}(x)=F_i^{n}(x-u\Delta t)$となるので、
$$
f_i^{n+1}=F_i^{n}(x_i-u\Delta t),\quad g_i^{n+1}=\frac{dF_i^{n}}{dx}(x_i-u\Delta t)
$$
を計算すれば良いです。

<details>
<summary>具体的な計算</summary>

まず、$F_i^n(x)$ を微分すると、$$\frac{dF_i^n}{dx}(x) = 3a_i (x-x_i)^2 + 2b_i (x-x_i) + c_i$$となります。
1. $x=x_i$ (つまり $x-x_i=0$) の条件
    - $F_i^n(x_i) = d_i = f_i^n$
    - $\frac{dF_i^n}{dx}(x_i) = c_i = g_i^n$

    これにより、$c_i, d_i$ が決定します。
2. $x=x_{i-1}$ (つまり $x-x_i=-\Delta x$) の条件
    上流点 $i-1$ での値 $f_{i-1}^n$ と勾配 $g_{i-1}^n$ を用いて連立方程式を作ります。
    $$
    \begin{aligned}
    f_{i-1}^n &= -a_i \Delta x^3 + b_i \Delta x^2 - c_i \Delta x + d_i \\
    g_{i-1}^n &= 3a_i \Delta x^2 - 2b_i \Delta x + c_i
    \end{aligned}
    $$
    既知の $c_i=g_i^n, d_i=f_i^n$ を代入し、$a_i, b_i$ について整理して解くと以下のようになります。
    $$
    \begin{aligned}
    a_i &= \frac{g_i^n + g_{i-1}^n}{\Delta x^2} - \frac{2(f_i^n - f_{i-1}^n)}{\Delta x^3}, \\
    b_i &= \frac{3(f_{i-1}^n - f_i^n)}{\Delta x^2} + \frac{2g_i^n + g_{i-1}^n}{\Delta x}.
    \end{aligned}
    $$

</details>

## JAXによる実装

JAXを用いて、これら3つの手法を実装しました。特にCIP法では、値と勾配をタプルで管理し、`jax.lax.fori_loop` で効率的に更新しています。

```python
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

def solve_1d_convection_upwind(num_grid_x: int, time_step: float, num_steps: int, u: float, initial_f: jax.Array) -> jax.Array:
    dx = 1 / (num_grid_x - 1)
    courant_number = u * time_step / dx
    
    if u >= 0:
        def body_fun(n, f):
            df = -courant_number * jnp.diff(f) # shape: (num_grid_x - 1,)
            return f.at[1:].add(df)
    else:
        def body_fun(n, f):
            df = -courant_number * jnp.diff(f)
            return f.at[:-1].add(df)
        
    return jax.lax.fori_loop(0, num_steps, body_fun, initial_f)

def solve_1d_convection_quick(num_grid_x: int, time_step: float, num_steps: int, u: float, initial_f: jax.Array) -> jax.Array:
    dx = 1 / (num_grid_x - 1)
    courant_number = u * time_step / dx
    
    if u >= 0:
        def body_fun(n, f):
            df = -0.125 * courant_number * (3 * f[3:] + 3 * f[2: -1] -7 * f[1:-2] + f[:-3])
            f = f.at[2:-1].add(df)
            f = f.at[:2].set(f[2]).at[-1].set(f[-2])
            return f
    else:
        def body_fun(n, f):
            return ...
    
    return jax.lax.fori_loop(0, num_steps, body_fun, initial_f)

def solve_1d_convection_cip(num_grid_x: int, time_step: float, num_steps: int, u: float, initial_f: jax.Array) -> jax.Array:
    dx = 1 / (num_grid_x - 1)
    
    if u >= 0:
        def body_fun(n, state):
            f, g = state
            coeff_a = (-2 * (f[1:] - f[:-1]) + dx * (g[1:] + g[:-1])) / dx ** 3
            coeff_b = (-3 * (f[1:] - f[:-1]) + dx * (g[:-1] + 2 * g[1:])) / dx ** 2
            coeff_c = g[1:]
            coeff_d = f[1:]
            x = -u * time_step
            f = f.at[1:].set(coeff_a * x ** 3 + coeff_b * x ** 2 + coeff_c * x + coeff_d)
            g = g.at[1:].set(3 * coeff_a * x ** 2 + 2 * coeff_b * x + coeff_c)
            return f, g
    else:
        def body_fun(n, state):
            f, g = state
            coeff_a = (2 * (f[:-1] - f[1:]) + dx * (g[:-1] + g[1:])) / dx ** 3
            coeff_b = (3 * (f[1:] - f[:-1]) - dx * (2 * g[:-1] + g[1:])) / dx ** 2
            coeff_c = g[:-1]
            coeff_d = f[:-1]
            x = -u * time_step
            f = f.at[:-1].set(coeff_a * x ** 3 + coeff_b * x ** 2 + coeff_c * x + coeff_d)
            g = g.at[:-1].set(3 * coeff_a * x ** 2 + 2 * coeff_b * x + coeff_c)
            return f, g
    
    initial_g = jnp.zeros_like(initial_f).at[1:-1].set((initial_f[2:] - initial_f[:-2]) / (2 * dx))
    f, _ =  jax.lax.fori_loop(0, num_steps, body_fun, (initial_f, initial_g))
    return f
```

## 数値実験

風上差分・QUICK・CIPの3つの手法を実際に比較していきましょう。

```python
u = 1.0
num_grid_x = 40
time_step = 1e-3
num_steps = 300

xs = jnp.linspace(0, 1, num_grid_x)
initial_f = jnp.where(xs < 0.2, 1.0, 0.0)
exact_f = jnp.where(xs < 0.2 + u * time_step * num_steps, 1.0, 0.0)

f_upwind = solve_1d_convection_upwind(num_grid_x, time_step, num_steps, u, initial_f)
f_quick  = solve_1d_convection_quick(num_grid_x, time_step, num_steps, u, initial_f)
f_cip    = solve_1d_convection_cip(num_grid_x, time_step, num_steps, u, initial_f)

plt.figure(figsize=(10, 6))
plt.plot(xs, initial_f, color="gray", alpha=0.5, ls=":", label="Initial Distribution")
plt.plot(xs, exact_f, marker="x", color="black", ls="-", lw=1.5, label="Exact Distribution")
plt.plot(xs, f_upwind, marker="o", markerfacecolor="white", ls="--", color="blue", label="Upwind (1st Order)")
plt.plot(xs, f_quick, marker="^", markerfacecolor="white", ls="--", color="green", label="QUICK (3rd Order)")
plt.plot(xs, f_cip, marker="s", markerfacecolor="white", ls="--", color="red", label="CIP Scheme")

plt.xlabel("x")
plt.ylabel("f")
plt.title(f"1D Convection Comparison (N={num_grid_x}, Steps={num_steps})")
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.tight_layout()

plt.show()
```

![alt text](image.png)

- 風上差分法は安定しているものの、数値差分の影響で解が滑らかになってしまっています。
- QUICK法は風上差分法よりも精度は高くなりますが、数値振動が発生しています。
- CIP法がもっとも真の解に近く、形状を維持しています。

## まとめ

第4章では対流方程式の数値解法について学びました。数値拡散をいかに抑え込むでさまざまな手法が提案されていることを確認しました。
第5章からはいよいよ流れ解析の手法についてみていきます。