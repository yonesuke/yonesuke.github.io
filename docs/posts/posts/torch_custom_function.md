---
title: "PyTorchのカスタム関数を実装する"
date: 2026-04-07
slug: torch_custom_function
draft: false
math: true
authors:
    - yonesuke
categories:
    - Python
---

PyTorchでカスタム関数を実装する方法を調べていたので備忘録としてまとめておきます。

<!-- more -->

単純にPyTorchで関数を定義すると実行時間が遅くなってしまうことがあり、そういった場合には別の言語で書かれた関数を呼び出すことで高速化をすることができます。
しかし、そうすると自動微分が効かなくなってしまうという問題が発生します。
そこで、PyTorchではカスタム関数を定義し、backward関数を手動で設定することによって自動微分を効かせることが出来るようになるようです。
有名な方法としては、`torch.autograd.Function`を継承して、`forward`と`backward`メソッドを実装する方法があります。
しかし、この方法では`torch.compile`を使用することが出来ないため、`torch.library.custom_op`を使用してカスタム関数を定義することが推奨されているようです。
この記事ではそれぞれの方法でカスタム関数を実際に定義する方法を紹介していきます。

題材として、三重対角ソルバーを実装してみたいと思います。
scipyでは`scipy.linalg.solve_banded`という関数があったり、jaxでは`jax.lax.linalg.tridiagonal_solve`という関数があったりしますが、PyTorchには三重対角ソルバーが存在しないため、カスタム関数を定義してみたいと思います。(なんでPyTorchには三重対角ソルバーがないんでしょうかね？[issueで議論](https://github.com/pytorch/pytorch/issues/118225)はされているようですが、なかなか実装されないですね。)
scipyにはLAPACKの関数を呼び出す`scipy.linalg.get_lapack_funcs`という関数があり、こちらから三重対角ソルバーを呼び出すことができます。直接LAPACKの関数を呼び出しても良いかもしれませんが、ここではscipyを介してLAPACKの関数を呼び出す方法で実装してみたいと思います。

## LAPACKカーネル

まず、LAPACKの`?gtsv`ルーチンをラップするヘルパー関数を定義します。`?gtsv`は三重対角線形系 $Ax = b$ を直接法で解くルーチンで、`sgtsv`（単精度）・`dgtsv`（倍精度）などがあります。`get_lapack_funcs`に配列を渡すと、その dtype に合った実装を自動で選んでくれます。

```python
import numpy as np
from scipy.linalg.lapack import get_lapack_funcs

def _gtsv(
    lower: np.ndarray,
    diag: np.ndarray,
    upper: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """Call LAPACK ?gtsv (copies inputs to protect originals)."""
    dl = lower.copy()
    d = diag.copy()
    du = upper.copy()
    b = rhs.copy()
    (gtsv,) = get_lapack_funcs(("gtsv",), (dl, d, du, b))
    _, _, _, x, info = gtsv(dl, d, du, b)
    if info != 0:
        raise RuntimeError(f"LAPACK gtsv failed: info={info}")
    return x
```

## backward の導出

$Ax = b$ をスカラー損失 $L$ について微分することを考えます。
連鎖律と随伴法から

$$
\frac{\partial L}{\partial b} = A^{-T} \frac{\partial L}{\partial x}
$$

が成り立ちます。すなわち $v = A^{-T} \frac{\partial L}{\partial x}$ を求めるには $A^T$ に対して同じ三重対角ソルバーを呼べばよく、$A$ の下対角と上対角を入れ替えるだけで済みます。

行列微分 $\frac{\partial L}{\partial A} = -v x^T$ を三重対角成分に制限すると

$$
\frac{\partial L}{\partial \mathrm{diag}[i]}  = -v_i x_i, \quad
\frac{\partial L}{\partial \mathrm{upper}[i]} = -v_i x_{i+1}, \quad
\frac{\partial L}{\partial \mathrm{lower}[i]} = -v_{i+1} x_i
$$

バッチ RHS $X, B \in \mathbb{R}^{n \times k}$ の場合は各列について同じ式が成り立つので、$k$ 方向に sum を取ればよいです。

```python
import torch

def _compute_grads(v, x):
    if x.dim() == 1:
        grad_diag  = -v * x
        grad_upper = -v[:-1] * x[1:]
        grad_lower = -v[1:]  * x[:-1]
    else:  # (n, k) batched
        grad_diag  = -(v * x).sum(dim=1)
        grad_upper = -(v[:-1] * x[1:]).sum(dim=1)
        grad_lower = -(v[1:]  * x[:-1]).sum(dim=1)
    return grad_lower, grad_diag, grad_upper
```

## torch.autograd.Functionを継承してカスタム関数を定義する

まずは、`torch.autograd.Function`を継承してカスタム関数を定義する方法を紹介します。

```python
import torch
import numpy as np

class TridiagonalSolver(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lower, diag, upper, rhs):
        x_np = _gtsv(
            lower.detach().numpy(),
            diag.detach().numpy(),
            upper.detach().numpy(),
            rhs.detach().numpy(),
        )
        x = torch.tensor(x_np, dtype=diag.dtype)
        ctx.save_for_backward(lower, diag, upper, x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        lower, diag, upper, x = ctx.saved_tensors
        # A^T solve: swap lower and upper
        v_np = _gtsv(
            upper.detach().numpy(),
            diag.detach().numpy(),
            lower.detach().numpy(),
            grad_output.detach().numpy(),
        )
        v = torch.tensor(v_np, dtype=diag.dtype)
        grad_lower, grad_diag, grad_upper = _compute_grads(v, x)
        return grad_lower, grad_diag, grad_upper, v  # grad_rhs = v

def thomas_solve_lapack(lower, diag, upper, rhs):
    return TridiagonalSolver.apply(lower, diag, upper, rhs)
```

`ctx.save_for_backward`には forward で使う入力と出力 `x` を保存しておき、backward で取り出します。
`rhs` は backward で直接使わないので保存する必要はありません。
`grad_rhs` は $v$ そのものなので、そのまま返します。

## torch.library.custom_opを使用してカスタム関数を定義する

`torch.autograd.Function`では`torch.compile`を使用することが出来ません。
`torch.compile`に対応するには、`torch.library.custom_op`を使用してカスタム関数を定義する必要があります。

```python
import torch

@torch.library.custom_op("mylib::tridiagonal_solve", mutates_args=())
def _tridiagonal_solve_kernel(
    lower: torch.Tensor,
    diag: torch.Tensor,
    upper: torch.Tensor,
    rhs: torch.Tensor,
) -> torch.Tensor:
    x = _gtsv(lower.numpy(), diag.numpy(), upper.numpy(), rhs.numpy())
    return torch.tensor(x, dtype=diag.dtype)

@_tridiagonal_solve_kernel.register_fake
def _(lower, diag, upper, rhs):
    return rhs.new_empty(rhs.shape)
```

`register_fake`は`torch.compile`がシェイプ推論をするために必要な「形だけ」の実装です。実際には numpy 呼び出しは行いません。

次に、`torch.library.register_autograd`で backward を登録します。
`setup_context`で forward の入出力から backward に必要なテンソルを `ctx` に保存し、`backward`でそれを使って勾配を計算します。

```python
def _setup_context(ctx, inputs, output):
    lower, diag, upper, _ = inputs
    ctx.save_for_backward(lower, diag, upper, output)

def _backward(ctx, grad_output):
    lower, diag, upper, x = ctx.saved_tensors
    # backward でも同じカスタム op を再利用することで torch.compile がグラフ分割を回避できる
    v = _tridiagonal_solve_kernel(upper, diag, lower, grad_output)
    grad_lower, grad_diag, grad_upper = _compute_grads(v, x)
    return grad_lower, grad_diag, grad_upper, v

torch.library.register_autograd(
    "mylib::tridiagonal_solve",
    _backward,
    setup_context=_setup_context,
)

def thomas_solve_custom(lower, diag, upper, rhs):
    return _tridiagonal_solve_kernel(lower, diag, upper, rhs)
```

backward の中で `_tridiagonal_solve_kernel` を再利用しているのがポイントです。
`.numpy()` 呼び出しを含む実装を直接 backward に書くと `torch.compile` がグラフを分割してしまいますが、同じカスタム op を経由することで opaque なカーネルとして扱われ、グラフブレークを防げます。

## 動作確認

`torch.autograd.gradcheck`を使うと、数値微分と自動微分の結果を比較して勾配が正しいかどうかを確認することが出来ます。

```python
import numpy as np

rng = np.random.default_rng(42)
n = 10
lower_np = rng.standard_normal(n - 1) * 0.5
upper_np = rng.standard_normal(n - 1) * 0.5
diag_np  = np.abs(rng.standard_normal(n)) + 3.0
rhs_np   = rng.standard_normal(n)

def to_tensor(a, grad=False):
    return torch.tensor(a, dtype=torch.float64, requires_grad=grad)

args_grad = [to_tensor(a, grad=True) for a in [lower_np, diag_np, upper_np, rhs_np]]

for name, fn in [
    ("autograd.Function", thomas_solve_lapack),
    ("custom_op        ", thomas_solve_custom),
]:
    ok = torch.autograd.gradcheck(fn, args_grad, eps=1e-6, atol=1e-5)
    print(f"{name}: {ok}")
```

また、`torch.compile`に対応した`thomas_solve_custom`については、コンパイル後も正しく動作することを確認できます。

```python
def model(lo, di, up, b):
    return thomas_solve_custom(lo, di, up, b).pow(2).sum()

compiled = torch.compile(model)
lo, di, up, b = [to_tensor(a, grad=True) for a in [lower_np, diag_np, upper_np, rhs_np]]
loss = compiled(lo, di, up, b)
loss.backward()

# グラフブレークがないことを確認
expl = torch._dynamo.explain(model)(lo, di, up, b)
print(f"graph breaks: {len(expl.break_reasons)}")  # => 0
```