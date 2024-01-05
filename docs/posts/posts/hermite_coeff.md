---
title: "Hermite多項式の係数"
date: 2023-01-31
slug: hermite_coeff
draft: false
math: true
authors:
    - yonesuke
categories:
    - Python
    - Mathematics
---

Hermite多項式の係数をpythonで求める方法を紹介します。
係数自体は三項間漸化式で求められますが、高次の係数を求めるときには再帰が必要になり計算量が増えてしまいます。
`functools`モジュールの`cache`を使うと再帰を高速化できます。

<!-- more -->

```python
from functools import cache

@cache
def hermite_coeff(n: int) -> list:
    """Compute the coefficients of the nth Hermite polynomial."""
    if not isinstance(n, int):
        raise TypeError("n must be an integer")
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return [1]
    elif n == 1:
        return [0, 2]
    else:
        return [- 2 * (n - 1) * hermite_coeff(n - 2)[0]]\
            + [2 * hermite_coeff(n - 1)[i] - 2 * (n - 1) * hermite_coeff(n - 2)[i + 1] for i in range(n - 2)]\
            + [2 * hermite_coeff(n - 1)[n - 2], 2 * hermite_coeff(n - 1)[n - 1]]
```

`hermite_coeff(n)`はn次のHermite多項式の係数を返します。
`hermite_coeff(n)`は`n`が整数でないとき、`n`が負のときに`TypeError`と`ValueError`を返します。
    
```python
for n in range(11):
    print(f"coeff of H_{n} = {hermite_coeff(n)}")
```

```text
coeff of H_0 = [1]
coeff of H_1 = [0, 2]
coeff of H_2 = [-2, 0, 4]
coeff of H_3 = [0, -12, 0, 8]
coeff of H_4 = [12, 0, -48, 0, 16]
coeff of H_5 = [0, 120, 0, -160, 0, 32]
coeff of H_6 = [-120, 0, 720, 0, -480, 0, 64]
coeff of H_7 = [0, -1680, 0, 3360, 0, -1344, 0, 128]
coeff of H_8 = [1680, 0, -13440, 0, 13440, 0, -3584, 0, 256]
coeff of H_9 = [0, 30240, 0, -80640, 0, 48384, 0, -9216, 0, 512]
coeff of H_10 = [-30240, 0, 302400, 0, -403200, 0, 161280, 0, -23040, 0, 1024]
```

`functools.cache`を知らなかったので学びになりました。
他の再帰を含むような計算、例えばフィボナッチ数列の計算でも使えると思います。