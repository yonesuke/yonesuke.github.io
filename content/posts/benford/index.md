---
title: "ベンフォードの法則"
date: 2023-07-01T09:40:29+09:00
draft: true
math: true
author: Ryosuke Yoneda
---

ベンフォードの法則は「自然界に現れる多くの数値の最初の桁の値はある特定の分布に従う」ことを指す法則である。
ふと出くわして非常に面白かったのでまとめておく。

## ベンフォードの法則

## 指数関数の場合
べき乗則に従うようなデータに対しては、ベンフォードの法則が成り立つことが簡単に示せる。
簡単に$n$番目のデータが$b^n$の場合を考える。
$b^n$の最初の桁の値は$\log_{10}$をとった際の小数点によって決まる。


## 具体例

### フィボナッチ数列

フィボナッチ数列は$a_{0}=a_{1}=1$として、一般の$n\geq2$について
$$
a_{n}=a_{n-1} + a_{n-2}
$$
で定められる数列$\\{a_{n}\\}$を指す。
pythonでこの数列を実装するには再帰を利用する必要があるが、メモ化を行うことで高速化することができる。pythonでは幸い`functools.cache`が提供されているので簡単に実装ができる。
```python
from functools import cache

@cache
def fib(n):
    if n == 0 or n == 1:
        return 1
    return fib(n-1) + fib(n-2)
```
この関数を用いて、フィボナッチ数列の最初の桁の値の分布を調べてみよう。
```python
import numpy as np
import matplotlib.pyplot as plt

def first_digit(n):
    return int(str(n)[0])

n_max = 1000
fib_first_digits = np.bincount([first_digit(fib(n)) for n in range(n_max)])

plt.bar(range(10), fib_first_digits / n_max)
plt.xlabel("First digit")
plt.ylabel("Frequency")
plt.show()
```
{{< figure src="image.png" width=700 >}}

### 日本の人口