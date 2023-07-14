---
title: "ベンフォードの法則"
date: 2023-07-01T09:40:29+09:00
draft: false
math: true
author: Ryosuke Yoneda
---

ベンフォードの法則は「自然界に現れる多くの数値の最初の桁の値はある特定の分布に従う」ことを指す法則である。
ふと出くわして非常に面白かったのでまとめておく。

## ベンフォードの法則
ベンフォードの法則は以下のように定義される。
自然界に現れる多くの数値の最初の桁の値は、一様分布ではなく、次のような分布に従う。
$$
P(d) = \log_{10}\left(1 + \frac{1}{d}\right)
$$
ただし、$d=1,2,\dots,9$である。
もちろんこれは常に正しいわけではなく、経験的にそうなっている、というものである。
ただ、以下でみるようにべき的なふるまいを見せるデータに対してはベンフォードの法則が成り立つことが示せる。
この意味で、むしろ自然界にはべき的なふるまいをするデータがたくさんあることはよく知られているので、
ベンフォードの法則が成り立つようなデータが多くあることは自然なのかもしれない。

## 指数関数の場合
べき乗則に従うようなデータに対しては、ベンフォードの法則が成り立つことが簡単に示せる。
簡単に$n$番目のデータが$b^n$の場合を考える。

$b^n$の最初の桁の値は$\log_{10}$をとった際の小数点によって決まる。
例えば、$b^{3}$の先頭が5ならば、$\log_{10} 5\leq 3\log_{10}b\ (\bmod 1)<\log_{10} 6$である。
よって、$n\log_{10} b \ (\bmod 1)$が$[0, 1]$区間で従う分布がそのまま$b^n$の一桁目の分布に従う。
**$\log_{10} b$が無理数ならば**、ワイルの均等分布定理から一様分布に従うことがわかる。
すなわち、十分に大きな$n$において、先頭の桁が$d$となる確率は
$$
\log_{10}(d+1) - \log_{10} d = \log_{10}\left(1 + \frac{1}{d}\right)
$$
とわかる。これはベンフォードの法則の法則に他ならない。


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

plt.plot(np.arange(1, 10), fib_first_digits[1:] / n_max, label='Fibonacci')
xs = np.arange(1, 10)
plt.plot(xs, np.log10(1 + 1 / xs), label='Benford')
plt.legend()
```
{{< figure src="benford_fib.png" width=700 >}}

図を見てもわかるように、フィボナッチ数列の先頭の桁がベンフォードの法則に従うことが確かめられた。
これが成り立つことは上の指数関数の例から証明を行うことができる。
フィボナッチ数列の一般項は、
$$
a_{n} = \frac{1}{\sqrt{5}}\left(\frac{1+\sqrt{5}}{2}\right)^{n} - \frac{1}{\sqrt{5}}\left(\frac{1-\sqrt{5}}{2}\right)^{n}
$$
で与えられる。
第2項は$|(1-\sqrt{5})/2|<1$なので、$n$が大きくなれば無視できる項となる。
そうすると、フィボナッチ数列はべき的なふるまいをするとみなすことができるので、
先ほどの指数関数の場合と同様にベンフォードの法則が成り立つことがわかる。

### 世界の人口

世界の国々の人口についてもベンフォードの法則が成り立つのかを確認してみよう。
世界の人口は`pandas_datareader`を利用すると簡単に取得することができる。

```python
from pandas_datareader import wb

df = wb.download(indicator='SP.POP.TOTL', country='all', start=2000, end=2022)
df = df.pivot_table(columns='country', index='year', values='SP.POP.TOTL')
print(df.iloc[:5,:5].to_markdown())
```

|   year |   Afghanistan |   Africa Eastern and Southern |   Africa Western and Central |     Albania |     Algeria |
|-------:|--------------:|------------------------------:|-----------------------------:|------------:|------------:|
|   2000 |   1.9543e+07  |                   4.01601e+08 |                  2.69612e+08 | 3.08903e+06 | 3.07746e+07 |
|   2001 |   1.96886e+07 |                   4.12002e+08 |                  2.7716e+08  | 3.06017e+06 | 3.1201e+07  |
|   2002 |   2.10003e+07 |                   4.22741e+08 |                  2.84952e+08 | 3.05101e+06 | 3.16247e+07 |
|   2003 |   2.26451e+07 |                   4.33807e+08 |                  2.92978e+08 | 3.03962e+06 | 3.20559e+07 |
|   2004 |   2.35536e+07 |                   4.45282e+08 |                  3.01265e+08 | 3.02694e+06 | 3.25102e+07 |

このデータに対して、各年で最初の桁の数字の分布を確認する。

```python
import pandas as pd
df_count = pd.DataFrame(
    df.applymap(lambda n: int(str(n)[0])).apply(np.bincount, axis=1).tolist(),
    index = df.index
)

import matplotlib.pyplot as plt
for year, v in df_count.iterrows():
    plt.plot(
        v.index[1:], v.values[1:] / v.sum(),
        color='tab:blue', alpha=(int(year) - 2000) / 22,
        label=year
    )
xs = np.arange(1, 10)
plt.plot(xs, np.log10(1 + 1 / xs), color='tab:orange', label='Benford')
plt.legend(ncols=3)
```

{{< figure src="benford_population.png" width=700 >}}

世界の人口についてもベンフォードの法則への当てはまりが良いことがわかる。

## まとめ

ベンフォードの法則を簡単に紹介し、指数関数に従うデータに対してはこの法則が成り立つことを確認した。
また、フィボナッチ数列や世界の人口についてもベンフォードの法則が成り立つことをみた。
ベンフォードの法則は、データの偽造を検出するために利用されることもあるそうなので、
気が向いたらそちらも試してみたいと思う。