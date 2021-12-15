---
title: "Egorov's theorem"
date: 2021-11-26T13:05:46+09:00
draft: true
---

Egorovの定理は関数列の各点収束と一様収束の関係を述べたもので、次の主張になります。

> 有限測度空間$(X,\mathcal{F},\mu)$上の可測関数列$f_{n}\colon X\to\mathbb{C}$があって、可測関数$f$にほとんど至るところ各点収束する。このとき、任意の$\varepsilon>0$に対して$\mu(A)<\varepsilon$なるある可測集合$A\subset X$が存在して、$X \setminus A$上$f_{n}$は$f$に一様収束する。

## 証明
$f_{n}$が$f$にほとんど至るところ各点収束することを集合を用いて表現します。
自然数$n,k$に対して集合$E_{n,k}$を次のように定義します。
$$
E_{n,k}=\bigcap_{l=n}^{\infty} \left\\{ x\in X \mid |f_{l}(x)-f(x)| > \frac{1}{k} \right\\}
$$
これが可測集合であることは明らかです。
このとき、$n\to\infty$で
$$
\lim_{n\to\infty}\mu\left(E_{n,k}\right)=0
$$

## 有限でない場合の反例