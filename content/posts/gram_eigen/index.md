---
title: "Gram行列の固有値の数値計算"
date: 2021-08-02T16:59:14+09:00
draft: false
---

カーネル関数$k(\cdot,\cdot)$が与えられたとき、データ点$\\{x_{i}\\}_{i=1}^{n}$に対するGram行列(グラム行列)は
$$
K=\begin{pmatrix}k(x_{1},x_{1}) & \cdots & k(x_{1},x_{n})\\\\\vdots & \ddots & \vdots\\\\ k(x_{n},x_{1}) & \cdots & k(x_{n},x_{n})\end{pmatrix}
$$
で与えられます。色々な場面に登場するのですが、RBFカーネルからガウス過程を生成する際にその固有値計算で詰まったところがあったのでかんたんにまとめておきます。

## Gram行列の性質
カーネル関数の定義は対称性$k(x,y)=k(y,x)$が成り立ってグラム行列が(半)正定値行列になることです。すなわち、任意のベクトル$\bm{c}$に対して、
$$
\bm{c}^{\top}K\bm{c}\geq0
$$
が成り立つことです。なので、カーネル関数から生成されたGram行列は常に半正定値行列になります。
半正定値性と固有値が非負であることは同値なのでGram行列の固有値は常に非負です。

次にガウス過程から関数をサンプルすることを考えましょう。$f\sim\mathcal{GP}(m,k)$について、特にかんたんのために平均を$0$としておきましょう。

## 固有値計算

{{< figure src="out.png">}}

## 原因と対応策