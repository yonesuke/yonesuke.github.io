---
title: "Egorov's theorem"
date: 2022-03-22T13:05:46+09:00
draft: false
math: true
---

Egorovの定理は関数列の概収束と概一様収束の関係を述べたものになります。

>有限測度空間$(X,\mathcal{F},\mu)$上の可測関数列$f_{n}\colon X\to\mathbb{C}$に対して、
$f_{n}$が可測関数$f$に概収束するならば、$f_{n}$は$f$に概一様収束する。

ここで、
- 関数列$f_{n}$が$f$に**概収束**するとは、ほとんどすべての$x\in X$に対して$f_{n}(x)$が$f(x)$に収束することである。すなわち、$f_{n}$はほとんど至るところ$f$に各点収束する。
- 関数列$f_{n}$が$f$に**概一様収束**するとは、任意の$\varepsilon>0$に対して$\mu(E)<\varepsilon$なる可測集合$E\in\mathcal{F}$が存在して、$E^{\mathrm{c}}$上では$f_{n}$が$f$に一様収束するようにできることである。

ちなみにT. Taoの[An Introduction to Measure Theory](https://bookstore.ams.org/gsm-126)では、概収束のことをpointwise almost everywhere convergence, 概一様収束のことをalmost uniform convergenceと呼んでいます。(Taoの英語の本をずっと英語で読んでいたので対応する日本語を調べる必要がありました。)

## 証明
点$x_{0}$において$f_{n}$が$f$に収束することを集合を用いて表してみましょう。
$$\begin{aligned}&f_{n}(x_{0})\to f(x_{0})\\\\ \Leftrightarrow& \forall m>0, \exists N\in\mathbb{N}, \mathrm{s.t.}\ n\geq N \Rightarrow |f_{n}(x_{0})-f(x_{0})|\leq\frac{1}{m}\\\\ \Leftrightarrow& \forall m>0, \exists N\in\mathbb{N}, \mathrm{s.t.}\ n\geq N \Rightarrow x_{0}\in\left\\{x\in X\mid |f_{n}(x)-f(x)|\leq\frac{1}{m}\right\\}\\\\ \Leftrightarrow& \forall m>0, \exists N\in\mathbb{N}, \mathrm{s.t.}\ x_{0}\in\bigcap_{n\geq N}\left\\{x\in X\mid |f_{n}(x)-f(x)|\leq\frac{1}{m}\right\\}\\\\ \Leftrightarrow& \forall m>0,  x_{0}\in\bigcup_{N\in\mathbb{N}}\bigcap_{n\geq N}\left\\{x\in X\mid |f_{n}(x)-f(x)|\leq\frac{1}{m}\right\\}\\\\ \Leftrightarrow& x_{0}\in\bigcap_{m\in\mathbb{N}}\bigcup_{N\in\mathbb{N}}\bigcap_{n\geq N}\left\\{x\in X\mid |f_{n}(x)-f(x)|\leq\frac{1}{m}\right\\}\\\\\end{aligned}$$
となります。これより、点$x_{0}$において$f_{n}$が$f$に収束しないことは
$$
x_{0}\in\bigcup_{m\in\mathbb{N}}\bigcap_{N\in\mathbb{N}}\bigcup_{n\geq N}\left\\{x\in X\mid |f_{n}(x)-f(x)|>\frac{1}{m}\right\\}
$$
で書くことができます。ここで、
$$
E_{N,m}=\bigcup_{n\geq N} \left\\{ x\in X \mid |f_{n}(x)-f(x)| > \frac{1}{m} \right\\}
$$
を定義すると(これが可測集合であることは明らかです)、収束しない点の測度は0なので$\mu(\bigcup_{m\in\mathbb{N}}\bigcap_{N\in\mathbb{N}}E_{N,m})=0$がわかります。特に測度の劣加法性から、任意の$m\in\mathbb{N}$に対して
$$
\mu\left(\bigcap_{N\in\mathbb{N}}E_{N,m}\right)=0
$$
がわかります。$E_{N,m}$が集合として$N$に関して単調減少であり、しかも$\mu(X)<\infty$であることから、$\lim_{N\to\infty}\mu\left(E_{N,m}\right)=0$とも書けます。これは$m$を決めるたびごとに定まる$N$に関する数列だと思うと、任意の$\varepsilon>0$と任意の$m$に対してある$N_{m}$があって、$\mu\left(E_{N_{m},m}\right)<\varepsilon/2^{m}$となるようにできます。このときに集合$E$を
$$
E=\bigcup_{m\in\mathbb{N}}E_{N_{m},m}
$$
で定めると、$\mu(E)<\varepsilon$は測度の加法性から明らかです。$E^{\mathrm{c}}$は、
$$
E^{\mathrm{c}}=\bigcap_{m\in\mathbb{N}}\bigcap_{n\geq N_{m}}\left\\{ x\in X \mid |f_{n}(x)-f(x)| \leq \frac{1}{m} \right\\}
$$
と書けて、これは$E^{\mathrm{c}}$上では$x$の選び方によらずに関数の差が$1/m$で抑えられていることを意味します。よって、$E^{\mathrm{c}}$上で一様収束することが示せました。


## 有限でない場合の反例
有限でない場合の例として、測度空間$(\mathbb{R}, \mathcal{B}, \mu)$で、$f_{n}=1_{[n,n+1]}$を考えます。この関数が$f=0$に各点収束することは明らかですが、$\mu(E)<1$であれば$E^{\mathrm{c}}$上で$f_{n}$の値が1になるようなものを作ることができてしまうので、この場合には概一様収束にはなりえません。
