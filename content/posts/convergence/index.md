---
title: "関数列の収束について"
date: 2022-04-30T11:29:26+09:00
draft: false
---

Terrence Taoの"[An introduction to measure theory](https://terrytao.files.wordpress.com/2012/12/gsm-126-tao5-measure-book.pdf)"のゼミがつい先日終わりました。演習問題も全部解いたのでかなり力がついたと思います。
演習問題1.5.2で関数列の収束についての問題があり、収束の関係を図式に起こすと良いよ、と書いてあったのでまとめてみました。

$(X,\mathcal{B},\mu)$を測度空間とします。このとき関数列の収束に関して次のような図式があります。

{{< svg >}}convergence.svg{{< /svg >}}

それぞれの収束は次のとおりです。$f_{n}\colon X\to\mathbb{C},f\colon X\to\mathbb{C}$はそれぞれ可測関数であるとします。
- **uniform conv.**: 任意の$\varepsilon>0$に対してある$N$が存在して、$n\geq N$で$|f_{n}(x)-f(x)|<\varepsilon$が任意の$x\in X$で成立するとき、$f_{n}$は$f$に**一様収束**(uniform convergence)すると言います。
- **point-wise conv.**: 任意の$x\in X$で$f_{n}(x)$が$f(x)$に収束するとき、$f_{n}$が$f$に**各点収束**(point-wise convergence)すると言います。より正確に言うと、任意の$x\in X$と$\varepsilon>0$に対してある$N$が存在して、$n\geq N$で$|f_{n}(x)-f(x)|<\varepsilon$が成立するとき各点収束すると言います。
- **point-wise conv. (a.e.)**: (測度$\mu$のもとで)ほとんど至るところの$x\in X$で$f_{n}(x)$が$f(x)$に収束するとき、$f_{n}$は$f$に**概収束**(pointwise almost everywhere convergence)すると言います。
- **$L^{\infty}$ conv.**: 任意の$\varepsilon>0$に対してある$N$が存在して、$n\geq N$で$|f_{n}(x)-f(x)|<\varepsilon$がほとんど至るところの$x\in X$で成立するとき、$f_{n}$は$f$に **$L^{\infty}$収束**($L^{\infty}$ convergence, converge uniformly almost everywhere)すると言います。
- **almost uniform conv.**: 任意の$\varepsilon>0$に対してある可測集合$E\in\mathcal{B}$で$\mu(E)\leq\varepsilon$なるものが存在して、$E^{\mathrm{c}}$上で$f_{n}$が$f$に一様収束するとき、$f_{n}$は$f$に**概一様収束**(almost uniform convergence)すると言います。
- **$L^{1}$ conv.**: $n\to\infty$で$\\|f_{n}-f\\|\_{L^{1}(\mu)}=\int_{X}|f_{n}(x)-f(x)|d\mu\to0$となると、$f_{n}$は$f$に **$L^{1}$収束**($L^{1}$ convergence)すると言います。
- **conv. in measure**: 任意の$\varepsilon>0$に対して$n\to\infty$で$\mu(\\{x\in X\mid|f_{n}(x)-f(x)|\geq\varepsilon\\})\to0$が$0$となるとき、$f_{n}$が$f$に**測度収束**(convergence in measure)すると言います。
- **conv. in distribution**: **分布収束**(convergence in distribution)に関しては確率空間$\mu(X)=1$で定まるものです。$f_{n}\colon X\to\mathbb{R},f\colon X\to\mathbb{R}$をそれぞれ可測関数とします。**累積分布関数**(cumulative distribution function)$F\colon\mathbb{R}\to[0,1]$を$F(\lambda)\coloneqq\mu(\\{x\in X\mid f(x)\leq\lambda\\})$で定義します。$f_{n}$の累積分布関数$F_{n}$が$f$の累積分布関数$F$の連続点において各点収束するとき、$f_{n}$は$f$に分布収束すると言います。

## 収束の関係
上に添付した図式の矢印はそれぞれ次のようになります。

- 一様収束するならば各点収束する。
- 各点収束するならば概収束する。
- 一様収束するならば$L^{\infty}$収束する。
- $L^{\infty}$収束するならば概一様収束する。
- 概一様収束するならば概収束する。
- $L^{1}$収束するならば測度収束する。
- 概一様収束するならば測度収束する。
- 測度収束するならば分布収束する。

また、条件付きで点線の矢印の関係も成り立ちます。

- $L^{\infty}$収束するならば測度$0$の集合を除いたところで一様収束する。
- $\mu(X)<\infty$のもとで概収束するならば概一様収束する。
- 関数列$f_{n}$が優関数で抑えられるとき概収束するならば概一様収束する。
- $\mu(X)<\infty$のもとで$L^{\infty}$収束するならば$L^{1}$収束する。