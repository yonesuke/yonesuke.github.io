---
title: "特異値分解"
date: 2021-05-31T16:22:52+09:00
draft: false
---

行列$A$を$m\times n$の実行列とします。
このときある直交行列$U\in\mathbb{R}^{m\times m},V\in\mathbb{R}^{n\times n}$が存在して、
$$U^{\mathsf{T}}AV=\Sigma=\begin{pmatrix}\mathrm{diag}(\sigma_{1},\dots,\sigma_{r}) & O_{r\times(n-r)} \\\\ O_{(m-r)\times r} & O_{(m-r)\times (n-r)}\end{pmatrix}\in\mathbb{R}^{m\times n}$$
となるようにできます。このような分解を特異値分解と言います。

## 証明
$A^{\mathsf{T}}A$は実対称行列なので固有ベクトル$\{v_{1},\dots,v_{n}\}$と固有値$\{\xi_{1},\dots,\xi_{n}\}$が存在して
$$
A^{\mathsf{T}}Av_{i}=\xi_{i}v_{i},\quad (v_{i},v_{j})=\delta_{ij}
$$
となるようにできます。
また、
$$
\xi_{i}=(v_{i},\xi_{i}v_{i})=(v_{i},A^{\mathsf{T}}Av_{i})=(Av_{i},Av_{i})\geq0
$$
なので固有値は常に0以上です。特に、$i=1,\dots,r$で$\xi_{i}>0$かつ$i>r$で$\xi_{i}=0$となるようにしておきます。
実はこの固有ベクトル$v_{i}$たちが直交行列$V$に対応します。

次に$i=1,2,\dots,r$に対して$u_{i}=Av_{i}/\sqrt{\xi_{i}}$としましょう。すると、
$$
(u_{i}, u_{j})=\frac{1}{\sqrt{\xi_{i}\xi_{j}}}(Av_{i},Av_{j})=\frac{1}{\sqrt{\xi_{i}\xi_{j}}}(v_{i},A^{\mathsf{T}}Av_{j})=\frac{\xi_{j}\delta_{ij}}{\sqrt{\xi_{i}\xi_{j}}}=\delta_{ij}
$$
となり、$u_{i}$たちは互いに直交します。このとき$m-r$個のベクトル$\{u_{r+1},\dots,u_{m}\}$を持ってきて正規直交基底$\{u_{1},\dots,u_{m}\}$を構成することができます。
実はこのベクトル$u_{i}$たちが直交行列$U$に対応します。

上で得られたベクトルを用いて行列
$$
V=(v_{1},\dots,v_{n}),\quad U=(u_{1},\dots,u_{m})
$$
を構成します。これが直交行列なのは明らかです。
$U^{\mathsf{T}}AV$という行列を計算してみると、
$$
U^{\mathsf{T}}AV[i,j]=(u_{i},Av_{j})
$$
となります。
- $j=r+1,\dots,n$においては
    $$
    (Av_{j},Av_{j})=(v_{j},A^{\mathsf{T}}Av_{j})=(v_{j},0)=0
    $$
    なので$(u_{i},Av_{j})=0$となります。
- $i=r+1,\dots,m$かつ$j=1,\dots,r$においても
    $$
    (u_{i},Av_{j})=(u_{i},\sqrt{\xi_{j}}u_{j})=0
    $$
    となります。
- $1\leq i,j\leq r$のときには、
    $$
    (u_{i},Av_{j})=(u_{i},\sqrt{\xi_{j}}u_{j})=\sqrt{\xi_{j}}\delta_{ij}
    $$
    となります。

よって、$\sqrt{\xi_{i}}=\sigma_{i}$と置くことで
$$U^{\mathsf{T}}AV=\Sigma=\begin{pmatrix}\mathrm{diag}(\sigma_{1},\dots,\sigma_{r}) & O_{r\times(n-r)} \\\\ O_{(m-r)\times r} & O_{(m-r)\times (n-r)}\end{pmatrix}$$
となることが示されました。
特に$A=U\Sigma V^{\mathsf{T}}$もわかります。

## rank-$k$近似としてのSVD
行列$A$をSVDしたものはベクトルを用いた表示をするならば、
$$
A=\sum_{i=1}^{r}\sigma_{i}u_{i}v_{i}^{\mathsf{T}}
$$
となります。これを$k\leq r$に対して
$$
A_{k}=\sum_{i=1}^{k}\sigma_{i}u_{i}v_{i}^{\mathsf{T}}
$$
と和を$k$までで打ち切ったものを行列$A$のrank-$k$近似と呼ぶことにします。
実は、ランクが多くても$k$の行列全体の中で、$A_{k}$はFrobeniusノルムのもとで$A$に最も近い行列であることが示されます。
すなわち、ランクが$k$以下の任意の行列$B$に対して、
$$
\\|A-A_{k}\\|_{F}\leq\\|A-B\\|_{F}
$$
となります。これをEckart–Young–Mirskyの定理と言います。

### 補題
行列$A$の各行を$V_{k}=\mathrm{Span}(v_{1},\dots,v_{k})$へ射影したものは$A_{k}$に一致します。

これを確認するために$A$の$i$行目を$a_{i}$と置きます。
横ベクトル$a$に対する$V_{k}$への射影が
$\sum_{i=1}^{k}(a^{\mathsf{T}},v_{i})v_{i}^{\mathsf{T}}$になることを用いると、
$$
\begin{pmatrix}\sum_{j=1}^{k}(a_{1}^{\mathsf{T}},v_{j})v_{j}^{\mathsf{T}} \\\\ \sum_{j=1}^{k}(a_{2}^{\mathsf{T}},v_{j})v_{j}^{\mathsf{T}} \\\\ \vdots \\\\ \sum_{j=1}^{k}(a_{m}^{\mathsf{T}},v_{j})v_{j}^{\mathsf{T}}\end{pmatrix}=\sum_{j=1}^{k}\begin{pmatrix} (a_{1}^{\mathsf{T}},v_{j})v_{j}^{\mathsf{T}}\\\\(a_{2}^{\mathsf{T}},v_{j})v_{j}^{\mathsf{T}} \\\\ \vdots \\\\(a_{m}^{\mathsf{T}},v_{j})v_{j}^{\mathsf{T}}\end{pmatrix}=\sum_{j=1}^{k}Av_{j}v_{j}^{\mathsf{T}}=\sum_{i=1}^{r}\sum_{j=1}^{k}\sigma_{i}u_{i}v_{i}^{\mathsf{T}}v_{j}v_{j}^{\mathsf{T}}=\sum_{i=1}^{k}\sigma_{i}u_{i}v_{i}^{\mathsf{T}}=A_{k}
$$
となり、示されました。

### 証明
$B$の$i$行目を$b_{i}$と置き、$V=\{b_{1},\dots,b_{m}\}$とすると、$\dim V\leq k$となります。
