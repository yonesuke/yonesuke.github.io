---
title: "特異値分解"
date: 2021-05-31
draft: false
math: true
authors:
    - yonesuke
---

行列$A$を$m\times n$の実行列とします。
このときある直交行列$U\in\mathbb{R}^{m\times m},V\in\mathbb{R}^{n\times n}$が存在して、
$$U^{\mathsf{T}}AV=\Sigma=\begin{pmatrix}\mathrm{diag}(\sigma_{1},\dots,\sigma_{r}) & O_{r\times(n-r)} \\\\ O_{(m-r)\times r} & O_{(m-r)\times (n-r)}\end{pmatrix}\in\mathbb{R}^{m\times n}$$
となるようにできます。このような分解を特異値分解と言います。

<!-- more -->

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
\\|A-A_{k}\\|\_{F}\leq\\|A-B\\|\_{F}
$$
となります。これをEckart–Young–Mirskyの定理と言います。Frobeniusノルムの他にも2ノルムでもこの定理が成り立つそうです。

### 特異値に関する関係式
特異値の性質を少しまとめておきます。
$\sigma_{i}(M)$を行列$M$の$i$番目に大きい特異値とします。
また、ランクが$r$のときに、$i>r$であれば$\sigma_{i}(M)=0$としておきます。

- 最大特異値について$\sigma_{1}(M)=\max_{\\|x\\|=1}\\| Mx\\|$です。
- $\sigma_{i}(M)=\sigma_{1}(M-M_{i-1})$です。
- $\sigma_{1}(A+B)\leq\sigma_{1}(A)+\sigma_{1}(B)$です。三角不等式からわかります。
- $i,j\in\mathbb{N},i+j-1\leq n$について$\sigma_{i}(A)+\sigma_{j}(B)\geq\sigma_{i+j-1}(A+B)$です。
$$
\begin{aligned}
\sigma_{i}(A)+\sigma_{j}(B)=&\sigma_{1}(A-A_{i-1})+\sigma_{1}(B-B_{j-1})\\\\\geq&\sigma_{1}(A+B-(A_{i-1}+B_{j-1}))\\\\\geq&\sigma_{1}(A+B-(A+B)\_{i+j-2})=\sigma_{i+j-1}(A+B)
\end{aligned}
$$
ここで$\mathrm{rank}(A_{i-1}+B_{j-1})\leq i+j-2=\mathrm{rank}((A+B)_{i+j-2})$を用いました。

### 証明
まず$\\|A-A_{k}\\|\_{F}^{2}=\sum_{i=k+1}^{r}\sigma_{i}^{2}$です。
また、$i>k$で$\sigma_{i}(B)=0$より
$$
\sigma_{i+k}(A)=\sigma_{i+(k+1)-1}((A-B)+B)\leq\sigma_{i}(A-B)+\sigma_{k+1}(B)=\sigma_{i}(A-B)
$$
がわかるので、
$$
\begin{aligned}
\\|A-B\\|^{2}\_{F}=&\sum_{i=1}^{n}\sigma_{i}^{2}(A-B)\geq\sum_{i=1}^{r-k}\sigma_{i}^{2}(A-B)\\\\\geq&\sum_{i=1}^{r-k}\sigma_{i+k}^{2}(A)=\sum_{i=k+1}^{r}\sigma_{i}^{2}(A)=\\|A-A_{k}\\|_{F}^{2}
\end{aligned}
$$
となり、示されました。

## 実装
実装の流れは$A^{\mathsf{T}}A$の固有値と固有ベクトルを計算したあと、そこから$U,\Sigma,V$を求めるだけです。
今回はこの画像を使ってみましょう。

{{< figure src="https://image.ganref.jp/photos/members/takudashin20120925/49106026da62b707f8d87e24fe34b9e3_3.jpg" width=700 >}}

コードはこんな感じです。

{{< gist yonesuke 81753f5b58643b2ff508f8daf66e3044 >}}

これをもとにrankが1,2,3,10,30,50の場合のグラフをプロットすると次のようになります。

{{< figure src="rank.png" width=700 >}}

rankが50の段階でもかなり元の画像に近づいてるのが確認できます。