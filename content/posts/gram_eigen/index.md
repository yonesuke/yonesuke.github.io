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

## ガウス過程に従う関数の生成
次にガウス過程から関数をサンプルすることを考えましょう。$f\sim\mathcal{GP}(m,k)$について、特にかんたんのために平均を$0$としておきましょう。
このとき、$\\{x_{i}\\}_{i=1}^{n}$上で関数$f$のベクトル$\bm{f}$は
$$
\bm{f}\sim\mathcal{N}(\bm{0},K)
$$
という多次元ガウス分布に従うものになります。これはGram行列を分散共分散行列とするガウス分布で、代表的なサンプル方法は$K$のコレスキー分解行列と$n$次元の独立標準正規分布に従う乱数との行列・ベクトル積を計算することで得られます。


## 固有値分布とガウス過程の関係
RBFカーネルからGram行列を生成し、コレスキー分解することを考えましょう。ここでRBFカーネルは
$$
k(x,y)=\exp\left(-\frac{(x-y)^{2}}{2l^{2}}\right)
$$
で与えられます。Julia実装をしてみると、
```julia
using LinearAlgebra
xs = -2:0.02:2
K = [exp(-(x-y)^2) for x in xs, y in xs]
cholesky(K)
```
となりますが、これを実行すると、
```julia
ERROR: PosDefException: matrix is not positive definite; Cholesky factorization failed.
```
と帰ってきて、Gram行列が正定値でないと言われてしまいます。理論ではGram行列は正定値であるはずなので、これは数値的な誤差に起因していると考えられます。
そこでGram行列の固有値分布を確認してみることにします。この際、RBFカーネルだけでなく周期カーネルとMatérnカーネルについても固有値分布と対応するガウス過程のサンプルをプロットしました。この結果が次のようになります。

{{< figure src="out.png">}}
固有値分布については降順にソートしたものの絶対値をとったものをプロットしています。ガウス過程のサンプル方法については後述します。

RBFカーネルと周期カーネルの固有値分布に着目すると、指数的な減衰の後、途中で$0$を横切って負の値をとっていることが確認できます。これは固有値が非常に小さく、数値的な誤差によって負だと出力してしまったケースだと考えることができます。一方で、Matérnカーネルの固有値分布に着目すると、ベキ的な減衰が起こっており、負の値を取る等の誤差が見られるわけではありません。

次にガウス過程のサンプルに目を向けてみましょう。RBFカーネルと周期カーネルについては非常に滑らかな関数が出力されています。実際、RBFカーネルと周期カーネルから出力される関数は確率$1$で$C^{\infty}$級の関数がサンプルされることが知られています。一方で、Matérnカーネルについてはそこまで滑らかではない関数が出力されており、これは実際Matérnカーネルによりサンプルされる関数は有限回微分可能な関数になることにも一致します。

固有値分布とガウス過程の特徴についてそれぞれ確認しましたが、実はこの間には密接な関係があります。固有値分布の減衰が指数的である場合にはガウス過程から生成される関数は$C^{\infty}$級の滑らかさをもつことが知られています。一方で、減衰がベキ的な場合には関数は有限回微分可能な関数になることがわかっています。これらの事実は上の議論にも一致します。この結果を踏まえると、滑らかな関数を出力するカーネル関数は、固有値分布が指数的に減衰するため、数値的な誤差により負の固有値が生じやすくなります。そのため、コレスキー分解も失敗する可能性があります。以上がRBFカーネルのコレスキー分解が失敗した主な理由です。

## 対応策(?)
コレスキー分解にこだわらなければ、$K=M^{2}$となる対称行列$M$を見つけることで$M$と$n$次元の独立標準正規分布に従う乱数との行列・ベクトル積を計算することにより多次元ガウス分布を生成しても構いません。
実は行列のルートを計算するJulia実装の関数`sqrt`の内部実装では、非常に小さい負の固有値があればそれを$0$に置き換えて計算してくれるそうです。[公式ページ](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#Base.sqrt)によると次のような説明があります。

> If `A` is real-symmetric or Hermitian, its eigendecomposition (eigen) is used to compute the square root. For such matrices, eigenvalues λ that appear to be slightly negative due to roundoff errors are treated as if they were zero More precisely, matrices with all eigenvalues `≥ -rtol*(max |λ|)` are treated as semidefinite (yielding a Hermitian square root), with negative eigenvalues taken to be zero. `rtol` is a keyword argument to `sqrt` (in the Hermitian/real-symmetric case only) that defaults to machine precision scaled by `size(A,1)`.

十分に小さい負の固有値についてはゼロとしてしまっても(数値計算の上では)問題ないので、これを用いた計算をするならば、
```julia
using LinearAlgebra
xs = -2:0.02:2
K = [exp(-(x-y)^2) for x in xs, y in xs]
println(sqrt(K)*randn(length(xs)))
```
で十分だと思います。