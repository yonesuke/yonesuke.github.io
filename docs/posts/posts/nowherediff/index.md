---
title: "至るところ微分不可能な連続関数: 初等的な構成方法"
date: 2021-05-30
draft: false
math: true
authors:
    - yonesuke
---

$[-1,1]$上の関数
$$
\varphi(x)=|x|
$$
を考え、
これを$\varphi(x+2)=\varphi(x)$として$\mathbb{R}$上へ拡張します。
このとき、
$$
f(x)=\sum_{n=0}^{\infty}\left(\frac{3}{4}\right)^{n}\varphi(4^{n}x)
$$
は$\mathbb{R}$上の連続関数ですが至るところ微分不可能であることが知られています。
以下でこれを示していきましょう。

<!-- more -->

## 連続であること
連続であることの証明はかんたんです。
$$
f_{k}(x)=\sum_{n=0}^{k}\left(\frac{3}{4}\right)^{n}\varphi(4^{n}x)
$$
とおくと、$f_{k}$は連続関数です。
$$
|f(x)-f_{k}(x)|\leq\sum_{n=k+1}^{\infty}\left(\frac{3}{4}\right)^{n}=4\left(\frac{3}{4}\right)^{k+1}
$$
となるので、$x$によらずに一様に抑えることができます。
これは連続関数$f_{k}$が$f$に一様収束していることを表します。
連続関数の一様収束極限は連続関数なので$f$は連続関数です。

## 微分不可能であること
任意の$x_{0}\in\mathbb{R}$を一つ固定しましょう。
$m\in\mathbb{N}$に対して、$\delta_{m}=\pm\frac{1}{2}4^{-m}$とおき、
符号は$4^{m}x_{0}$と$4^{m}(x_{0}+\delta_{m})$の間に整数が来ないようにします。
$$
\gamma_{n}=\frac{\varphi(4^{n}(x_{0}+\delta_{m}))-\varphi(4^{n}x_{0})}{\delta_{m}}
$$
という値を計算してみましょう。
- $n>m$のときには$4^{n}\delta_{m}$は$2$の倍数となります。
$\varphi(x)$が周期$2$の関数であることを思い出すと、$\gamma_{n}=0$がわかります。
- $n=m$のときには$4^{m}x_{0}$と$4^{m}(x_{0}+\delta_{m})$の間に整数が来ないので、
$|\gamma_{m}|=|4^{m}\delta_{m}/\delta_{m}|=4^{m}$となります。
- $n < m$のときには、一般に$|\varphi(x)-\varphi(y)|\leq|x-y|$なので、
$|\gamma_{n}|\leq4^{n}$がわかります。

以上の準備のもと$f$が微分不可能であることを示しましょう。
具体的には、上で定義した$\delta_{m}$を用いて、$x_{0}+\delta_{m}\to x_{0}$の極限で微分が発散することを確認します。
$$
\left|\frac{f(x_{0}+\delta_{m})-f(x_{0})}{\delta_{m}}\right|
=\left|\sum_{n=0}^{\infty}\left(\frac{3}{4}\right)^{n}\frac{\varphi(4^{n}(x_{0}+\delta_{m}))-\varphi(4^{n}x_{0})}{\delta_{m}}\right|
=\left|\sum_{n=0}^{\infty}\left(\frac{3}{4}\right)^{n}\gamma_{n}\right|
$$
となりますが、$\gamma_{n}$が$n>m$で消えること、また$|x+y|\geq|x|-|y|$であることを用いると、
$$\begin{aligned}\left|\sum_{n=0}^{\infty}\left(\frac{3}{4}\right)^{n}\gamma_{n}\right|=&\left|\sum_{n=0}^{m}\left(\frac{3}{4}\right)^{n}\gamma_{n}\right|\\\\=&\left|\left(\frac{3}{4}\right)^{m}\gamma_{m}+\sum_{n=0}^{m-1}\left(\frac{3}{4}\right)^{n}\gamma_{n}\right|\\\\\geq&\left(\frac{3}{4}\right)^{m}|\gamma_{m}|-\left|\sum_{n=0}^{m-1}\left(\frac{3}{4}\right)^{n}\gamma_{n}\right|\\\\\geq&3^{m}-\sum_{n=0}^{m-1}3^{m}=\frac{1}{2}(3^{m}+1)\to\infty\end{aligned}$$
となり、$\delta_{m}$を用いた点列の収束だと微分が発散することがわかります。
よって$f$が微分不可能であることを示すことができました。

## グラフの概形
$f$自身は無限和で定義されているためプロットすることはできません。
その代わりに$f_{k}$をプロットすることにしました。
そのときのpythonファイルです。

{{< gist yonesuke e4df40545421d82c3f92e7ce097ce687 >}}

$ k = 0,1,3,10 $の場合の$ f _ {k} $をプロットしたものが次のようになります。

{{< figure src="plot.png" width=700 >}}

$f_{0}$は$\varphi$にほかなりません。
$f_{1}$は$\varphi(4x)$によって引き伸ばされたものを足し込むことによって微分不可能な点が新たに増えていることがわかります。
このように$\varphi(4^{m}x)$によって微分不可能点が$\mathbb{R}$全体に伝播していって最終的に$f$が微分不可能になる様子が確認できます。