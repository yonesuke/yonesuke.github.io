---
title: "はめ込みと埋め込み"
date: 2022-08-30
draft: false
math: true
authors:
    - yonesuke
---

松本『[多様体の基礎](http://www.utp.or.jp/book/b302120.html)』とTu『[トゥー多様体](https://www.shokabo.co.jp/mybooks/ISBN978-4-7853-1586-3.htm)』を読みながらはめ込みと埋め込みについて書いてあったことをまとめます。

## はめ込みと埋め込み

{{< thmlike type="Definition" title="はめ込み" >}}
$f\colon M\to N$が**はめ込み**であるとは、任意の$p\in M$における$f$の微分$(df) _ {p}\colon T_{p} M\to T_{f(p)} N$が単射になることである。
{{< /thmlike >}}

ユークリッド空間のもとでのはめ込みは$\mathbb{R}^{m}$からより高い次元$\mathbb{R}^{m}$への包含写像
$$
i\colon\mathbb{R}^{m}\to\mathbb{R}^{n};(x_{1},\dots,x_{m})\mapsto(x_{1},\dots,x_{n},0,\dots,0)
$$
となります。この写像は局所的であり、これを多様体へと拡張したものが次の**はめ込み定理**になります。

{{< thmlike type="Theorem" title="はめ込み定理">}}
$f\colon M\to N$がはめ込みであるとする。任意の点$p\in M$に対してある座標近傍$(U;\phi)$と$f(p)\in N$に対するある座標近傍$(V;\psi)$があって、$\phi(p)$のある近傍で、
$$
(\psi\circ f\circ \phi^{-1})(r_{1},\dots,r_{m})=(r_{1},\dots,r_{n},0,\dots,0)
$$
なるものが存在する。
{{< /thmlike >}}

この意味ですべてのはめ込みは局所的には包含写像になることがわかります。しかし、この場合$f(M)$は後で見る部分多様体にはならないことがあります。この場合$f$に次の条件を与えた**埋め込み**を考えます。

{{< thmlike type="Definition" title="埋め込み" >}}
$f\colon M\to N$が**埋め込み**であるとは、$f$がはめ込みであってかつ、$f(M)$に$N$の相対位相を入れたときに$f\colon M\to f(M)$が同相写像となることである。
{{< /thmlike >}}

ただしはめ込み定理があるので、はめ込みは局所的には埋め込みになります。

{{< thmlike type="Proposition">}}
$C^{r}$級はめ込みは**局所的には**$C^{r}$級埋め込みである。すなわり、任意の点$p\in M$に対して,
$p$のある開近傍$U$が存在して、$f|_{U}\colon U\to N$は埋め込みである。
{{< /thmlike >}}

## 部分多様体
{{< thmlike type="Definition" title="正則部分多様体" >}}
$n$次元$C^{r}$級多様体$N$の部分集合$S$が$k$次元の **$C^{r}$級正則部分多様体**であるとは、任意の点$p\in S$に対して、ある座標近傍$(U;x_{1},\dots,x_{n})$が存在して、
$$
U\cap S=\\{(x_{1},\dots,x_{n})\in U\mid x_{k+1}=\cdots=x_{n}=0\\}
$$
が成り立つことである。
{{< /thmlike >}}

『トゥー多様体』ではこのような$N$のチャートを$S$に**適合するチャート**と呼びます。

{{< thmlike type="Proposition" >}}
$n$次元$C^{r}$級多様体$N$の$k$次元$C^{r}$級正則部分多様体$S$はそれ自身$k$次元$C^{r}$級多様体である。
{{< /thmlike >}}

{{< proof title="Of Proposition" >}}
$(U;\phi)=(U;x_{1},\dots,x_{n}),(V,\psi)=(V;y_{1},\dots,y_{n})$を適合するチャートとし、$U\cap V\ne\emptyset$とする。このとき、$p\in U\cap V\cap S$に対して
$$
\phi(p)=(x_{1}(p),\dots,x_{k}(p),0,\dots,0),\psi(p)=(y_{1}(p),\dots,y_{k}(p),0,\dots,0)
$$
である。そこで、$\phi,\psi$の出力を$k$次元目までのところに制限したものを$\phi_{S},\psi_{S}$と書くことにすると、$\psi_{S}\circ\phi_{S}^{-1},\phi_{S}\circ\psi_{S}^{-1}$はそれぞれ$C^{r}$級の座標変換関数になる。
よって適合するチャートから構成した族$\\{(U\cap S;\phi_{S})\\}$は$S$の$C^{r}$級のアトラスになる。よって$S$は$k$次元$C^{r}$級多様体である。
{{< /proof >}}

次の2つの定理によって多様体の埋め込みと正則部分多様体は同じものとみなすことができます。

{{< thmlike type="Theorem" >}}
$n$次元$C^{r}$級多様体$N$の$l$次元$C^{r}$級部分多様体を$L$とする。$L$から$N$への包含写像$i\colon L\hookrightarrow N$は$C^{r}$級の埋め込みである。
{{< /thmlike >}}

{{< thmlike type="Theorem" >}}
$f\colon M\to N$が$C^{r}$級の埋め込みであれば、その像$f(M)$は$N$の$C^{r}$級正則部分多様体であり、$f\colon M\to f(M)$は$C^{r}$級微分同相写像である。
{{< /thmlike >}}