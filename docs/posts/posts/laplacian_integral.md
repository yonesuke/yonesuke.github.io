---
title: "Laplacianの積分表現"
date: 2021-07-30
draft: false
math: true
authors:
    - yonesuke
---

領域$\Omega\subset\mathbb{R}^{n}$上で定義された関数$u\in C^{2}(\Omega)$についてLaplacian(ラプラシアン)は
$$
\Delta u(x)=\sum_{i=1}^{n}\frac{\partial^{2}u}{\partial x_{i}^{2}}(x)
$$
で表されます。このとき、$\partial B(x,r)=\\{y\in\mathbb{R}^{n}\mid |x-y|\=r\\}$とおくと、
$$
\Delta u(x)=\lim_{r\to+0}\frac{2n}{r^{2}|\partial B(x,r)|}\int_{\partial B(x,r)}u(y)-u(x)d\sigma_{y}
$$
が成り立ちます。$d\sigma_{y}$は$\partial B(x,r)$上の面積要素です。
この表現を得るには$u(y)$を$x$まわりでTaylor展開することが大事になるのですが、
その際、平均値の定理によって得られるTaylor展開だと剰余項の評価が難しくなります。
積分型のTaylor展開を用いることでこの問題を解決することができます。

<!-- more -->

## 積分型のTaylor展開
はじめに多重指数$\alpha\in\mathbb{N}\_{\geq 0}^{n}$を導入します。
$$
|\alpha|=\alpha_{1}+\cdots+\alpha_{n},\quad
x^{\alpha}=x_{1}^{\alpha_{1}}\cdots x_{n}^{\alpha_{n}}
$$
とし、多重指数による微分を
$$
D^{\alpha}f=\frac{\partial^{|\alpha|}f}{\partial x_{1}^{\alpha_{1}}\cdots\partial x_{n}^{\alpha_{n}}}
$$
とします。
このとき、関数$f$が$C^{k+1}$級であるとき、点$x$まわりの積分型のTaylor展開は
$$\begin{aligned}&f(y)=\sum_{|\alpha|\leq k}\frac{D^{\alpha}f(x)}{\alpha!}(y-x)^{\alpha}+\sum_{|\beta|=k+1}R_{\beta}(y)(y-x)^{\beta},\\\\&R_{\beta}(y)=\frac{k+1}{\beta!}\int_{0}^{1}(1-t)^{k}D^{\beta}f(x+t(y-x))dt\end{aligned}$$
で与えられます。このように剰余項$R_{\beta}$が積分の形で明示的に与えられるのが特徴です。
授業ではよく平均値の定理を用いた証明を習うと思うのですが、意外にこの積分型のTaylor展開が役に立つことがあるので覚えておいて損はないと思います。

## 証明
$u$を$x$周りで展開しましょう。$u\in C^{2}(\Omega)$なので、
$$\begin{aligned}u(y)-u(x)=&\sum_{i=1}^{n}\frac{\partial u}{\partial x_{i}}(x)(y_{i}-x_{i})+2\sum_{i\ne j}(y_{i}-x_{i})(y_{j}-x_{j})\int_{0}^{1}(1-t)\frac{\partial^{2}u}{\partial x_{i}\partial x_{j}}(x+t(y-x))dt\\\\&+\sum_{i=1}^{n}(y_{i}-x_{i})^{2}\int_{0}^{1}(1-t)\frac{\partial^{2}u}{\partial x_{i}^{2}}(x+t(y-x))dt\end{aligned}$$
です。これを積分すると、
$$\begin{aligned}\int_{\partial B(x,r)}u(y)-u(x)d\sigma_{y}=&\sum_{i=1}^{n}\frac{\partial u}{\partial x_{i}}(x)\int_{\partial B(x,r)}y_{i}-x_{i}d\sigma_{y}\\\\&+2\sum_{i\ne j}\int_{\partial B(x,r)}(y_{i}-x_{i})(y_{j}-x_{j})\left[\int_{0}^{1}(1-t)\frac{\partial^{2}u}{\partial x_{i}\partial x_{j}}(x+t(y-x))dt\right]d\sigma_{y}\\\\&+\sum_{i=1}^{n}\int_{\partial B(x,r)}(y_{i}-x_{i})^{2}\left[\int_{0}^{1}(1-t)\frac{\partial^{2}u}{\partial x_{i}^{2}}(x+t(y-x))dt\right]d\sigma_{y}\end{aligned}$$
となります。
まず、$y_{i}-x_{i}$は$\partial B(x,r)$の奇関数なのでこの積分は$0$となります。
また、$y-x=rz,\ d\sigma_{y}=r^{n-1}d\sigma_{z}$と変数変換すると、$|\partial B(x,r)|=r^{n-1}|\partial B(0,1)|$より、
$$
\begin{aligned}&\frac{2n}{r^{2}|\partial B(x,r)|}\int_{\partial B(x,r)}(y_{i}-x_{i})(y_{j}-x_{j})\left[\int_{0}^{1}(1-t)\frac{\partial^{2}u}{\partial x_{i}\partial x_{j}}(x+t(y-x))\right]d\sigma_{y}\\\\=&\frac{2n}{|\partial B(0,1)|}\int_{\partial B(0,1)}z_{i}z_{j}\left[\int_{0}^{1}(1-t)\frac{\partial^{2}u}{\partial x_{i}\partial x_{j}}(x+trz)\right]d\sigma_{z}\\\\\to&\frac{2n}{|\partial B(0,1)|}\int_{\partial B(0,1)}z_{i}z_{j}\left[\int_{0}^{1}(1-t)\frac{\partial^{2}u}{\partial x_{i}\partial x_{j}}(x)\right]d\sigma_{z}=\frac{n}{|\partial B(0,1)|}\frac{\partial^{2}u}{\partial x_{i}\partial x_{j}}(x)\int_{\partial B(0,1)}z_{i}z_{j}d\sigma_{z}\end{aligned}
$$
と$r\to+0$の極限で求まります。ただし極限操作の交換は優収束定理から正当化されます。
さらに、$i\ne j$ならば$z_{i}z_{j}$の積分は$0$になり、$i=j$ならば、$z_{i}^{2}$の積分は$i$に依存しないので、
$$
\frac{n}{|\partial B(0,1)|}\frac{\partial^{2}u}{\partial x_{i}^{2}}(x)\int_{\partial B(0,1)}z_{i}^{2}d\sigma_{z}=\frac{1}{|\partial B(0,1)|}\frac{\partial^{2}u}{\partial x_{i}^{2}}(x)\int_{\partial B(0,1)}\sum_{i=1}^{n}z_{i}^{2}d\sigma_{z}=\frac{\partial^{2}u}{\partial x_{i}^{2}}(x)
$$
と計算できます。よって、
$$
\frac{2n}{r^{2}|\partial B(x,r)|}\int_{\partial B(x,r)}u(y)-u(x)d\sigma_{y}\to\sum_{i=1}^{n}\frac{\partial^{2}u}{\partial x_{i}^{2}}(x)=\Delta u(x)
$$
となり、示されました。