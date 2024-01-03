---
title: "安定分布に従うノイズの生成方法"
date: 2021-04-21
draft: false
math: true
authors:
    - yonesuke
---

安定分布に従うノイズの生成方法について簡単にまとめておきます。
ここでは代表的なWeronの方法を用います。

<!-- more -->

## Weronの方法
はじめに区間$(-\pi/2,\pi/2)$上の一様分布から乱数$V$を、平均$1$の指数分布に従う乱数$W$をそれぞれ生成します。
- $\alpha\ne1$の場合、
    $$
    \begin{aligned}
    X=S_{\alpha,\beta}\times\frac{\sin(\alpha(V+B_{\alpha,\beta}))}{(\cos V)^{1/\alpha}}\times\left(\frac{\cos(V-\alpha(V+B_{\alpha,\beta}))}{W}\right)^{(1-\alpha)/\alpha}
    \end{aligned}
    $$
    を計算します。ここで$B_{\alpha,\beta},S_{\alpha,\beta}$はそれぞれ、
    $$\begin{aligned}B_{\alpha,\beta}&=\frac{\arctan(\beta\tan(\pi\alpha/2))}{\alpha},\\\\S_{\alpha,\beta}&=\left(1+\beta^{2}\tan^{2}(\pi\alpha/2)\right)^{1/(2\alpha)}\end{aligned}$$
    です。

- $\alpha=1$の場合には、
    $$
    X = \frac{2}{\pi}\left[(\frac{\pi}{2}+\beta V)\tan V - \beta\log\left(\frac{\frac{\pi}{2}W\cos V}{\frac{\pi}{2}+\beta V}\right)\right]
    $$
    を計算します。

これより、$X\sim S(\alpha,\beta,1,0)$なる安定分布ノイズが得られます。
より一般に$S(\alpha,\beta,\gamma,\delta)$に従うノイズがほしければ$\gamma^{1/\alpha} X+\delta$とすれば良いです。

## 補足
$\beta=0$の場合には$S(\alpha,0,1,0)$に従うノイズの生成式は一つにまとめられて、
$$
X=\frac{\sin(\alpha V)}{(\cos V)^{1/\alpha}}\times\left(\frac{\cos((1-\alpha)V)}{W}\right)^{(1-\alpha)/\alpha}
$$
とすればよいです。

この式において更に$\alpha=1$にすれば$X=\tan V$となり、これは平均$0$、$\gamma=1$のCauchy分布に従うノイズの生成方法に一致します。

また、$\alpha=2$の場合には$P,Q\sim U(0,1)$として
$$
X=-2\cos\pi P\sqrt{-\log Q}
$$
となり、これは平均$0$、分散$2$の正規分布に従うノイズの生成方法であるBox–Muller法に一致します。