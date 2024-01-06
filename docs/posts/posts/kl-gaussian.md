---
title: "多変量正規分布間のKL距離"
date: 2021-04-19
slug: kl_gaussian
draft: false
math: true
authors:
    - yonesuke
categories:
    - Mathematics
    - Probability
---

# 多変量正規分布間のKL距離

確率分布の間の"近さ"を測る代表的なものとしてKL距離(Kullback–Leibler divergence)があります。特に多変量正規分布間のKL距離は変分下界を計算する際に登場することもあったりして応用上も重要です。その導出を行います。

<!-- more -->

## 準備
平均$\mu$、分散共分散行列$\Sigma$の$N$次元正規分布$\mathcal{N}(u\mid\mu,\Sigma)$の確率密度関数は
$$
p(u)=\frac{1}{\sqrt{2\pi}^{N}\sqrt{| \Sigma |}}\exp\left[-\frac{1}{2}(u-\mu)^{\mathsf{T}}\Sigma^{-1}(u-\mu)\right]
$$
で与えられます。
また、確率分布$q$に対する確率分布$p$のKL距離は
$$
\text{KL}[p \|| q]=\int p(u)\log\frac{p(u)}{q(u)}du
$$
で定義されます。

## 導出
2つの$N$次元正規分布$p(u)=\mathcal{N}(u \mid \mu _ {1},\Sigma _ {1}),q(u)=\mathcal{N}(u \mid \mu _ {2},\Sigma _ {2})$に対して$\text{KL}[p \|| q]$を計算します。
$\log$部分を展開すると、
$$
\log\frac{p(u)}{q(u)}=\frac{1}{2}\log\frac{|\Sigma _ {2}|}{|\Sigma _ {1}|} -\frac{1}{2}(u-\mu _ {1})^{\mathsf{T}}\Sigma _ {1}^{-1}(u-\mu _ {1}) + \frac{1}{2}(u-\mu _ {2})^{\mathsf{T}}\Sigma _ {2}^{-1}(u-\mu _ {2})
$$
と3つにわかれるので
$$ \begin{aligned} \text{KL}[p \|| q] = &\frac{1}{2}\log\frac{|\Sigma _ {2}|}{|\Sigma _ {1}|}\int_{\mathbb{R}^{N}}p(u)du \\\\ & - \frac{1}{2}\int_{\mathbb{R}^{N}}(u-\mu _ {1})^{\mathsf{T}}\Sigma _ {1}^{-1}(u-\mu _ {1})p(u)du \\\\ & + \frac{1}{2}\int_{\mathbb{R}^{N}}(u-\mu _ {2})^{\mathsf{T}}\Sigma _ {2}^{-1}(u-\mu _ {2}) p(u)du \end{aligned} $$
とできます。各行を計算していきます。

### 一行目
これは簡単で
$$
\frac{1}{2}\log\frac{|\Sigma _ {2}|}{|\Sigma _ {1}|}\int_{\mathbb{R}^{N}}p(u)du=\frac{1}{2}\log\frac{|\Sigma _ {2}|}{|\Sigma _ {1}|}
$$
となります。

### 二行目
定数項を積分の外に出すと、
$$
-\frac{1}{2\sqrt{2\pi}^{N}\sqrt{| \Sigma _ {1}|}}\int _ {\mathbb{R}^{N}}\left[(u-\mu _ {1})^{\mathsf{T}}\Sigma _ {1}^{-1}(u-\mu _ {1})\right]\exp\left[-\frac{1}{2}(u-\mu _ {1})^{\mathsf{T}}\Sigma _ {1}^{-1}(u-\mu _ {1})\right]du
$$
となります。$v=u-\mu _ {1}$に平行移動すると、
$$
-\frac{1}{2\sqrt{2\pi}^{N}\sqrt{|\Sigma_{1}|}}\int _ {\mathbb{R}^{N}}\left(v^{\mathsf{T}}\Sigma _ {1}^{-1}v\right)\exp\left(-\frac{1}{2}v^{\mathsf{T}}\Sigma _ {1}^{-1}v\right)dv
$$
とかんたんにできて、更に$\Sigma _ {1}^{-1/2}v=w$と変数変換すると、
$$
\begin{aligned}
&-\frac{1}{2\sqrt{2\pi}^{N}\sqrt{| \Sigma _ {1}|}}\int _ {\mathbb{R}^{N}}w^{\mathsf{T}}w \text{e}^{-w^{\mathsf{T}}w/2}\sqrt{|\Sigma _ {1}|} dw \\\\ = &-\frac{1}{2\sqrt{2\pi}^{N}}\sum_{i=1}^{N}\int _ {\mathbb{R}^{N}}w _ {i}^{2}\text{e}^{-(w _ {1}^{2}+\dots+w _ {N})/2}dw _ {1}\dots dw _ {N}
\end{aligned}
$$
と展開できます。対称性から各$i$に対して値は同じなので、
$$
\begin{aligned}
-\frac{N}{2\sqrt{2\pi}^{N}}\int _ {\mathbb{R}^{N}}w_{1}^{2} \text{e}^{-(w_{1}^{2}+\dots+w_{N}^{2})/2}dw_{1}\cdots dw_{N}
\end{aligned}
$$
とできます。更に$w_{1}$とそれ以外について積分を分けると、
$$
\begin{aligned}
-\frac{N}{2\sqrt{2\pi}^{N}}\left(\int_{\mathbb{R}}w_{1}^{2}\text{e}^{-w_{1}^{2}/2}dw_{1}\right)\left(\int_{\mathbb{R}}\text{e}^{-w^{2}/2}dw\right)^{N-1}=-\frac{N}{2}
\end{aligned}
$$
とできます。

### 三行目
定数項を積分の外に出すと、
$$
\frac{1}{2\sqrt{2\pi}^{N}\sqrt{| \Sigma _ {1}|}}\int _ {\mathbb{R}^{N}}\left[(u-\mu _ {2})^{\mathsf{T}}\Sigma _ {2}^{-1}(u-\mu _ {2})\right]\exp\left[-\frac{1}{2}(u-\mu _ {1})^{\mathsf{T}}\Sigma _ {1}^{-1}(u-\mu _ {1})\right]du
$$
となります。$v=u-\mu_{1}$に平行移動すると、
$$
\begin{aligned}
&\frac{1}{2\sqrt{2\pi}^{N}\sqrt{|\Sigma_{1}|}}\int_{\mathbb{R}^{N}}\left[(v-(\mu_{2}-\mu_{1}))^{\mathsf{T}}\Sigma_{2}^{-1}(v-(\mu_{2}-\mu_{1}))\right]\exp\left(-\frac{1}{2}v^{\mathsf{T}}\Sigma _ {1}^{-1}v\right)dv\\\\ =&\frac{1}{2\sqrt{2\pi}^{N}\sqrt{|\Sigma_{1}|}}\int _ {\mathbb{R}^{N}}\left(v^{\mathsf{T}}\Sigma_{2}^{-1}v\right)\exp\left(-\frac{1}{2}v^{\mathsf{T}}\Sigma_{1}^{-1}v\right)dv\\\\&-\frac{1}{\sqrt{2\pi}^{N}\sqrt{|\Sigma_{1}|}}\int _ {\mathbb{R}^{N}}\left((\mu_{2}-\mu_{1})^{\mathsf{T}}\Sigma_{2}^{-1}v\right)\exp\left(-\frac{1}{2}v^{\mathsf{T}}\Sigma_{1}^{-1}v\right)dv\\\\&+\frac{(\mu_{2}-\mu_{1})^{\mathsf{T}}\Sigma_{2}^{-1}(\mu_{2}-\mu_{1})}{2\sqrt{2\pi}^{N}\sqrt{|\Sigma_{1}|}}\int_{\mathbb{R}^{N}}\exp\left(-\frac{1}{2}v^{\mathsf{T}}\Sigma _ {1}^{-1}v\right)dv
\end{aligned}
$$
とできます。
#### 一行目
$\Sigma _ {1}^{-1/2}v=w$と変数変換すると、
$$
\frac{1}{2\sqrt{2\pi}^{N}}\int_{\mathbb{R}^{N}}\left(w^{\mathsf{T}}\Sigma_{1}^{1/2}\Sigma_{2}^{-1}\Sigma_{1}^{1/2}w\right)\text{e}^{-w^{\mathsf{T}}w/2}dw
$$
となります。ここで$i\ne j$のとき、
$$
\int_{\mathbb{R}^{N}}w_{i}w_{j}\text{e}^{-w^{\mathsf{T}}w/2}dw=0
$$
であるから、上の計算は
$$
\begin{aligned}
&\frac{1}{2\sqrt{2\pi}^{N}}\sum_{i=1}^{N}\int_{\mathbb{R}^{N}}\left[\Sigma_{1}^{1/2}\Sigma_{2}^{-1}\Sigma_{1}^{1/2}\right]\_{i,i}w_{i}^{2}\text{e}^{-w^{\mathsf{T}}w/2}dw\\\\=&\frac{\text{tr}\left(\Sigma_{1}^{1/2}\Sigma_{2}^{-1}\Sigma_{1}^{1/2}\right)}{2\sqrt{2\pi}^{N}}\int_{\mathbb{R}^{N}}w_{1}^{2}\text{e}^{-w^{\mathsf{T}}w/2}dw
\end{aligned}
$$
と簡単になって、上の二行目の計算を繰り返すと、
$$
\frac{1}{2}\text{tr}\left(\Sigma_{1}\Sigma_{2}^{-1}\right)
$$
が得られます。

#### 二行目
これは$v$に関する奇関数になるので積分は$0$になります。

#### 三行目
これは簡単で
$$
\frac{(\mu_{2}-\mu_{1})^{\mathsf{T}}\Sigma_{2}^{-1}(\mu_{2}-\mu_{1})}{2\sqrt{2\pi}^{N}\sqrt{|\Sigma_{1}|}}\int_{\mathbb{R}^{N}}\exp\left(-\frac{1}{2}v^{\mathsf{T}}\Sigma_{1}^{-1}v\right)dv
=\frac{1}{2}(\mu_{2}-\mu_{1})^{\mathsf{T}}\Sigma_{2}^{-1}(\mu_{2}-\mu_{1})
$$
です。

以上をまとめると、
$$
\text{KL}[p \|| q]=\frac{1}{2}\log\frac{|\Sigma_{2}|}{|\Sigma_{1}|}+\frac{1}{2}\text{tr}\left(\Sigma_{1}\Sigma_{2}^{-1}\right)+\frac{1}{2}(\mu_{2}-\mu_{1})^{\mathsf{T}}\Sigma_{2}^{-1}(\mu_{2}-\mu_{1})-\frac{N}{2}
$$
と求まりました。長い計算でした。