---
title: "多変量正規分布間のKL距離"
date: 2021-04-18T01:55:17+09:00
draft: false
---

hogefuga

$1/3$

$$
\frac{d\theta_{i}}{dt}
$$

確率分布の間の"近さ"を測る代表的なものとしてKL距離(Kullback–Leibler divergence)があります。特に多変量正規分布間のKL距離は変分下界を計算する際に登場することもあったりして応用上も重要です。その導出を行います。

# 準備
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

# 導出
2つの$N$次元正規分布$p(u)=\mathcal{N}(u \mid \mu _ {1},\Sigma _ {1}),q(u)=\mathcal{N}(u \mid \mu _ {2},\Sigma _ {2})$に対して$\text{KL}[p \|| q]$を計算します。
$\log$部分を展開すると、
$$
\log\frac{p(u)}{q(u)}=\frac{1}{2}\log\frac{|\Sigma _ {2}|}{|\Sigma _ {2}|} -\frac{1}{2}(u-\mu _ {1})^{\mathsf{T}}\Sigma _ {1}^{-1}(u-\mu _ {1}) + \frac{1}{2}(u-\mu _ {2})^{\mathsf{T}}\Sigma _ {2}^{-1}(u-\mu _ {2})
$$
と3つにわかれるのでそれぞれと$p(u)$との積の積分を計算していきます。

- $\frac{1}{2}\log\frac{|\Sigma _ {2}|}{|\Sigma _ {2}|}$

これは$u$に依存しない定数なので、
$$
\frac{1}{2}\log\frac{|\Sigma _ {2}|}{|\Sigma _ {2}|}\int_{\mathbb{R}^{N}}p(u)du=\frac{1}{2}\log\frac{|\Sigma _ {2}|}{|\Sigma _ {2}|}
$$
となります。

- $\frac{1}{2}(u-\mu _ {1})^{\mathsf{T}}\Sigma _ {1}^{-1}(u-\mu _ {1})$

定数項を積分の外に出すと、
$$
\frac{1}{2\sqrt{2\pi}^{N}\sqrt{| \Sigma _ {1}|}}\int _ {\mathbb{R}^{N}}\left[(u-\mu _ {1})^{\mathsf{T}}\Sigma _ {1}^{-1}(u-\mu _ {1})\right]\exp\left[-\frac{1}{2}(u-\mu _ {1})^{\mathsf{T}}\Sigma _ {1}^{-1}(u-\mu _ {1})\right]du
$$
となります。$v=u-\mu _ {1}$に平行移動すると、
$$
\frac{1}{2\sqrt{2\pi}^{N}\sqrt{|\Sigma_{1}|}}\int _ {\mathbb{R}^{N}}\left(v^{\mathsf{T}}\Sigma _ {1}^{-1}v\right)\exp\left(-\frac{1}{2}v^{\mathsf{T}}\Sigma _ {1}^{-1}v\right)dv
$$
とかんたんにできて、更に$\Sigma _ {1}^{-1/2}v=w$と変数変換すると、
$$\begin{aligned}
 &\frac{1}{2\sqrt{2\pi}^{N}\sqrt{| \Sigma _ {1}|}}\int _ {\mathbb{R}^{N}}w^{\mathsf{T}}w \text{e}^{-w^{\mathsf{T}}w/2}\sqrt{|\Sigma _ {1}|} dw \\\\ = &\frac{N}{2\sqrt{2\pi}^{N}}\int _ {\mathbb{R}^{N}}w _ {1}^{2}\text{e}^{-(w _ {1}^{2}+\dots+w _ {N})/2}dw _ {1}\dots dw _ {N} \\\\= &\frac{N}{2\sqrt{2\pi}^{N}}\int _ {\mathbb{R}}w _ {1}^{2}\text{e}^{-w _ {1}^{2}/2}dw _ {1}\left(\int _ {\mathbb{R}}\right)^{N-1}\end{aligned}$$
と展開できます。対称性から$w _ {1}$のみの計算に押し付けて$N$倍しています。各$i,j$によらずに積分の値は

- $\frac{1}{2}(u-\mu _ {2})^{\mathsf{T}}\Sigma _ {2}^{-1}(u-\mu _ {2})$

$$\begin{aligned} a &= b+c\\\\ &=d+e \end{aligned}$$

$$ \begin{aligned} f(x) & = x^2+2x+1 \\\\ & = (x+1)^2 \end{aligned} $$