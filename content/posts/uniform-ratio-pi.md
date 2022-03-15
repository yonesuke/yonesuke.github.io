---
title: "Uniform ratio distribution and pi"
date: 2022-03-15T13:05:46+09:00
draft: false
---

3月14日は円周率の日ということもあって次のツイートを見つけました。

{{<tweet 1503307755976765443 >}}

この証明を行っていきます。

## Uniform ratio distribution
$[0,1]$区間の一様乱数$X,Y$に対して$X/Y$が取る分布はuniform ratio distributionという名前がついています。この分布は手で計算することができて、
$$
\begin{aligned}
P_{X/Y}(u)\coloneqq& \int_{0}^{1}\int_{0}^{1}\delta\left(\frac{x}{y}-u\right)dxdy\\\\
=&\begin{cases}
1/2, & 0<u<1\\\\
1/(2u^{2}), & u\geq1
\end{cases}
\end{aligned}
$$
となることが知られています。

参考: https://mathworld.wolfram.com/UniformRatioDistribution.html

### 証明
- $0<u<1$ のとき、$\delta\left(\frac{x}{y}-u\right)$が$x$の関数だと思うと、
$$
P_{X/Y}(u)=\int_{0}^{1}\int_{0}^{1}y\delta(x-uy)dxdy
$$
になります。ある$x$で$x-uy=0$となるのでとなるので、これを$x$で積分することで、
$$
P_{X/Y}(u)=\int_{0}^{1}ydy=\frac{1}{2}
$$
が得られます。
- $u\geq1$のとき、$\delta\left(\frac{x}{y}-u\right)$が$y$の関数だと思うと、
$$
P_{X/Y}(u)=\int_{0}^{1}\int_{0}^{1}\frac{x}{u^{2}}\delta\left(y-\frac{x}{u}\right)dydx
$$
になります。ある$y$で$y-x/u=0$となるので、これを$y$で積分することで、
$$
P_{X/Y}(u)=\int_{0}^{1}\frac{x}{u^{2}}dx=\frac{1}{2u^{2}}
$$
が得られます。

## Calculating probability
求めたい確率は、$X/Y$に最も近い整数が偶数となる事象の確率です。
これは、$X/Y$の取りうる値が次の集合$A$に含まれれば良いことになります。
$$
A\coloneqq\left[0,\frac{1}{2}\right]\cup\bigcup_{n=1}^{\infty}\left[2n-\frac{1}{2},2n+\frac{1}{2}\right]
$$
よって求める確率は、
$$
\begin{aligned}
\int_{A}P_{X/Y}(u)du=&\int_{0}^{1/2}\frac{1}{2}du+\sum_{n=1}^{\infty}\int_{2n-1/2}^{2n+1/2}\frac{1}{2u^{2}}du\\\\
=&\frac{1}{4}+\sum_{n=1}^{\infty}\left(\frac{1}{4n-1}-\frac{1}{4n+1}\right)\\\\
=&\frac{1}{4}+\left(\frac{1}{3}-\frac{1}{5}+\frac{1}{7}-\frac{1}{9}+\cdots\right)
\end{aligned}
$$
となります。この無限和はライプニッツの公式を思い出すと、
$$
1-\frac{1}{3}+\frac{1}{5}-\frac{1}{7}+\frac{1}{9}-\cdots=\frac{\pi}{4}
$$
となので、これを代入すると、
$$
\int_{A}P_{X/Y}(u)du=\frac{1}{4}+1-\frac{\pi}{4}=\frac{5-\pi}{4}
$$
と求まりました！！！
