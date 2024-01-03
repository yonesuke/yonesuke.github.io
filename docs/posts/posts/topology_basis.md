---
title: "位相の基底"
date: 2023-02-17
draft: false
math: true
authors:
    - yonesuke
---

ホモロジーゼミの中で位相の基底に関する議論が出てきたのでそれについてまとめます。
この記事は大部分が松坂の集合位相入門によっています。

## 定義

### 位相の生成

考える空間を$S$とし、その部分集合全体を集めた集合を$B(S)$で定めます。
$B(S)$の任意の部分集合$M$に対して$O(M)$を$O\supset M$であるような任意の位相に対する下限(すなわちそのような位相たちの共通部分)で定めます。

$O(M)$が具体的にどのような$S$の部分集合で表現されるかを決めるのが次の定理です。

{{< thmlike type="Theorem" >}}
$M$を$B(S)$の任意の部分集合とする。このとき、
$$
O(M)=\left\\{\bigcup_{\lambda\in\Lambda} B_{\lambda} \mathrel{}\middle|\mathrel{}  B_{\lambda}\in M_{0}\right\\}
$$
である。ただし、
$$
M_{0}=\left\\{ \bigcap_{i\in I} A_{i} \mathrel{}\middle|\mathrel{}  A_{i}\in M, \\# I<\infty\right\\}
$$
である。
{{< /thmlike >}}

ここで、$\\#I=0$であれば$\bigcap_{i\in I}A_{i}=S$で定めます。積の単位元を全体にしているようなものです。

### 位相の基底

{{< thmlike type="Definition" title="位相の基底">}}
$M$が位相$O$の部分集合で$O$の任意の元$A$が
$$
A=\bigcup_{\lambda\in\Lambda} W_{\lambda},\quad W_{\lambda}\in M
$$
と表されるとき、$M$は$O$の**基底**であるという。
{{< /thmlike >}}

すなわち、任意の開集合が$M$の元を用いて*展開*できる、という意味で基底になっている、ということです。

## 性質

{{< thmlike type="Theorem">}}
$O$を$S$における1つの位相とする。$M$が$O$の基底であることと、任意の$A\in O$と任意の$x\in A$に対して、
$$
x\in W,\quad W\subset A
$$
となる$W\in M$が存在することは同値である。
{{< /thmlike >}}

{{< proof >}}
- $M$が$O$の基底であるとき、$A$を$O$の任意の元とすると、基底の定義からある$W_{\lambda}\in M$が存在して、
$$
A=\bigcup_{\lambda\in\Lambda} W_{\lambda}
$$
が成り立つ。このとき、$x\in A$なので、ある$\lambda^{\ast}\in \Lambda$が存在して$x\in W_{\lambda^{\ast}}$となるので示される。
- 逆に任意の$A\in O$と任意の$x\in A$に対して$x\in W_{x},W_{x}\subset A$なる$W_{x}\in M$が存在するとき、
$$
A=\bigcup_{x\in A}\\{x\\}=\bigcup_{x\in A} W_{x}
$$
となるのでたしかに$M$は$O$の基底である。
{{< /proof >}}

{{< thmlike type="Theorem">}}
空でない集合$S$について、$B(S)$の部分集合$M$が$O(M)$の基底であることは次の2つと同値である。
1. 任意の$x\in S$に対してある$W\in M$が存在して$x\in W$となる。
2. 任意の$W_{1},W_{2}\in M$で$W_{1}\cap W_{2}\ne \emptyset$であるとき、任意の$x\in W_{1}\cap W_{2}$に対して、ある$W\subset W_{1} \cap W_{2}$が存在して$x\in W$なる$W\in M$が存在する。
{{< /thmlike >}}

{{< proof >}}
- $M$が$O(M)$の基底であるとき、1,2がそれぞれ成り立つことを示す。
    1. $S\in O(M)$に対して、ある$W_{\lambda}\in M$が存在して、$S=\bigcup_{\lambda\in\Lambda}W_{\lambda}$が成り立つ。
    よって任意の$x\in S$に対して、ある$\lambda^{\ast}\in \Lambda$が存在して$x\in W_{\lambda^{\ast}}$となるので示される。
    2. $W_{1},W_{2}\in M$であれば$W_{1},W_{2}\in O(M)$である。このとき、$W_{1}\cap W_{2}\in O(M)$であるから、ある$W_{\lambda}\in M$が存在して、$W_{1}\cap W_{2}=\bigcup_{\lambda\in\Lambda}W_{\lambda}$が成り立つ。このとき、任意の$x\in W_{1}\cap W_{2}$に対して、ある$\lambda^{\ast}\in \Lambda$が存在して$x\in W_{\lambda^{\ast}}$となるので示される。
- 逆に1,2が成り立つとき、$M$が$O(M)$の基底であることを示す。$O(M)$の任意の元$A$はTheorem 1から、
$$
A=\bigcup_{\lambda\in\Lambda} B_{\lambda},\quad B_{\lambda}=\bigcap_{i\in I_{\lambda}}W_{i}^{(\lambda)}
$$
で定められる。よって、$M$が$O(M)$の基底であるためには$B=\bigcap_{i\in I}W_{i}$が$M$の元の和集合で定められることを示せれば良い。

    - $\\#I\geq 2$のときについて考える。2の条件から、任意の$x\in W_{1}\cap W_{2}$に対してある$W_{x}\subset W_{1}\cap W_{2}$が存在して$x\in W_{x}$である。
    よって、$W_{1}\cap W_{2}=\bigcup_{x\in W_{1}\cap W_{2}} W_{x}$で表される。
    同様にして、要素2以上の有限添字集合$I$について$W_{i}(i\in I)$で共通部分が存在するときについても、$B=\bigcap_{i\in I}W_{i}=\bigcup_{x\in\bigcap_{i\in I}W_{i}}W_{x}$で表現できる。
    - $\\#I=1$のとき、ある$W\in M$が存在して$B=W$で表現される。
    - $\\#I=0$のとき、$B=S$である。1の条件から、任意の$x\in S$に対してある$W_{x}\in M$が存在して$x\in W_{x}$である。よって、$B=S=\bigcup_{x\in S}W_{x}$で表される。

    よって、いずれの場合も$B$は$M$の元の和集合で定められるので$M$は$O(M)$の基底である。
{{< /proof >}}