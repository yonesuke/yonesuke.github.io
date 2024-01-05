---
title: "Brouwer's fixed-point theorem"
date: 2022-08-18
slug: brouwer_fixed_point_theorem
draft: false
math: true
authors:
  - yonesuke
categories:
  - Mathematics
  - Topology
---
ホモロジーゼミの中でブラウワーの不動点定理の証明が出てきました。特に円盤$D^{2}$上でのブラウワーの不動点定理は基本群を用いて簡便に証明ができることを学んだので備忘録としてまとめておきます。

!!! success "ブラウワーの不動点定理"
    $D^{2}\to D^{2}$の任意の連続関数は不動点を持つ。

<!-- more -->

## ホモトピーからの準備

### 誘導準同型
連続関数を$\varphi\colon(X,x_{0})\to(Y,y_{0})$のように書いたとき、$\varphi\colon X\to Y$で$\varphi(x_{0})=y_{0}$なるものとします。位相空間の間の連続関数$\varphi$から誘導される基本群の間の準同型を次のように定めます。
$$
\varphi_{\ast}\colon \pi_{1}(X,x_{0})\to\pi_{1}(Y,y_{0}),\quad\varphi_{\ast}([f])=[\varphi\circ f]
$$
これは$x_{0}$を基点とする$X$内のループ(をホモトピー同値類で割ったもの)を$\varphi$で$y_{0}$を基点とする$Y$内のループ(をホモトピー同値類で割ったもの)に移す写像になります。この写像はwell-definedです。

連続関数の間の合成という操作は誘導準同型を通して準同型写像の間の合成に移ります。すなわち、$\psi\colon(X,x_{0})\to(Y,y_{0}),\varphi\colon(Y,y_{0})\to(Z,z_{0})$に対して、
$$
(\varphi\circ\psi) _ {\ast}=\varphi_{\ast}\circ\psi_{\ast}
$$
になります。これは$(\varphi\circ\psi) _ {\ast}([f])=[(\varphi\circ\psi)\circ f]=[\varphi \circ (\psi \circ f)]=\varphi_{\ast}([\psi \circ f])=\varphi_{\ast}(\psi_{\ast}[f])=\varphi_{\ast}\circ\psi_{\ast}([f])$からわかります。

### レトラクション
$X$から$A\subset X$への**レトラクション**を次のように定義します。

!!! note "レトラクション"
    連続関数$r\colon X\to A$がレトラクションであるとは、任意の$a\in A$で$r(a)=a$となることである。

!!! info
    レトラクションは射影のようなものと見ることができます。線形写像$P$が射影であることは$P^{2}=P\circ P=P$で特徴づけられましたが、同様にレトラクションは$r\circ r=r$となります。


また、レトラクションの対となるものとして**包含写像**が定まります。
$$
i\colon A\hookrightarrow X
$$

すると、$r\circ i=\mathrm{id} _ {A}$になります。誘導準同型からの帰結として、
$$
r_{\ast}\circ i_{\ast}=\mathrm{id} _ {\pi _ {1}(A,x_{0})}
$$
になります。特に、$i_{\ast}\colon\pi_{1}(A,x_{0})\to\pi_{1}(X,x_{0})$は単射準同型になります。

## 証明

!!! abstract "Proof"
    背理法で示す。ある連続関数$h\colon D^{2}\to D^{2}$が存在して、任意の$x\in D^{2}$で$h(x)\ne x$であるとしよう。
    このとき、$r\colon D^{2}\to\mathbb{S}^{1}$のレトラクションを次のように構成できる。

    - $x\in D^{2}$と$h(x)\in D^{2}$に対して、$h(x)$を始点として$x$を通るような半直線を定めることができる。この半直線と$\mathbb{S}^{1}$の交点を$r(x)$とする。

    すると、$x\in\mathbb{S}^{1}$であれば$r(x)=x$になるので、このように構成した$r$は$\mathbb{S}^{1}$へのレトラクションになる。

    このレトラクションの存在から、包含写像$i\colon\mathbb{S}^{1}\to D^{2}$の誘導準同型$i_{\ast}\colon \pi_{1}(\mathbb{S}^{1},x_{0})\to \pi_{1}(D^{2},x_{0})$は単射になる。しかし、$\mathbb{S}^{1}$の基本群は$\mathbb{Z}$と同型であるのに対して、$D^{2}$の基本群は自明群に同型である。よって$i_{\ast}$は単射にはなりえず、矛盾。

    以上より、任意の連続関数$h\colon D^{2}\to D^{2}$は不動点を持つ。
