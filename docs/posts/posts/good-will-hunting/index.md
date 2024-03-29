---
title: "Good Will Hunting Problem"
date: 2021-05-31
slug: good_will_hunting
draft: false
math: true
authors:
    - yonesuke
categories:
    - Mathematics
    - Graph Theory
---

マット・デイモンとロビン・ウィリアムズ主演の映画『グッド・ウィル・ハンティング/旅立ち』(Good Will Hunting)の中で、MITの廊下に掲示されたグラフ理論の問題を清掃をしていたマット・デイモンが解いてしまうシーンがあります。
中学生とかのときに初めてこの映画を見たときにはよっぽど難しい問題なんだろうな、と思ったのですが、最近見返してみると定義に従って素直に計算すれば解ける問題だということがわかったのでまとめておきます。

!!! quote
    Given the graph $G$, find

    1. The adjacency matrix, $A$
    2. The matrix giving the number of 3 step walks
    3. The generating function for walks from $i\to j$
    4. The generating function for walks from $1\to3$

    ![](graph.png)

<!-- more -->

## 問1
グラフの隣接行列$A$の$(i,j)$成分は$i$から$j$に向かう枝があれば$1$、そうでなければ$0$となるように定義されています。
今の場合、頂点$3$と頂点$4$の間には2本枝があるのでその場合は重み付きの枝だと思って$2$とします。
そうすると隣接行列は、
$$
A=
\begin{pmatrix}
0 & 1 & 0 & 1\\\\1 & 0 & 2 & 1\\\\0 & 2 & 0 & 0\\\\1 & 1 & 0 & 0
\end{pmatrix}
$$
となります。

## 問2
頂点$i$から始まって3ステップで頂点$j$に行くことができる場合の数(経路の数)を聞いています。
一般に$n$ステップに拡張した数を$\omega_{n}(i\to j)$と書くことにします。
例えば2ステップにおける経路の数は頂点$i$からどこかの頂点$k$を経由して頂点$j$にたどり着くことができる場合の数を$k$に渡って和を取れば良いので、
$$
\omega_{2}(i\to j)=\sum_{k=1}^{4}\omega_{1}(i\to k)\omega_{1}(k\to j)
$$
となることがわかります。$\omega_{1}(i\to j)$は隣接行列$A$の$(i,j)$成分に対応することを用いると、
$\omega_{2}(i\to j)$は$A^{2}$の$(i,j)$成分になることがわかります。
より一般に
$$
\omega_{n}(i\to j)=A^{n}[i,j]
$$
もわかります。ただし行列$M$の$(i,j)$成分を$M[i,j]$と書いています。

今問題で聞かれているのは3ステップに到達できる経路の数なので、求める行列は
$$
A^{3}=\begin{pmatrix}
2 & 7 & 2 & 3\\\\7 & 2 & 12 & 7\\\\2 & 12 & 0 & 2\\\\3 & 7 & 2 & 2
\end{pmatrix}
$$
となります。

## 問3
頂点$i$から頂点$j$の経路の数に関する母関数を聞いています。
母関数の定義は
$$
F(z)=\sum_{n=0}^{\infty}\omega_{n}(i\to j)z^{n}
$$
です。母関数の優れている点はべき級数の係数がそのまま$n$ステップかかる経路の数を表すところです。
そのため$z$は本質ではなく、収束についてもあまり気にする必要はありません。形式的べき級数だと思う、ということです。

$\omega_{n}(i\to j)=A^{n}[i,j]$だったことを思い出すと、
$$
F(z)=\sum_{n=0}^{\infty}A^{n}[i,j]z^{n}=\left(\sum_{n=0}^{\infty}z^{n}A^{n}\right)[i,j]
$$
となります。さらにスペクトル半径が1よりも小さい行列のべき級数に対して
$$
\sum_{k=0}^{\infty}M^{k}=(I-M)^{-1}
$$
が成り立つことを思い出すと、
$$
F(z)=(I-zA)^{-1}[i,j]
$$
となります。(スペクトル半径は1より小さくなるくらい$z$は十分に小さい、と思いこめば良いです。今$z$の値は大事ではないので。)

可逆な行列$M$に対して
$$
M^{-1}[i,j]=(-1)^{i+j}\frac{\det \widetilde{M} _ {ij}}{\det M}
$$
が成り立つことを思い出すと
( $\widetilde{M} _ {ij}$は$i$行目と$j$列目を除いた小行列)、
$$
F(z)=(-1)^{i+j}\frac{\det (\widetilde{I} _ {ij}-z\widetilde{A} _ {ij})}{\det(I-zA)}
$$
が得られます。

## 問4
問3をもとに実際に頂点1から頂点3についての母関数を聞いています。$i=1, j=3$を代入すると、
$$
F(z)=\frac{\det(\widetilde{I} _ {13}-z\widetilde{A} _ {13})}{\det(I-zA)}
$$
です。
$$
\begin{aligned}
&\det(I-zA)=\det\begin{pmatrix}1 & -z & 0 & -z \\\\ -z & 1 & -2z & -z \\\\ 0 & -2z & 1 & 0 \\\\ -z & -z & 0 & 1\end{pmatrix} = 4z^4-2z^3-7z^2+1,\\\\&\det(\widetilde{I} _ {13}-z\widetilde{A}_{13})=\det\begin{pmatrix} -z & 1 & -z \\\\ 0 & -2z & 0 \\\\ -z & -z & 1 \end{pmatrix}=2z^{3}+2z^{2}
\end{aligned}
$$
なので、母関数は
$$
F(z)=\frac{2z^{3}+2z^{2}}{4z^{4}-2z^{3}-7z^{2}+1}=\frac{2z^{2}}{4z^{3}-6z^{2}-z+1}
$$
と求まります。$z$が十分に小さいと思って原点付近でTaylor展開をしてみると、
$$
F(z)=2z^{2}+2z^{3}+14z^{4}+18z^{5}+\cdots
$$
となります。$z^{3}$の係数は2で、これは確かに$A^{3}[1,3]=2$に一致します。

## コード
こういった行列計算はSympyを使うと便利です。

<script src="https://gist.github.com/yonesuke/44694b88d8d92eebe7d69fd7182ac444.js"></script>