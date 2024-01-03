---
title: "包除原理"
date: 2021-05-16
draft: false
math: true
authors:
    - yonesuke
---

測度空間$(X,\mathcal{B},\mu)$の有限測度集合$A_{i}(i=1,\dots,n)$に対して
$$
\mu\left(\bigcup_{i=1}^{n}A_{i}\right)=\sum_{J\subset[n];J\ne\emptyset}(-1)^{|J|-1}\mu\left(\bigcap_{i\in J}A_{i}\right)
$$
が成り立ちます。これを包除原理(Inclusion-exclusion principle)と呼びます。

<!-- more -->

## 証明
証明は
$$
\int_{X}\left(1-\prod_{i=1}^{n}(1-1_{A_{i}})\right)d\mu
$$
を二通りに計算することにより求まります。

はじめに愚直に展開してみることにします。
$$
\prod_{i=1}^{n}(1-x_{i})=\sum_{J\subset[n]}(-1)^{|J|}\prod_{i\in J}x_{i}
$$
は展開すればわかるので、この$x_{i}$に$1_{A_{i}}$を代入すると、
$$
1-\prod_{i=1}^{n}(1-1_{A_{i}})=1-\sum_{J\subset[n]}(-1)^{|J|}\prod_{i\in J}1_{A_{i}}
=\sum_{J\subset[n];J\ne\emptyset}(-1)^{|J|-1}1_{\bigcap_{i\in J}A_{i}}
$$
となります。ここで$1_{A}1_{B}=1_{A\cap B}$を用いました。以上より、
$$
\int_{X}\left(1-\prod_{i=1}^{n}(1-1_{A_{i}})\right)d\mu
= \sum_{J\subset[n];J\ne\emptyset}(-1)^{|J|-1}\mu\left(\bigcap_{i\in J}A_{i}\right)
$$
がわかります。これで包除原理の右辺が示されました。

次に賢い計算をしましょう。$1-1_{A}=1_{\overline{A}}$なので、
$$
1-\prod_{i=1}^{n}(1-1_{A_{i}})=1-\prod_{i=1}^{n}1_{\overline{A_{i}}}=1-1_{\bigcap_{i=1}^{n}\overline{A_{i}}}
=1_{\overline{\bigcap_{i=1}^{n}\overline{A_{i}}}}
=1_{\bigcup_{i=1}^{n}A_{i}}$$
となります。最後にドモルガンの法則を用いました。
よって、
$$
\int_{X}\left(1-\prod_{i=1}^{n}(1-1_{A_{i}})\right)d\mu
=\int_{X} 1_{\bigcup_{i=1}^{n}A_{i}}d\mu
= \mu\left(\bigcup_{i=1}^{n}A_{i}\right)
$$
となります。
これで包除原理の左辺が導かれ、包除原理が示されました。