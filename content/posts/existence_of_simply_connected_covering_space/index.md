---
title: "単連結な被覆空間の存在"
date: 2023-02-17T22:17:24+09:00
draft: true
math: true
author: Ryosuke Yoneda
---

ホモロジーゼミの基本群パートの一つの山場である、被覆空間の分類定理がやってきました。
この定理を示すためには、**単連結な被覆空間の存在証明**が必要になります。

{{< thmlike type="Theorem" >}}
弧状連結かつ局所弧状連結な位相空間$X,\tilde{X}$で、$\tilde{X}$は$X$の被覆空間とする。$\tilde{X}$が単連結になるための必要十分条件は$X$が半局所単連結であることである。
{{< /thmlike >}}

## 準備

はじめに、この証明に必要ないくつかの定義・定理を示します。

{{< thmlike type="Definition" title="半局所単連結" >}}
位相空間$X$が**半局所単連結**であるとは、任意の$x\in X$に対して$x$の開近傍$U$が存在して、包含写像$i\colon U\hookrightarrow X$から誘導される基本群の準同型写像$i_{\ast}\colon \pi_{1}(U,x)\hookrightarrow \pi_{1}(X,x)$が自明になることである。
{{< /thmlike >}}

## 証明

はじめに、必要条件を示します。
すなわち、$X$が単連結な被覆空間を持つとき、$X$が半局所単連結であることを示します。

{{< proof >}}
$p\colon \tilde{X}\to X$が被覆空間で$\tilde{X}$が単連結とする。
任意の$x\in X$とその開近傍$U\subset X$に対して、そのリフト$\tilde{U}\subset \tilde{X}$が存在して、$p| _ {\tilde{U}}\colon \tilde{U}\to U$が同相写像となる。
このとき、$U$内の$x_{0}$を基点とする任意のループ$\gamma$に対して、対応する$\tilde{U}$内の$\tilde{\gamma}$が存在する。$\tilde{X}$は単連結であるから$\tilde{\gamma}$は$\tilde{x}$の基点による定数関数にホモトピックになる。対応する$\gamma$も$x_{0}$の基点による定数関数にホモトピックになる。
これより$X$は半局所単連結である。
{{< /proof >}}

次に、十分条件を示します。
すなわち、$X$が半局所単連結であるとき、$X$が単連結な被覆空間を持つことを示します。

{{< proof >}}
弧状連結、局所弧状連結、半局所単連結な位相空間$X$で$x_{0}$を基点とする。
このとき、単連結な被覆空間を
$$
\begin{aligned}
\tilde{X} &=  \left\\{[\gamma] \mathrel{}\middle|\mathrel{} \gamma\colon \textrm{path in } X \textrm{ starting at } x_{0}\right\\} \\\\
p &\colon \tilde{X}\to X;\quad[\gamma] \mapsto \gamma(1)
\end{aligned}
$$
で定める。$[\gamma]$は基点と終点を定めたときのpathをホモトピックなもので割った同値類である。
このように構成された空間が実際に単連結な被覆空間であることを示す。

1. **$p$はwell-defined**

    $[\gamma_{1}]=[\gamma_{2}]$のとき、$\gamma_{1}(1)=\gamma_{2}(1)$であるから、$p$はwell-definedである。

2. **$p$は全射**

    任意の$x\in X$に対して、$X$は弧状連結であるから、$x_{0},x$をそれぞれ基点・終点とするpath$\gamma$が存在する。このとき、$p([\gamma])=\gamma(1)=x$であるから、$p$は全射である。


3. **$X$の基底 $\mathcal{U}$**

4. **$\tilde{X}$の基底 $\tilde{\mathcal{U}}$**

5. **$p$は局所同型**

6. **$p$は被覆空間**

7. **$\tilde{X}$は単連結**
{{< /proof >}}