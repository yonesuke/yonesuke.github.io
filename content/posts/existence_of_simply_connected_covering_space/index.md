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

{{< thmlike type="Theorem">}}
$O$を$S$における1つの位相とする。$M$が$O$の基底であることと、任意の$A\in O$と任意の$x\in A$に対して、
$$
x\in W,\quad W\subset A
$$
となる$W\in M$が存在することは同値である。
{{< /thmlike >}}

{{< thmlike type="Theorem">}}
空でない集合$S$について、$B(S)$の部分集合$M$が$O(M)$の基底であることは次の2つと同値である。
1. 任意の$x\in S$に対してある$W\in M$が存在して$x\in W$となる。
2. 任意の$W_{1},W_{2}\in M$で$W_{1}\cap W_{2}\ne \emptyset$であるとき、任意の$x\in W_{1}\cap W_{2}$に対して、ある$W\subset W_{1} \cap W_{2}$が存在して$x\in W$なる$W\in M$が存在する。
{{< /thmlike >}}

証明は[位相の基底](../topology_basis)を参照してください。

## 証明

はじめに、*必要条件*を示します。
すなわち、$X$が単連結な被覆空間を持つとき、$X$が半局所単連結であることを示します。

{{< proof >}}
$p\colon \tilde{X}\to X$が被覆空間で$\tilde{X}$が単連結とする。
任意の$x\in X$とその開近傍$U\subset X$に対して、そのリフト$\tilde{U}\subset \tilde{X}$が存在して、$p| _ {\tilde{U}}\colon \tilde{U}\to U$が同相写像となる。
このとき、$U$内の$x_{0}$を基点とする任意のループ$\gamma$に対して、対応する$\tilde{U}$内の$\tilde{\gamma}$が存在する。$\tilde{X}$は単連結であるから$\tilde{\gamma}$は$\tilde{x}$の基点による定数関数にホモトピックになる。対応する$\gamma$も$x_{0}$の基点による定数関数にホモトピックになる。
これより$X$は半局所単連結である。
{{< /proof >}}

次に、*十分条件*を示します。
すなわち、$X$が半局所単連結であるとき、$X$が単連結な被覆空間$\tilde{X}$を持つことを示します。
証明は具体的に$\tilde{X}$を構成し、それが単連結であることを示すことで行います。

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

    はじめに$X$の部分集合族を次のように

    $$
    \mathcal{U}=\left\\{\textrm{path-connected open set }U\subset X \mathrel{}\middle|\mathrel{} \pi_{1}(U)\hookrightarrow\pi_{1}(X)\colon\textrm{trivial} \right\\}
    $$

    で$\mathcal{U}$を定める。
    これが$X$の基底をなすことを示す。これはTheorem 3を用いることによりわかる。

    - $X$の任意の開集合$A$と$A$内の任意の点$x\in A$を取る。$X$が半局所単連結であることから、$x$を含むある開集合$V$が存在して、$\pi_{1}(V)\hookrightarrow\pi_{1}(X)$は自明となる。このとき、$x\in A\cap V$であり、$A\cap V$は開集合であるから$X$が局所弧状連結であることから$x$のある弧状連結開近傍$W$が存在して$W\subset A\cap V$となる。$\pi_{1}(W)\hookrightarrow\pi_{1}(V)\hookrightarrow\pi_{1}(X)$が自明である。これらより$W\in\mathcal{U}$であるからTheorem 3より$\mathcal{U}$は$X$の基底となる。

4. **$\tilde{X}$の基底 $\tilde{\mathcal{U}}$**

    任意の$U\in\mathcal{U}$と$x_{0}\in X$を起点とし$U$内の一点を終点とするpath$\gamma$を取る(このような$\gamma$が取れることは$X$の弧状連結性からわかる)。
    これに対して$\tilde{X}$の部分集合

    $$
    U_{[\gamma]}=\left\\{[\gamma\cdot\eta] \mathrel{}\middle|\mathrel{} \eta\colon \textrm{path in } U \textrm{ with } \eta(0)=\gamma(1) \right\\}
    $$

    を定める。これは$\gamma$のとり方に対してwell-definedである。これは$\gamma\sim\gamma'$ならば$[\gamma\cdot\eta]=[\gamma'\cdot\eta]$からわかる。
    このとき、
    $$
    [\gamma']\in U_{[\gamma]} \Longrightarrow U_{[\gamma]} = U_{[\gamma']}
    $$

    が成り立つ。
    - $U_{[\gamma']}\subset U_{[\gamma]}$であること: $[\gamma']\in U_{[\gamma]}$であるとき、$\gamma(1)$を起点とする$U$内のpath $\eta$が存在して$\gamma'=\gamma\cdot\eta$が成り立つ。このとき、$U_{[\gamma']}$内の任意の元は$U$内のあるpath $\mu$が存在して$[\gamma'\cdot\mu]=[\gamma\cdot\eta\cdot\mu]$と表される。$\eta\cdot\mu$は$U$内のpathであり、$(\eta\cdot\mu)(0)=\eta(0)=\gamma(1)$であるから$[\gamma\cdot\eta\cdot\mu]=[\gamma\cdot(\eta\cdot\mu)]$は$U_{[\gamma]}$の元である。よって、$U_{[\gamma']}\subset U_{[\gamma]}$である。
    - $U_{[\gamma]}\subset U_{[\gamma']}$であること: 上と同様に$\gamma'=\gamma\cdot\eta$と表現する。$U_{[\gamma]}$の任意の元は$U$内のあるpath $\mu$を用いて$[\gamma\cdot\mu]$と書ける。$[\gamma\cdot\mu]=[\gamma'\cdot(\overline{\eta}\cdot \mu)]$であり、$\overline{\eta}\cdot \mu$は$U$内のpathであるから$[\gamma'\cdot(\overline{\eta}\cdot \mu)]$は$U_{[\gamma']}$の元である。よって、$U_{[\gamma]}\subset U_{[\gamma']}$である。

    この結果を用いると$\tilde{\mathcal{U}}=\\{U_{[\gamma]}\\}$が$\tilde{X}$の基底をなすことがわかる。これはTheorem 4が満たされることを確認することにより示される。

    - 任意の$[\gamma]\in\tilde{X}$に対して、$p([\gamma])=\gamma(1)\in X$の開近傍$U\in \mathcal{U}$を取る。このとき、$[\gamma]\in U_{[\gamma]}\in \tilde{\mathcal{U}}$である。
    - $U_{[\gamma]},V_{[\gamma']}\in\tilde{\mathcal{U}}$を任意にとって、$[\gamma'']\in U_{[\gamma]}\cap V_{[\gamma']}$なる元が存在するとする。上の議論から$U_{[\gamma]}=U_{[\gamma'']},V_{[\gamma']}=V_{[\gamma'']}$となる。このとき、$\gamma''(1)\in U\cap V$であり、その開近傍$W\subset U\cap V$を取ると、$W_{[\gamma'']}\subset U_{[\gamma]}\cap V_{[\gamma']}$なる$W_{[\gamma'']}\in\tilde{\mathcal{U}}$が得られる。

5. **$p$は局所同相**

    まず$p| _ {U_{[\gamma]}}\colon U_{[\gamma]}\to U$は全単射であることを示す。
    - 全射であること: 任意の$x\in U$に対して、$\gamma(1)$と$x$をそれぞれ起点・終点とするpath $\eta$であって$\eta([0,1])\subset U$なるものが存在する(このような$\eta$が取れることは$U$の弧状連結性よりわかる)。このとき、$[\gamma\cdot\eta]\in U_{[\gamma]}$であり、$p| _ {U_{[\gamma]}}([\gamma\cdot\eta])=\eta(1)=x$となるから、$p| _ {U_{[\gamma]}}$は全射である。
    - 単射であること: 任意の$x\in U$に対して、$\gamma(1)$と$x$をそれぞれ起点・終点とするpath $\eta_{1},\eta_{2}$が取れる。このとき、$\pi_{1}(U)\hookrightarrow \pi_{1}(X)$は自明より$\eta_{1}\sim\eta_{2}$である。これより、$[\gamma\cdot\eta_{1}]=[\gamma\cdot\eta_{2}]$であるから$p| _ {U_{[\gamma]}}$は単射である。

    次に$p| _ {U_{[\gamma]}}\colon U_{[\gamma]}\to U$が連続であることを示す。
    - $V\subset U$に対して

    最後に$\left(p| _ {U_{[\gamma]}}\right)^{-1}\colon U\to U_{[\gamma]}$が連続であることを示す。
    - 任意の開基底$V_{[\gamma']}\subset U_{[\gamma]}$に対して$p(V_{[\gamma']})=V\subset U$は開集合であるから示された。

6. **$p$は被覆空間**

7. **$\tilde{X}$は単連結**
{{< /proof >}}