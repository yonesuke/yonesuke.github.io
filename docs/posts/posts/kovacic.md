---
title: "Kovacicのアルゴリズムを用いて調和振動子を解く"
date: 2022-06-09
slug: kovacic
draft: false
math: true
authors:
    - yonesuke
categories:
    - Mathematics
    - Differential Equation
---

Kovacicのアルゴリズムは有利係数の2階線形常微分方程式を解くアルゴリズムです。与えられた微分方程式が解くことができる場合にはその解を出力し、解くことができない場合にはそうであることがわかるという非常に便利なアルゴリズムになっています。ここで言う"解ける"という言葉は微分ガロア理論の意味で用いられています。僕自身は微分ガロア理論には詳しくはないので細かいことはわかりませんが、細かいことがわからなくてもKovacicのアルゴリズムを使うことができるものになっています。

<!-- more -->

以前所属していた研究室には微分方程式の非可積分性に関する研究があり、その中でKovacicのアルゴリズムを知る機会がありました。Kovacicのアルゴリズムが適用できる微分方程式で僕が一番馴染み深かったのが調和振動子のシュレディンガー方程式でした。実際にその方程式に対してKovacicのアルゴリズムを適用するとたしかに固有エネルギーとその固有状態を得ることができたときはものすごく感動しました。物理の授業で習うような生成消滅演算子を用いた方法でなくとも、可積分なのかどうかという観点からその固有エネルギーを求めることができるのは非常に面白いと思います。というわけでここでその計算の流れを紹介したいと思います。

ここでの計算はすべて
[https://tetobourbaki.hatenablog.com/entry/2018/11/03/231445](https://tetobourbaki.hatenablog.com/entry/2018/11/03/231445)
とそこに添付のPDFを参考にしています。

## 調和振動子
調和振動子のシュレディンガー方程式は
$$
i\hbar\frac{\partial}{\partial t}\phi(x,t)=\left(-\frac{\hbar^{2}}{2m}\frac{\partial^{2}}{\partial x^{2}}+\frac{m\omega^{2}}{2}x^{2}\right)\phi(x,t)
$$
で与えられます。$\phi(x,t)$は波動関数です。特に時間依存しないシュレディンガー方程式は、
$$
\left(-\frac{\hbar^{2}}{2m}\frac{d^{2}}{dx^{2}}+\frac{m\omega^{2}}{2}x^{2}\right)\phi(x)=E\phi(x)
$$
となります。この方程式は解くことができて、固有エネルギーと固有状態は$n\in\mathbb{Z}_ {\geq 0}$に対して

$$\begin{aligned}&E=\left(n+\frac{1}{2}\right)\hbar\omega,\\\\&\phi(x)=H_ {n}\left(\sqrt{\frac{m\omega}{\hbar}}x\right)\exp\left(-\frac{m\omega}{2\hbar}x^{2}\right)\end{aligned}$$

で与えられます。規格化の定数は除いています。$H_{n}$は$n$次のエルミート多項式です。


ここでは微分方程式を解くことに注目するので係数は無次元化しておきます。$m=\hbar^{2},\hbar\omega=1$とすると微分方程式は
$$
\left(-\frac{1}{2}\frac{d^{2}}{dx^{2}}+\frac{1}{2}x^{2}\right)\phi(x)=E\phi(x)
$$
となります。

## Kovacicのアルゴリズム
Kovacicのアルゴリズムに従って調和振動子のシュレディンガー方程式を解いていきます。以下の計算はすべて上のリンクに従っていますので適宜そちらを参照しながら計算を追ってもらうと良いと思います。

### 前処理
はじめに微分方程式を$\ddot{\eta}=r\eta$の形に前処理をします。シュレディンガー方程式は$\ddot{\phi}=(x^{2}-2E)\phi$となります。複素の微分方程式としてこれを考えるので$x$を$z$にして、
$$
\ddot{\eta}=r\eta,\quad r=z^{2}-2E
$$
となります。$r$は$\mathbb{C}$上に特異点はないので、$\Gamma=\emptyset$です。また、$r=(1/z)^{-2}-2E$であるので、$z=\infty$における位数は$-2$です。

### Case 1
#### Step 0
$z=\infty$における位数は偶数なのでStep 1に行きます。

#### Step 1
$(\infty_{3})$を計算します。$z=\infty$における位数は$m=-2\nu$として、$\nu=1$となります。
$\sqrt{r}$の$z=\infty$周りの展開は、
$$
\sqrt{r}=\sqrt{(1/z)^{-2}-2E}=(1/z)^{-1}-E(1/z)^{1}-\frac{E^{2}}{2}(1/z)^{3}+\cdots
$$
となります。よって、
$$
[\omega]_ {\infty}=[\sqrt{r}]_ {\infty}=z
$$
となります。$a$は$\sqrt{r}$のローラン展開の$z^{\nu}=z^{1}$の係数、$b$は$r-[\sqrt{r}]_ {\infty}^{2}$のローラン展開の$z^{\nu-1}=z^{0}$の係数となるので、
$$
a=1,\quad b=z^{2}-2E-z^{2}=-2E
$$
です。よって、
$$
\alpha_{\infty}^{\pm}=\frac{1}{2}(\pm\frac{b}{a}-\nu)=\mp E-\frac{1}{2}
$$
と求まりました。Step 2に行きます。

#### Step 2
$s=(s(c))_ {c\in\Gamma\cup\{\infty\}}$を定めますが、今$\Gamma=\emptyset$なので$s=\pm$について考えれば良いです。このとき、$d_{\pm}=\alpha_ {\infty}^{\pm}=\mp E-1/2$となります。これらが非負の整数である必要があることから場合分けが生じます。

- $d_ {+}$が非負の整数$n\in\mathbb{Z}_ {\geq0}$の場合$E=-n-1/2$であり、
$$
\theta_{+}=+[\omega]_ {\infty}=z
$$
となります。
- $d_ {-}$が非負の整数$n\in\mathbb{Z}_ {\geq0}$の場合$E=n+1/2$であり、
$$
\theta_{-}=-[\omega]_ {\infty}=-z
$$
となります。

これをもとにStep 3に向かいます。

#### Step 3
Step 2に引き続き場合分けが生じます。
- $s=+$のとき、$E=-n-1/2$であり、$\theta_{+}=z$でした。このもとで考えるべき微分方程式は
$$
\ddot{P}+2z\dot{P}-2nP=0
$$
となります。これは任意の$n\in\mathbb{Z}_ {\geq0}$に対して$n$次の多項式解を持ちます(多項式の係数に関する式が出てきます。最高次の係数を$0$以外に適当に定めると他の係数がすべて決まっていきます)。これを$\tilde{H}_ {n}$と置くことにすると、調和振動子の固有エネルギーと固有状態はそれぞれ、
$$
E=-n-\frac{1}{2},\quad\eta=\tilde{H}_{n}(z)e^{z^{2}/2}
$$
と求まることがわかりました。

- $s=-$のとき、$E=n+1/2$であり、$\theta_{-}=-z$でした。このもとで考えるべき微分方程式は
$$
\ddot{P}-2z\dot{P}+2nP=0
$$
となります。これは任意の$n\in\mathbb{Z}_ {\geq0}$に対して$n$次の多項式解を持ちます。しかもこれはエルミート多項式が従う微分方程式になっています。よって解はエルミート多項式$H_ {n}$であり、調和振動子の固有エネルギーと固有状態はそれぞれ、
$$
E=n+\frac{1}{2},\quad\eta=H_{n}(z)e^{-z^{2}/2}
$$
と求まることがわかりました。


## 考察
Kovacicのアルゴリズムを用いて調和振動子のシュレディンガー方程式を解きました。結果をまとめると、正のエネルギーを持つ場合には、$n\in\mathbb{Z}_ {\geq0}$として、
$$
E=n+\frac{1}{2},\quad \phi(x)=H_{n}(x)e^{-x^{2}/2}
$$
となります。これは量子力学で習う結果に完全に一致しています。一方で、負のエネルギーを持つものもあって、$n\in\mathbb{Z}_ {\geq0}$に対して
$$
E=-n-\frac{1}{2},\quad \phi(x)=\tilde{H}_{n}(x)e^{x^{2}/2}
$$
となります。これは量子力学では得られない解です。実際、$\phi$は二乗可積分関数ではないので波動関数にはなり得ません。微分ガロアのもとでは解にはなるけど波動関数の性質を満たさないギャップにこのような解が存在することがわかりました。これはこれで面白いような気がします。

今回は調和振動子のシュレディンガー方程式をKovacicのアルゴリズムを用いて解きました。他の一般のポテンシャルを持つシュレディンガー方程式に関する微分ガロア理論の意味での可積分性については例えば

- [arXiv: 0906.3532](https://arxiv.org/abs/0906.3532)
- [arXiv: 1008.3445](https://arxiv.org/abs/1008.3445)

が詳しいと思います。