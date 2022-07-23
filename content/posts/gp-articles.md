---
title: "ガウス過程に関する文献"
date: 2021-05-06T12:53:12+09:00
draft: false
math: true
author: Ryosuke Yoneda
---

ガウス過程とその機械学習への応用に関する文献はたくさんあります。ここで特に有用だと思ったものを紹介していきます。

## ガウス過程全般

- [ガウス過程と機械学習](https://www.amazon.co.jp/dp/4061529269)

    ガウス過程と機械学習に関する和書です。非常にわかりやすい本でこの本を読めば一通りの実装はできるようになると思います。一冊目としてはこの本で間違いない気がします。

- [Gaussian Processes for Machine Learning](http://gaussianprocess.org/gpml/)

    ガウス過程と機械学習への適用に関する代表的な本だと思います。PDFはオンラインで読むことができます。


## ガウス過程の数学的な基礎に関する文献

- [Gaussian Processes and Kernel Methods: A Review on Connections and Equivalences](https://arxiv.org/abs/1807.02582)

    ガウス過程はカーネル法と切っても切り離せない関係にあります。この論文ではカーネル法との関係に関する結果を簡潔にまとめていて非常に参考になります。例えばRBFカーネルによって生成されるサンプルの関数が確率1で$C^{\infty}$級のなめらかな関数になることはよく知られていますが、その証明の流れもこの論文を読めばわかります。

- [Reproducing Kernel Hilbert Spaces in Probability and Statistics](https://www.springer.com/gp/book/9781402076794)

    再生核ヒルベルト空間(RKHS)に関する本です。ガウス過程によって出力される関数が属する空間について議論する際にRKHSが出てきます。この本ではRKHSに関する議論やそのガウス過程との関係についても議論されています。この本の一番最後にカーネル関数とその対応するRKHSの具体例が27個載っていて圧巻です。

## 補助変数法

データが$N$個あるときにガウス過程回帰を行うと、データ数に応じた逆行列計算が必要で$O(N^{3})$の計算量がかかってしまいます。
データ数が非常に多くなったときにはこの計算量は現実的ではないので補助変数法と呼ばれるものを用いて計算量を削減する試みが行われています。

- [A Unifying View of Sparse Approximate Gaussian Process Regression](https://www.jmlr.org/papers/volume6/quinonero-candela05a/quinonero-candela05a.pdf)

    2005年に出た論文ですが、その時までに出ていた補助変数法の様々な手法をまとめた論文になっています。上で紹介した「ガウス過程と機械学習」ではこの中のFITCと呼ばれる方法が紹介されています。

- [Gaussian processes for Big data](https://dl.acm.org/doi/10.5555/3023638.3023667)

    変分ベイズ法を用いてガウス過程回帰を行う手法についてまとまった論文です。補助変数の配置などを含めたハイパーパラメーターの最適化が可能になります。

## 深層学習との関係

- [Deep Neural Networks as Gaussian Processes](https://arxiv.org/abs/1711.00165)

    各層のunit数を無限に飛ばしたニューラルネットワークがガウス過程に対応する、という話です。中間層が一層の場合の話は昔に知られていましたが、多層の場合もそうだそうです。紹介しておきながら腰を据えて読んだことはないのでちゃんと読みたいです。