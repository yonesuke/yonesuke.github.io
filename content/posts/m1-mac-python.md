---
title: "M1 MacでのPython環境構築(tensorflowとか)"
date: 2022-07-11T10:26:10+09:00
draft: false
author: Ryosuke Yoneda
---

Apple Siliconが搭載されたMacが手元に何台かあって、その上で色々と研究をしていたのですが、pythonの環境構築に結構手こずってしまいました。何度か試してうまく行ったものを備忘録として記しておきます。
(ここではpyenvを用いた環境構築について記しています。もちろん他にも良い方法はあると思います。)
インストールに手こずることがあれば随時この記事に書き足していきたいと思います。

特にtensorflowのインストールが難しかったのですが、以下の記事が参考になりました。ありがとうございます。
- https://qiita.com/chipmunk-star/items/90931fb5180b8024adcb

## pyenv
はじめにpyenvをインストールします。これは[pyenvのGitHubページ](https://github.com/pyenv/pyenv#installation)に従えばうまく動きます。

Apple Siliconではarmアーキテクチャが採用されているのでminiforgeを使わないといけない、という説明がネット上に無数にあるのでそれに従います。今ではminiforge以外も対応しているのかもしれないですが、よくわからないです。
この記事を書いている現時点(2022/07/11)ではminiforge3-4.10.3-10が最新のように見えるのでこれをインストールしてglobalの環境に設定しておきます。
```bash
pyenv install miniforge3-4.10.3-10
pyenv global miniforge3-4.10.3-10
```

## TensorFlow
tensorflowのインストールが一番大変で、いろんな記事を眺めてはインストールを試みましたが成功には至りませんでした。上のリンクに従うとすんなりとインストールが完了しました。

インストールに必要なのはtensorflow-deps, tensorflow-macos, tensorflow-metalの3つ(多い！！)らしいです。そのうちtensorflow-deps, tensorflow-macosをインストールするときのバージョンがtensorflowのバージョンに対応するらしいです。そのためにまずtensorflow-depsがインストール可能なバージョンを確認します。tensorflow-depsはcondaによるインストールを利用するらしいのでconda searchを行います。
```bash
-> % conda search -c apple tensorflow-deps
Loading channels: done
# Name                       Version           Build  Channel
tensorflow-deps                2.5.0               0  apple
tensorflow-deps                2.5.0               1  apple
tensorflow-deps                2.6.0               0  apple
tensorflow-deps                2.7.0               0  apple
tensorflow-deps                2.8.0               0  apple
tensorflow-deps                2.9.0               0  apple
```
これを見ると2.9.0のバージョンが一番新しそうです。次にtensorflow-macosのバージョンを確認します。こちらはpipを用いたインストールを行うそうなのでpip searchを行います。と言いたいところですがpip searchは現状使えないのでバージョン部分を空にするという裏技でインストール可能なバージョンをリストアップします。
```bash
-> % pip install tensorflow-macos==
ERROR: Could not find a version that satisfies the requirement tensorflow-macos== (from versions: 2.5.0, 2.6.0, 2.7.0, 2.8.0, 2.9.0, 2.9.1, 2.9.2)
ERROR: No matching distribution found for tensorflow-macos==
```
これを見ると2.9.2までのバージョンが提供されているようです。tensorflow-depsの結果と合わせてここでは2.9.0のバージョンをインストールしましょう。最終的なtensorflowのインストールコマンドは
```bash
conda install -c apple tensorflow-deps==2.9.0
python -m pip install tensorflow-macos==2.9.0
python -m pip install tensorflow-metal
```
になります。これでtensorflowが動くようになりました！！！感動！！！

## TensorFlow Probability
tensorflow probabilityはtensorflowで確率的な推論を用いる際に非常に重宝するライブラリです。こちらはtensorflowのバージョンに非常に強く依存します。この依存関係は[tensorflow probabilityのリリースノート](https://github.com/tensorflow/probability/releases)を読めばわかります。自分の環境にあったバージョンを選択してインストールしましょう。私はtensorflowのバージョンに合わせて0.17.0のバージョンをインストールしました。
```bash
python -m pip install tensorflow-probability==0.17.0
```

## GPflow
gpflowはtensorflowベースのガウス過程回帰ライブラリです。基本的な回帰に合わせてデータ数が多いときに有用な変分推論に基づく回帰(Sparse Variational Inferenceとか言われるもの)に非常に強いものとして知られています。この具体例は[ドキュメント](https://gpflow.github.io/GPflow/2.5.2/notebooks/advanced/gps_for_big_data.html)が詳しいです。
愚直にgpflowをインストールするとsetup.pyにあるtensorflowライブラリが必要と言われてインストールがうまくいきません。これはpipでtensorflowをインストールしたときのライブラリ名がtensorflow-macosであることに起因します。このままではうまく行かないので一旦setup.pyに書かれている依存ライブラリは無視してgpflowをインストールします(若干無茶なことをしていますが、動けば良いのスタンスでやっていきます)。これは--no-dependenciesオプションで実現できます。
```bash
python -m pip install --no-dependencies gpflow
```
あとはpython上で実際にgpflowを動かしてみてライブラリが足りないと怒られたらその都度ライブラリを入れていきましょう。幸いtensorflowをインストールするという山場は超えているので今の所問題は起きていません。今の所インストールが必要になったらライブラリは次の通りです。
```bash
python -m pip install tabulate
python -m pip install deprecated
python -m pip install multipledispatch
python -m pip install lark
```

gpflowのissueを眺めているとgpflow側もこの問題に気づいているようでこれから修正が入っていくことになるのだと思います。[こちらのissue](https://github.com/GPflow/GPflow/pull/1924)が参考になります(本当はこういう問題に気づいた時点でissueを投げていく姿勢が必要なのだろうな〜とは思いますが忙しくてやれない。。。)。

## JAX
jaxが自動微分とjitコンパイルがついたnumpy、なんて呼ばれ方がよくされています。これは本当にそのとおりで個人的に結構好きなライブラリです。jaxは最近Apple Siliconでのインストールに対応しています([こちらのissue](https://github.com/google/jax/issues/5501)が参考になります)。この前まではjaxをimportするとこれはexperimentalだ、みたいな警告文が出てきていたのですが最近はその警告文も消えてます。随分と進歩を感じます。
インストールは基本的に[GitHubページ](https://github.com/google/jax#installation)に従えば良いです。
```bash
python -m pip install --upgrade "jax[cpu]"
```