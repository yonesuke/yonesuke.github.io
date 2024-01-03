---
title: "MNISTをMLPで推論(Julia/Flux実装)"
date: 2021-08-23
draft: false
math: true
authors:
    - yonesuke
---

Juliaで機械学習をするための有名なライブラリに[Flux](https://github.com/FluxML/Flux.jl)があります。Fluxを使ってMNISTの手書き数字の推論を行ったのでその方法をまとめておきます。
コードは次のようになります。[これ](https://github.com/FluxML/model-zoo/blob/master/vision/mlp_mnist/mlp_mnist.jl)を参考に書きました。

{{< gist yonesuke afc39543fb1de5ff484fa812e3ca5a1d >}}

## パッケージ
基本的に`Flux`さえあれば良いです。今回はMNISTデータを用いるので`MLDatasets`というパッケージを用いてデータを読み込みます。これらのパッケージは事前にインストールしておく必要があります。JuliaのREPLやnotebook上で次を入力してください。
```julia
julia> import Pkg; Pkg.add(["Flux", "MLDatasets"])
```
これでパッケージを読み込むことができます。
```julia
using Flux
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold
using Flux.Losses: logitcrossentropy
using MLDatasets
```

## データの読み込み
MNISTデータを読み込みます。
```julia
x_train, y_train = MLDatasets.MNIST.traindata(Float32)
x_test, y_test = MLDatasets.MNIST.testdata(Float32)
```
MNISTの画像はサイズ(28,28,1)になっていますが、MLPには1次元の配列として渡したいので`flatten`で各データを1次元に落とします。
```julia
x_train = Flux.flatten(x_train) # 784×60000
x_test = Flux.flatten(x_test) # 784×10000
```
また、各画像の数字(0~9)はone-hotにしておきたいのでそちらは`onehotbatch`という関数で変換しておきます。
```julia
y_train = onehotbatch(y_train, 0:9) # 10×60000
y_test = onehotbatch(y_test, 0:9) # 10×10000
```

## モデルの定義
いよいよモデルの定義です。今回は一番簡単なMLPで実装していきます。
```julia
img_size = (28,28,1)
input_size = prod(img_size) # 784
nclasses = 10 # 0~9
# Define model
model = Chain(
    Dense(input_size, 32, relu),
    Dense(32, nclasses)
)
```
`Dense(input_size, output_size, f)`という関数$F\colon\mathbb{R}^{\mathrm{inputsize}}\to\mathbb{R}^{\mathrm{outputsize}}$は
$$
F(x) = f(Wx+b)
$$
になります。$f$は活性化関数です。$W,b$は内部で勝手に定義されます。デフォルトでは初期値$W,b$はGlorotの一様分布に従ってランダムに選ばれます。
また、$f$を指定しなければ活性化関数は恒等関数になります。すなわち非線形変換は行われません。
今回は活性化関数にReLU関数を用いました。
`Chain`は合成関数を作ります。すなわち、`Chain(F,G)`は$G\circ F$という関数に対応します。
今回定義したモデルは784次元の入力から10次元の出力を返します。出力の10次元の中で一番大きい要素のindexが推定される数字とします。

定義したモデルから学習すべきパラメータを取り出しておきます。
```julia
parameters = Flux.params(model)
```

## 学習の準備
データが多いのでミニバッチ学習を行いましょう。バッチサイズとエポック数を定義します。
```julia
batch_size = 256
epochs = 10
```
これをもとにtrainデータとtestデータをバッチに分けていきます。
```julia
train_loader = DataLoader((x_train, y_train), batchsize=batch_size, shuffle=true)
test_loader = DataLoader((x_test, y_test), batchsize=batch_size, shuffle=true)
```
また、学習則にはAdamを用いてみましょう。
```julia
opt = ADAM()
```

## 損失関数
損失関数を定義します。入力`x`に対して出力`ŷ=model(x)`は10次元のベクトルになりますが、これは特に正規化されていません。本来は出力の段階でsoftmax関数で正規化すべきかもしれませんが、推定の意味においては最大値を取りさえすれば良いので特に問題はありません。
また、softmaxを通した10次元の離散分布`softmax(ŷ)`とone-hotの分布`y`の間の交差エントロピー`crossentropy(softmax(ŷ), y)`を計算すると数値的な誤差が生まれやすいことが知られています。
数学的にこれと等価な`logitcrossentropy(ŷ, y)=crossentropy(softmax(ŷ), y)`を用いたほうが数値的にも安定します。
よって損失関数は次のように定義します。
```julia
function loss(x, y)
    ŷ = model(x)
    return logitcrossentropy(ŷ, y, agg=sum)
end
```
また、epochごとの損失関数の値と精度を計測する関数も定義しておきます。
```julia
function loss_accuracy(loader)
    acc = 0.0
    ls = 0.0
    num = 0
    for (x, y) in loader
        ŷ = model(x)
        ls += logitcrossentropy(ŷ, y, agg=sum)
        acc += sum(onecold(ŷ) .== onecold(y))
        num +=  size(x, 2)
    end
    return ls/num, acc/num
end
```

## 学習
いよいよ学習させます。各エポックごとにtrainデータとtestデータのlossと精度を出力する関数を定義しておきます。
```julia
function callback(epoch)
    println("Epoch=$epoch")
    train_loss, train_accuracy = loss_accuracy(train_loader)
    test_loss, test_accuracy = loss_accuracy(test_loader)
    println("    train_loss = $train_loss, train_accuracy = $train_accuracy")
    println("    test_loss = $test_loss, test_accuracy = $test_accuracy")
end
```
Flux内の`Flux.train!`関数でミニバッチ学習をしてもらいます。
```julia
for epoch in 1:epochs
    Flux.train!(loss, parameters, train_loader, opt)
    callback(epoch)
end
```
10エポックの学習で96％近くの学習精度を達成できます。
```
...
Epoch=10
    train_loss = 0.12148669401804606, train_accuracy = 0.9660333333333333
    test_loss = 0.14259111329317092, test_accuracy = 0.9594
```
また、手元の環境(M1 mac mini)で10エポック回すのに6.03秒かかりました。GPUとかは使わなくてもここまでの速度と精度が出るのは素晴らしいですね。
