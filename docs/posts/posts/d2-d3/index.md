---
title: "博士課程2年目3年目を振り返る"
slug: d2_d3
date: 2023-07-29
draft: false
math: true
authors:
    - yonesuke
categories:
    - Diary
---

# 博士課程2年目3年目を振り返る

大昔に[博士課程1年目を振り返る](../d1.md)という記事で博士課程1年目のときに起こったことを書いていったのだが、
2年目3年目については特に何も書いていなかった。
その代わりといっては何だが、研究室内のwikiでつらつらと簡単な日記を書いていたので、それをここに貼っておくことにする。
研究室内のネットワーク関連の情報も一部載っていたので、そちらは非表示で対応することにしている。
あと画像・PDFは移植が面倒すぎたので諦めた。
こうしてみると、色々と手を動かしてもがいた3年間だったなあ、と。あとモチベーション維持にも苦労した3年間でもあったかなあ。
いつかまた読み返してこんなこともあったなあと笑えるように備忘録として残しておく。

<!-- more -->

## 研究日誌

### 2023/03/29
* NNを使って臨界指数を計算する論文がアクセプトされた。アクセプトのメールを開いたときに感情が全く動かなかった。感情が全く動かない自分にむしろ驚いた。もう論文書くとかのモチベーションが0に収束してしまった感じがあるなあ。
* もう卒業らしい。長い学生生活も終焉を迎えてしまう。

### 2023/02/24
* sekiの上でflexgenを動かした。楽しいわな。しょうみ爆笑やわ。

### 2023/02/21
* huggingfaceに学習済みのパラメーターあるから誰かtensor networkで行列分解してどれくらいモデル圧縮できるかやってほしい。頼む。
* GPT2の実装完全に理解した。

### 2023/02/16
* [https://suzuri.jp/yonesuke1729](https://suzuri.jp/yonesuke1729) 蔵本モデルグッズを作ったので気が向いたら買ってください。

### 2023/01/25
* 臨界点近傍でもきちんと値が収束するところまで計算ができたらひとまず完成かなあ。色々試してみたい。
* Ott-AntonsenをODEに落とし込んで直接解くというアイデア、いろんな数値積分手法で試してる。ガウス求積法の中でGauss-Hermiteは結構期待してたんやけど全然あかんかった。あとDE公式は一瞬で破綻してる。使い方を間違ってるかも。$[-1,1]$の区間に変換してtrapezoidで値を足し込むのが一番良いのかもしれない。あとは$[-1,1]$の区間に変換してGauss-Legendreで解くのはありなのかも。それも試してみる。

### 2023/01/16
* (おそらく)卒業するのでこれまで蔵本モデルについて勉強した色々をまとめている。`jaxkuramoto`というパッケージとして出してる。ドキュメントを整備していく。
[https://github.com/yonesuke/jaxkuramoto](https://github.com/yonesuke/jaxkuramoto)
* 生協のレジシステムがyahoo newsになってておもろい。
[https://news.yahoo.co.jp/articles/c92e3b04375c89fa20a1640e906a6e9769994e3d](https://news.yahoo.co.jp/articles/c92e3b04375c89fa20a1640e906a6e9769994e3d)

### 2023/01/13
* 自動微分がかなりあついなあ。$\int_{a}^{b}f(x,\theta)\mathrm{d}x$の$a,b,\theta$微分とかも自動微分設定したら出来るなあ。だいぶおもろくなってきてる。これ蔵本モデルのself-consistent方程式の解析に応用できるかも。
* 27になった。

### 2023/01/11
* やってることneural odeのadjoint methodとほとんどパラレルな気がしてきた。
* パラメタライズされた離散力学系の固定点周りの微分はbackpropで求まるらしい。天才かな。
[https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html#implicit-function-differentiation-of-iterative-implementations](https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html#implicit-function-differentiation-of-iterative-implementations)

### 2023/01/10
* microsoft teamsで問題を作るサービスがあるんやけどそこでlatexが使えるらしくでびっくりした。
* forwardの方向微分とかはdual number使ったほうが良いとかはありそう、などと思った。
* jaxのgradのアイデアは基本的にはautogradと同じやと思うけど、jaxにはあとjitコンパイルとかvmapの並列化とかがあってそれも理解するには道のりがかなり遠い。
* これがわかりやすい。
[https://www.cs.toronto.edu/~rgrosse/courses/csc421_2019/readings/L06%20Automatic%20Differentiation.pdf](https://www.cs.toronto.edu/~rgrosse/courses/csc421_2019/readings/L06%20Automatic%20Differentiation.pdf)
* jaxの前身のautogradの実装を見ながら手元で実装してみてる。Boxってのを作っててそこの理解がまだあやふや。

### 2023/01/09
* リミットサイクルを持つ振動子のアイソクロンを書いて遊んでた。博論に載せる。
* reviseしたものをsubmitした。一つ肩の荷が降りた。

### 2023/01/03
* SSHがない。。。下にある><ぽいのを押したら一応sshはできる。
* vscodeのremote explorerからssh接続の一覧が消えてるんやけどなんで？？謎いなあ。

### 2023/01/01
* 数値計算結果がどうみても想定と違う結果を出してる。jaxのデフォルトがfloat32なのでそれに起因しているのかも。float64を有効にして計算しなおす。
* あけおめ。ことよろ。

### 2022/12/31
* どうでもよいSSH情報。`ssh server hoge`でserver先にコマンド`hoge`を送ってその結果を返してくれる。例えば`ssh euler nvidia-smi`でeulerのGPU使用率をlocalから手軽に確認できる。`ssh euler "cat ~/hoge/nohup.out"`で(nohupに実験結果を投げてたら)eulerで実験中の結果を確認できる。
* モクモクと博論の図をいじる。毎度恒例の「論文投稿直前matplotlib能力向上」が今回も観測されている。

### 2022/12/29
* JAXを使うべき(使わないべき)理由を色んな人が書いてくれてるので非常にためになる。もう2022年終わるんやけどな。 [https://www.reddit.com/r/MachineLearning/comments/st8b11/d_should_we_be_using_jax_in_2022/](https://www.cs.toronto.edu/~rgrosse/courses/csc421_2019/readings/L06%20Automatic%20Differentiation.pdf)

### 2022/12/28
* GitHub copilotがフーリエ級数展開を完全に理解してて怖いです
* フーリエ級数の足し算引き算掛け算微分積分を備えたclassを作成したほうが良い気がしてきた。適用範囲は広い気がする。
* 予備審査願に論文概要の提出が必要らしい。慌てて書く。書き終わったら家の掃除して実家帰る。

### 2022/12/25
* 来年はなんか新しいことを勉強してみたい。
* 月曜日と火曜日で博論をひとまず完成させるのとrebuttalを提出する。それで今年は終わりかなあ〜〜

### 2022/12/23
* 仕事納め頑張るぞ〜〜
* 博論及び発表スライドを一生作ってます。rebuttalの方も一通り終わったので年を越す前に再投稿するぞ。

### 2022/12/07
* ホモロジーのゼミで"自由群の部分群は自由群"っていうNielsen–Schreierの定理を知った。これいかつすぎんか？$\mathbb{S}^{1}\vee\mathbb{S}^{1}$の基本群が$\mathbb{Z}\ast\mathbb{Z}=F_{2}$になるっていうのがあって、そのcovering spaceの中で基本群が$F_{n}$になるようなものが構成されて、しかもcovering spaceの基本群はもとの空間の基本群の部分群になるから、、、というのが大まかな流れっぽい(なんか勘違いしてるかも)。いやあすごいなあ、と久しぶりに思った定理やった。

### 2022/11/28
* プログラムを少し書いた。一歩前進。pythonとtexを一生書く。

### 2022/11/21
* 宮崎駿もこの気持ち。
* 研究する気力がまじで湧いてこない。neural scaling analysisのrebuttalをしようと思ってjupyter notebookまで開きはするけどそこから手が動かない。あー踏ん張り時。石川佳純もこの気持ちやし頑張る。
* 先週は婚姻届を出した。特に区役所から電話もなかったので受理されたと思う。人生を感じる。
* 春先まで使ってたパソコンに今一番必要としているコードが残されていた。。。慌てて取り出した。

### 2022/11/09
* 研究に専念していきたい。
* 試験受かってた〜〜まじで嬉しい。肩の荷が下りた気がする。

### 2022/11/03
* 家のシャワーからお湯が出なくなった。大ピンチ。とりあえず今はanytimeのシャワーを使うことでしばらく生きていくことにする。
* 資格試験の勉強にすべての時間が吸われてる。

### 2022/10/28
* プロポーズしたとだけ書いて結果を書いてなかったがために何人かから心配されたんやけど無事にokしてもらってます笑。来月婚姻届を京都市に出す予定。
* neural sdeの実行時間。pytorch実装だと1000iterationで1時間もかかってたのにjax実装だと5000iterationで9秒だった。jaxの圧勝。

### 20022/10/26
* diffraxの使い方がなんとなくわかってきた
* 任意のことをする時間が足りない。まじで耐え時や〜〜〜

### 2022/10/18
* そんなこんなでこの2週間まじで研究できてない。頑張ろう。
* 昨日プロポーズをした。緊張した。まじで人生を感じる。

### 2022/10/08
* 博論に使うフォーマットを何にしようかなとぼんやり考えてる。overleafにphd thesisのtemplateがたくさんあるからそこからいい感じのを選びたいと思っていて。

### 2022/10/07
* 査読帰ってきた〜〜好意的ではあるがコメント多すぎな。ちゃんと読んでくれてるのは伝わるからそれは嬉しい。2週間後には返事を返せるように頑張ってコメントを潰していく。

### 2022/10/06
* PREに出してたのはそろそろ査読が帰ってきそう。ほんまは今ごろlevy kuramoto投稿してるはずやったのになあ。時間がないことをひたすら嘆くだけの人間になっている。
* なんだか気がついたら書かないと行けない論文がどんどん溜まってる。。。気張るしかない。。。

### 2022/10/05
* neural odeの話は卒業までにarxivに投稿してそれを博論にもぶち込む感じになるかなあ。acceptまでは時間的に厳しい気がする。
* levy kuramotoも早く完成させてarxivに上げたいけど資格試験の勉強もしなあかんしなあ。時間をくれ。
* torchsdeを使ってneural sdeを実装している。torchの書き方でつまるところがあって難しい。

### 2022/10/01
* ベーテ格子上のイジングモデルの話を途中までまとめた。
[https://yonesuke.github.io/posts/bethe_ising/](https://yonesuke.github.io/posts/bethe_ising/)
* もう10月とか聞いてないよ〜〜やめてくれ〜〜

### 2022/09/30
* Bethe lattice上のIsing modelの計算を追った。再帰的な構造があると物理量の計算が漸化式でかけて、あとはそれを力学系的な解析で安定が求まって臨界指数とかも芋づる式に求まっていくのは爽快。再帰的な構造はやはり素晴らしい。
* 投稿完了。アクセプトされたとしてアメリカに行くお金は足りるのか？？円安急に崩れへんかな。
* ちょっと前に『ウォール街の物理学者』を読んだ。臨界現象大好き人間としてはやっぱりドラゴンキング理論の話について書いてある章が面白かった。バブルが弾ける瞬間を臨界現象になぞらえてそういった急激な変化を予測する理論がドラゴンキング理論、という説明がなされてる。今の円安も急に変化が起きへんかな。ほんでそれをドラゴンキング理論で評価できへんかな。

### 2022/09/28
* 研究もしないと行けないけど資格勉強が割と切羽詰まって来てる気がする。色々やることが溜まってきてますねえ。秒速で終わらせていきたい。
* 非常勤の授業が今期も始まる。最後の講義ですねえ。
[https://yonesuke.github.io/teaching/2022-suutekirikai/](https://yonesuke.github.io/teaching/2022-suutekirikai/)
* ひとまず原稿が完成した。適当に推敲して投稿する。アメリカ行けるといいなあ。

### 2022/09/23
* documentを整理してた。
[https://yonesuke.github.io/jaxfss/](https://yonesuke.github.io/jaxfss/)
* NNの説明のとこを短くして実験結果を簡潔にまとめたら4ページに収まるようになる。気がする。

### 2022/09/19
* とか言いつつ締め切りが一週間伸びたので今日は資格勉強をした。
* 4ページにまとめるのむず〜〜明日台風やし一日こもってまとめる。

### 2022/09/17
* NeurIPSのworkshopに出す。締め切りが9/22やから気張らなあかん。燃えてきてる。
[https://ml4physicalsciences.github.io/2022/](https://ml4physicalsciences.github.io/2022/)

### 2022/09/11
* 大谷見てて守備が終わってるせいで失点してるんを見るの辛いな。
* [https://docs.manim.community/en/stable/index.html](https://docs.manim.community/en/stable/index.html)

### 2022/09/10
* 最近時速10km30分間で走ってたのを今日(昨日)は思い切って11kmにあげた。だいぶきつかった。当面の目標は時速12kmかな。ほんまは1時間走りたいけどだいぶ体力削がれて研究する気力が失われるのが悩ましいところではある。

### 2022/09/07
* PRE投稿のためにcover letterを書く。久々にword使ってみてるけど使い勝手の良さに感動してる。
* arxiv投稿done。おつかれ。
[https://arxiv.org/abs/2209.01777](https://arxiv.org/abs/2209.01777)

### 2022/09/06
* 自由積とか融合積とかあとはリー群ゼミで出てきたテンソル積の定義とかで普遍性の概念が自然と出てきててこういうのが圏論を勉強するモチベーションになるのかなあとか思いながらまあええかとなる。

### 2022/09/05
* 論文1のarxiv投稿ほぼdone 
* 論文2を頑張る

### 2022/09/02
* また一つmatplotlibの能力が上がった。論文の投稿間際はmatplotlib能力上がりがち。
[https://stackoverflow.com/questions/53433272/matplotlib-setting-a-tick-labels-background-colour](https://stackoverflow.com/questions/53433272/matplotlib-setting-a-tick-labels-background-colour)

### 2022/09/01
* 明日参考文献の微調整を行って先生にまた原稿を送る。あと図も微調整する。

### 2022/08/31
* 8月が終わる〜〜論文1が収束しそう。論文2を9月末までに完成させるのを目標にする。

### 2022/08/29
* Stochastic Gradient Langevin DynamicsのJAX実装を行った。JAX特有の書き方が必要で若干めんどくさかった。死ぬほど遅い。早いこと学習してほしい。

### 2022/08/27
* 数値計算して論文書いてを一生してた8月やった。振り返ってもanytimeか研究室かしか記憶がない。

### 2022/08/24
* 理論結果とずれてた理由は結局時間刻み幅が大きかったからやった。まあ確かにLevyノイズが入るから時間刻み幅をだいぶ小さくしないとゆらぎの影響がかなり効いてきてしまうのはそれっぽい理由ではある。

### 2022/08/23
* LevyKuramotoの数値計算結果眺めてたらCauchyノイズの場合の解析解が有限サイズのときの解となんだかズレてることに気づいた。明日JAXで実装し直してみることにする。
* 人事と喋った。一般的には内定は10月1日からしか出せないらしいけど博士課程の学生はそのルールは適用されないらしい。なので本当はもう内定を出せるらしい、とかいう謎の話を聞いた。なんかだるそうやわな。
* 論文をちょっとずつ書いて先生に送った。日々の積み重ねやわなあ。

### 2022/08/19
* 論文に新しいセクションを追加して文章を読み直して明日先生に見てもらう。頑張ろう。
* しばらく日記を書いてなかった。最近はホモロジーゼミが順調に進んでいて面白い。
[https://yonesuke.github.io/posts/brouwer-fixed-point-thm/](https://yonesuke.github.io/posts/brouwer-fixed-point-thm/)

### 2022/08/09
* Levy Kuramotoのコードをeulerで回した。明日はneural odeの数値計算と論文執筆を頑張る。
* 証券外務員の資格勉強を始めた。頑張ろう。

### 2022/08/06
* [https://aip.scitation.org/doi/full/10.1063/5.0093577](https://aip.scitation.org/doi/full/10.1063/5.0093577)
  * Ott-Antonsenは蔵本モデルの低次元縮約の方法として有名やけど、自然振動数の従う分布がコーシーとかの有理型関数じゃないと秩序変数の閉じたODEを得ることができないという難点があった。そんなわけで振動子やる人間はコーシー分布を使いまくってる。けどガウス分布でも秩序変数の式が求まるといいよねってことで頑張ってるのがこの論文。$e^{-x^{2}}=1/e^{x^{2}}$の分母を適当な次数までのテイラー展開で打ち切って有理型関数にみなす、というめちゃくちゃ泥臭いやり方。数値計算とも合ってる。

### 2022/08/03
* 夏休みやししばらく遊んでた。明日から研究を再開しよう。

### 2022/07/28
* 論文書かなあかんなあ。あとLie群の本の続き読んで行きたいなあと今日思った。
* neural odeの学習まあ上手くいってるわ。学習データの作り方が良くないだけやった。ただ、当たり前やけど学習の各ステップで勾配計算にODEを解く必要があるから必然的に時間はかかるわなあ。なんかいい感じに高速化する方法ないんかね。

### 2022/07/26
* 有限$\chi$スケーリングってのがあるんやな。共形場理論から導出されるのか。色々あるんやなあ。

### 2022/07/23
* 論文初稿を見てもらっている間に次の論文を書き始める。今までサボってたつけが来て非常に面倒くさいがどうしようもないので頑張る。creepy nutsのlazy boyにある「きっとコイツはカルマ」やなあとか思う。だいぶスケールちゃうけど。
* ホームページ大改修 
[https://yonesuke.github.io/](https://yonesuke.github.io/)

### 2022/07/21
* neural odeの学習がアホみたいに時間かかって学習も全然うまく行っていない問題をどうにかしないといけない。頑張ろう。
* 論文の終わりが見えてきた。明日先生に見せることを目標に頑張る。

### 2022/07/17
* 同期現象を研究するものとして興味深い動画やった。
[https://youtu.be/Wh6F8Ly5gfo](https://youtu.be/Wh6F8Ly5gfo)

### 2022/07/14
* パーコレーションのあれこれをまとめるサイトを作ってる。悪くない。
[https://yonesuke.github.io/percolation/critical_exponents.html](https://yonesuke.github.io/percolation/critical_exponents.html)
* 卒業するまでに蔵本モデルもまとめたい。

### 2022/07/12
* 前に作ったpraxのドキュメントの雛形を作った。結構かんたんやったのでこれから色々書いていきたい。
[https://prax.readthedocs.io/](https://prax.readthedocs.io/)
* 論文論文論文実装実装実装
* 就職先の同期がどうやら共形場理論をやってる博士っぽいので色々教えを請いたい。今度喋ったときにゼミできないか聞いてみるかな。

### 2022/07/11
* M1 Macのpython環境構築をまとめた。
[https://yonesuke.github.io/posts/m1-mac-python/](https://yonesuke.github.io/posts/m1-mac-python/)

### 2022/07/10
* これちゃんとわかってなかった。
[https://stackoverflow.com/questions/20778771/what-is-the-difference-between-0-0-0-0-127-0-0-1-and-localhost](https://stackoverflow.com/questions/20778771/what-is-the-difference-between-0-0-0-0-127-0-0-1-and-localhost)
* なるほどな〜vscodeだいぶ便利になるなあ。
[https://youtu.be/q2viJSYyKio](https://youtu.be/q2viJSYyKio)
* M1 Macでのtensorflowインストールがきれいにまとまってる。
[https://qiita.com/chipmunk-star/items/90931fb5180b8024adcb](https://qiita.com/chipmunk-star/items/90931fb5180b8024adcb)

### 2022/07/09
* percolationのC++実装 
[https://github.com/yonesuke/percolation](https://github.com/yonesuke/percolation)
  * 2次元bond percolationの計算をしてくれる。percolation probability / average cluster size / finite average cluster sizeが測れる。いい感じなきがする。
* こんな時間だが論文を書いた後に位相応答関数を頑張る。
* 朝にソーを見に行った。面白かったですねえ。

### 2022/07/06
* 今日も位相応答関数計算する。
* Cardy先生の中央大学での講義。
[https://arxiv.org/abs/math-ph/0103018](https://arxiv.org/abs/math-ph/0103018)

### 2022/07/05
* 2次元パーコレーションと共形場理論の関係？！わかりたすぎる〜〜〜 
[https://arxiv.org/abs/2203.08167](https://arxiv.org/abs/2203.08167)
* 理論で位相応答関数を求めるところはうまくいくけど結合関数が全く合わない。前までうまく動いてたのに全くコードが動かなくなってしまった。もう辛すぎる。位相応答関数と結合関数をあわせてGPで推定したいのにスタート地点にすら立てない。この研究向いてなさすぎ辛い〜〜〜〜〜〜〜〜〜〜もう明日の自分に期待する。こんな時間までやるのはホンマに良くない。夜ふかしして研究しても体に良くないことは博士課程を通して学んできたので。
* 昨日は東京日帰りで行ってきた。神宮とかも見てきてよかった。疲れてたのか帰ってから爆睡かましてた。
* パーコレーションでaverage cluster size $S_{L}(p)$の有限サイズスケーリングがうまく行った。適当にデータを調整して論文に載せよう。これでNNでFSS論文完成かな。ラストスパート頑張ろう。
  * 2次元bond percolationだと$p_{\mathrm{c}}=1/2,\gamma=43/18,\nu=4/3$になるらしい。


* パーコレーションの文献
  * 本: [パーコレーションの基本原理](https://www.amazon.co.jp/%E3%83%91%E3%83%BC%E3%82%B3%E3%83%AC%E3%83%BC%E3%82%B7%E3%83%A7%E3%83%B3%E3%81%AE%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86-%E7%89%A9%E7%90%86%E5%AD%A6%E5%8F%A2%E6%9B%B8-D-%E3%82%B9%E3%82%BF%E3%82%A6%E3%83%95%E3%82%A1%E3%83%BC/dp/4842702982)
  * PDF: [Introduction to Bernoulli percolation](https://www.ihes.fr/~duminil/publi/2017percolation.pdf)
  * PDF: [Percolation Theory](https://web.mit.edu/ceder/publications/Percolation.pdf)

### 2022/07/03
* ランダムグラフの最小全域木の和の期待値が$\zeta(3)$に収束することを確認した。
  * [https://github.com/yonesuke/randomMST](https://github.com/yonesuke/randomMST)
  * 最小全域木を求めるアルゴリズムはKruskal法で実装できるけどその中にもUnionFindが使われてて感動する。

### 2022/07/02
* パーコレーションの実装にUnionFindが使われてて感動した。あと、最近はなにか実装するときC++を使うようになった。なんか書いてて楽しい。

### 2022/06/29
* キャッスルのS2E11個人的に好きな回。

### 2022/06/28
* NASAのAPIを使って制限三体問題の周期軌道を適当にプロットしてみた。
[https://gist.github.com/yonesuke/cdb1603b9f69ed6e38f86b50cb68e70c](https://gist.github.com/yonesuke/cdb1603b9f69ed6e38f86b50cb68e70c)
* 今日やること。ただ親が来るからこんなに進捗は産めないかも。
  * GPflowのコードの整理
  * スパコンのコードの進捗確認
  * Levy Kuramotoの論文書く(1段落だけでも！！)
  * NASAのAPIにある制限三体問題のやつで色々遊ぶ。
    * 初期値を取り出して周期軌道の数値計算を行う。

### 2022/06/27
* スパコンにジョブを投げた。`ssh supercom ls /LARGE0/gr20114/a0144235/nsa/kuramoto | wc -l`でたまに出力ファイル数を監視しておく。トータルで84万のファイルができるのであまりよくない気がしてきた。
* 学内やと日経が無料で読めるん知らんかった。だいぶ便利やわな。

### 2022/06/26
* ヤクルトきちんと巨人戦のカード勝ってくれて嬉しい。
* 2ヶ月ランニングした成果としては安静時心拍数がだいぶ減ったのと2ヶ月前にピチピチやったスーツのズボンが昨日はスッと履けたこと。これはだいぶ嬉しい。ランニングこれからも続けていきたい。
* 昨日の結婚式はだいぶ良かった。結婚式行くたびに人生で大切なことはなんだろうか、みたいなこと考える。そして研究はその一番目ではないわな、といつも思う。
* だいぶおもろい。ボブ・サップとアーネスト・ホースト。
[https://youtu.be/vK-1m-q4etk](https://youtu.be/vK-1m-q4etk)

### 2022/06/23
* ちょっとヤクルト強すぎるなあ。こっから減速だけはせんといてほしいわな。
* 明日から代数トポロジーのゼミが始まってその初回の発表をする予定やのに一個も予習してない。まあそういうこともあるわな。
* 最近メンタル終わってて家相当汚くなってたからとりあえず掃除洗濯だけして心を新たにした。予習する。発表は基本群に関するとこ。

### 2022/06/22
* OpenMPの`#pragma omp parallel for collapse(2)`ってやつ改めてくそ便利やわ。`collapse`最近できたんかな。もっと早く知りたかった。nested for loopに対してまとめて並列化してくれる。
* Mac Studioが届いた。なんかありがとう。 
* M1 Macでの`GPFlow`のインストール方法がようやくわかった。
  * `TensorFlow`のインストールがまず大変。これが参考になった。このymlファイルで入る。
    * [https://zenn.dev/nnabeyang/articles/43f807c982715e](https://zenn.dev/nnabeyang/articles/43f807c982715e)
  * 次に`TensorFlow Probaility`のインストール。tensorflow 2.8に対応するtensorflow probabilityは0.16.0なのでこれを入れる。
    ```bash
    python -m pip install tensorflow-probability=0.16.0
    ```
  * 最後に`GPFlow`を入れたら良いんやけど、`setup.py`に書かれてるのは`tensorflow`な一方で、M1 Macに入るtensorflowは`tensorflow-macos`なのでどうやってもインストールできない。のでこのrequirementsを無視しよう。これは`--no-dependencies`オプションで実現できる。結局打つコマンドは、
    ```bash
    python -m pip install --no-dependencies gpflow
    ```
    である。これに気づくのにしばらく時間がかかってしまった。
* インストールはできたのでようやくスタート地点に立った。3ヶ月以上ガウス過程やってないので今から思い出す。
* `top`より`htop`のほうがきれいすねえ。`brew install htop`

### 2022/06/20
*  random regular graph上のODEが同期するかについての議論をまとめている。
[https://hackmd.io/@yonesuke/rylXRF6Kq](https://hackmd.io/@yonesuke/rylXRF6Kq)

### 2022/06/18
* 集中不等式がとにかく大事なので勉強していきたい。勉強していきたいというか色々応用してみたい。
* Strogatzの論文ではODEの同期とネットワークのスペクトルノルムの関係性について論じていた。特にネットワークの隣接行列$A$に対して$\Delta_{A}= A-\mathbb{E}[A]$を定義して$\|\Delta_{A}\|$について考えていた。特にErdos-Renyiグラフは隣接行列の各要素を確率$p$で$1$を立てるので独立な確率変数の和に関する形に書くことができるため、$\Delta_{A}$のスペクトルを集中不等式を通して評価することができた。しかし、今考えているのはrandom regular graphであって、各頂点から出る枝の本数を固定するため、隣接行列の各要素は独立にはならない。なので、**random regular graphにおける$\Delta_{A}$のスペクトルの評価に集中不等式を使うことができない！！** なにかうまいことを考えないといけない必要がある。

### 2022/06/17
* 多段SSHに`ProxyCommand`を使ってたけど`ProxyJump`を使ったほうがconfigがすっきりすることを知った。
* `absl`を使い始めた。`argparse`よりも好きかもしれない。
* 計算を打ち込んだので明日結果を解析して図に起こして論文にする。その間に安定分布ノイズが入った蔵本モデルの論文を書く。論文を書くぞ書くぞ書くぞ。
* あと、regular random graphのスペクトルの諸々も調べたらなんかしらの結果にはなる気がする。さっさと調べて論文にするぞ、という意気込みで研究をするぞ。

### 2022/06/15
* スノーピアサーシーズン3まで一気見した。
* 最近はapple tvの"we crashed"とdisney plusの"castle"と"Ms. marvel"を見てる
* 買いたい本
  * [An Introduction to Matrix Concentration Inequalities](https://www.amazon.co.jp/Introduction-Concentration-Inequalities-Foundations-Learning/dp/1601988389)
  * [Combinatorics and Random Matrix Theory](https://bookstore.ams.org/gsm-172)
  * [Spectra of Graphs](https://link.springer.com/book/10.1007/978-1-4614-1939-6)
  * [Stein, Elias M.; Shakarchi, Rami (2003). Fourier Analysis: An Introduction. Princeton University Press.](https://press.princeton.edu/books/hardcover/9780691113845/fourier-analysis)
  * [Stein, Elias M.; Shakarchi, Rami (2003). Complex Analysis. Princeton University Press.](https://press.princeton.edu/books/hardcover/9780691113852/complex-analysis)
  * [Stein, Elias M.; Shakarchi, Rami (2005). Real Analysis: Measure Theory, Integration, and Hilbert Spaces. Princeton University Press.](https://press.princeton.edu/books/hardcover/9780691113869/real-analysis)
  * [Stein, Elias M.; Shakarchi, Rami (2011). Functional Analysis: Introduction to Further Topics in Analysis. Princeton University Press.](https://press.princeton.edu/books/hardcover/9780691113876/functional-analysis)
  * [Lévy Processes and Stochastic Calculus](https://www.cambridge.org/core/books/levy-processes-and-stochastic-calculus/59B105C1B5B54D562AA096D7AE24F4D5)
* [https://arxiv.org/abs/2206.06776](https://hackmd.io/@yonesuke/rylXRF6Kq)
  * Vicsekモデルを確率過程として定式化してPDEにして色々計算している。これは求めていた方向の無限次元化なので必読な気がする。
* [https://arxiv.org/abs/2206.02768](https://arxiv.org/abs/2206.02768)
  * NNの幅無限の極限の解析にSDEを使ってる。NNGPになる話は有名で自分も何回か計算したけどその方向の無限の飛ばし方は現実との乖離があるのはそうで、この論文やとSDEとして色々計算してる。相関係数も実験とよく一致している図がある。
* [https://arxiv.org/abs/2105.06509](https://arxiv.org/abs/2105.06509)
  * ポテンシャルに特異点がある系の粒子数無限の極限を求める論文。正則な場合にはVlasov方程式になるのはよく知られているけどそうじゃなくても数学的に頑張って証明している。これは博論っぽい。自分ももう少ししたら書かなあかんのか〜〜〜〜〜〜〜〜〜〜
* 集中不等式の本を買った。集中不等式オタクになるわな。

### 2022/06/10
* 研究費の公正使用のe-learningをやる。
* $\pi_{1}(\mathbb{S}^{1})\simeq\mathbb{Z}$を被覆空間を通して証明する流れがだいぶすっきりわかった。homotopy/homologyゼミも始まるので楽しみ。
* strogatzの最近の論文とにらめっこした。
[https://arxiv.org/abs/2203.03152](https://arxiv.org/abs/2203.03152)
  * 定理11がかなり本質的で、例えばランダム正則グラフへの拡張についてもこの定理を用いると同期の判定がかなりやりやすくなる。
  * スペクトラルグラフ理論と力学系とのつながりが見えて面白い。この定理の精緻化を考えるのも面白い気がする。
  * とにかくstrogatzに勝ちたいわな。

### 2022/06/07
* とりあえず作った。
[https://github.com/kyoto-nd-lab](https://github.com/kyoto-nd-lab)
  * 研究室の人らでの共同の作業があったらここで行うことにする。
* 昼からC++を立川くんと書いてた。
[https://github.com/kyoto-nd-lab/network-connectivity](https://github.com/kyoto-nd-lab/network-connectivity)
  * あとはひたすら回すだけ。良いネットワークが見つかるといいな〜〜
  * まじでうまく行かんかった。まあこういうことの繰り返し。もう一年以上この問題のことを考えてる気がする。
  * 明日夕方にスパコンにブチ込むぞ。

### 2022/06/06
* 科研費が使えるようになったので早速本を色々と買ってみた。嬉しい。みなさんの税金で買った本と言っても過言ではないので頑張って勉強します。

### 2022/06/04
* 夜ランから朝ランに移行した。
* ホモトピーとホモロジーの良さげなpdf [http://www.math.uchicago.edu/~may/CONCISE/ConciseRevised.pdf](http://www.math.uchicago.edu/~may/CONCISE/ConciseRevised.pdf)
* 高校同期の起業アイデアを聞いてた。みんなブロックチェーンとか言ってる。それブロックチェーン使わなできへんのか？と思ってまうわな。
* ハケンアニメめっちゃ良かった！！アニメ全然見てないけど感動した。アニメ見てる民の感想も気になる。

### 2022/06/01
* 朝6時から研究とかしてた。偉すぎる。けどさすがに疲れた。
* 子供のときアメリカに住んでてずっとスポンジボブを見てたんやけどその中でも印象に残ってたエピソードがあってそれを久しぶりに見直して感動した。見てほしい。
[https://www.dailymotion.com/video/x6vi1em](https://www.dailymotion.com/video/x6vi1em)

### 2022/05/31
* 論文書いてる。
* Landau damping読み始めた。
* 明日アメリカでポスドクしてるブラジルの民と話すことになった。世界中に蔵本モデルの民がおるんやわな。

### 2022/05/29
* トップガンだいぶおもろかった。
* stranger thingsのseason 4 vol. 1が出たのでイッキ見してしまった。

### 2022/05/27
* テンソルネットワークの発表を聞くたびに2d isingの臨界現象を計算してみようって思ってるけど毎回やってない。
* トップガンを見に行くぞ。

### 2022/05/26
* ユニバ行ってきた。楽しすぎるなあ。研究とは。
* ssh越しにファイルを同期する[sshync](https://github.com/mateogianolio/sshync)を知った。
  * nodeでインストールする。m1 macに[volta]()ってやつでnodeを入れた後にsshyncをインストールした。

### 2022/05/23
* 土曜日にシン・ウルトラマンを見に行った。
* [http://blogs.perl.org/users/smylers/2011/08/ssh-productivity-tips.html](https://www.dailymotion.com/video/x6vi1em) sshのテクニックがたくさん載っている。
* `openssl`コマンドで色々遊んでみたいですねえ。

### 2022/05/20
* これだいぶ便利そう！！
[https://serverfault.com/questions/995103/change-ssh-ip-based-on-current-network](https://serverfault.com/questions/995103/change-ssh-ip-based-on-current-network)
* githubのmarkdownでtexが使える！！
[https://github.blog/2022-05-19-math-support-in-markdown/](https://github.blog/2022-05-19-math-support-in-markdown/)



### 2022/05/19
* 博士同期の結婚祝いを兼ねてトリキに行った。人生を感じますね〜〜〜就職先も一緒やからこれからも頑張っていきたいすねえ。
* ランニング1ヶ月やって心拍数がだいぶ落ちてきた。痩せたいなあ。

### 2022/05/18
* なるほどなあ。。。
[https://yuri.is/not-julia/](https://yuri.is/not-julia/)
* 高校の友達に呼ばれて10時から1時まで四条で飲んでた。無機質な博士課程の生活にかすかな彩りを与えてくれた友達に感謝。
  * [https://tabelog.com/kyoto/A2601/A260301/26020827/](https://tabelog.com/kyoto/A2601/A260301/26020827/) 2軒目にここに行ったんやけど深夜まで空いててめちゃめちゃ美味しかったのでおすすめ。今度誰か行きましょう。
* 今日は論文の数値実験のパートを書くぞ〜〜ここが終わったらあとはconclusionとabstractだけでひとまず終わり。早く終わらせてSDEに本腰入れたい。
  * tikzでneural netwok [https://tikz.net/neural_networks/](https://tikz.net/neural_networks/)
  * notebook [https://gist.github.com/yonesuke/a2393772669a620cba3cc54ce892f5af](https://gist.github.com/yonesuke/a2393772669a620cba3cc54ce892f5af)

### 2022/05/17
* 今日もちまちま論文書いてちまちまSDEの勉強した。こういったちまちました作業を粘り強くやっていくこと学ぶ博士課程になっている。
* 奨学金返還のリレー口座登録のために銀行まで行った。

### 2022/05/14
* 昨日友達からビットコインの実装本を譲り受けた。
[https://www.oreilly.co.jp/books/9784873119021/](https://www.oreilly.co.jp/books/9784873119021/)
* そんな大した数学も出てこないのでスラスラ読める。昨日からpythonでひたすら実装してる。トランザクションもできそうなので実際に暗号通貨を買ってなにかしてみたい。
* 暗号通貨の取引の秘匿性を担保するために楕円曲線暗号が使われていてそれの実装もした。楕円曲線の話は全く初めてやったけどかなり理解が深まった。公開鍵認証方式も実装して署名することもできた。今までssh-keygenで勝手にやってくれていたこともなんとなくわかる気がしてきて感慨深い。
  * フェルマーの小定理は大事な定理やとは思ってたけど楕円曲線暗号でここまで使われてるとは思わんかった。使われてるというか$F_{p}$上の要素の割り算の計算に必要ってだけやけど、それでもなんか感動してる。
* 実装: [https://github.com/yonesuke/programmingbitcoin](https://github.com/yonesuke/programmingbitcoin)

### 2022/05/13
* 13日の金曜日。
* SDEまじで勉強していく。頑張ろう〜〜

### 2022/05/12
* 大谷悪いながらもQSやっててさすがやった。頑張ってますねえ。
* 力学系ゼミの初回を行った。色々実装しながらやってみたけど面白かった。セパラトリクスの解とかも見ることができた。
* 家族がtimetreeを使ってるけど俺はgoogle calendarを使ってるからいい感じに連携させたい。timetreeはicalに対応してないからなのか、exportできなくて悲しい。幸いtimetreeはapiを提供してくれてるから10秒おきにtimetreeにアクセスしてapiを通してデータを抜き取ったあとgoogle calendarに投げることをすれば良さそう。
* と思って実装したんやけどtimetreeのapiが7日先の予定までしか拾ってくれないクソ仕様で悲しくなった。

### 2022/05/11
* 朝に人事の民と電話した。返事をしなあかん。来年から東京やな〜〜
* 今日も論文。

### 2022/05/10
* 今年もBSのエンゼルス戦見たあと中央食堂で飯食って研究室に行くのが定番になってきた。大谷の満塁ホームランすごかった。
* 今日も論文論文論文！！
* 学振関連でなんか忘れてへんか不安になるけど多分大丈夫の精神でやっていく。

### 2022/05/08
* 論文論文論文！！！
* todoistっての使ってみてる。悪くないのかもしれない。

### 2022/05/04
* 朝６時に起きてstrogatzの講演を聞いた。自分が去年出した論文の話だった。自分の名前があったのが嬉しかった。
* そのあとドクター・ストレンジを見てきた。ネタバレしちゃうのでここでは何も書かないでおく。
* 昼寝したので少しランニングして論文を書く。
* 最近博士の友達と話してて博士って何なんやろうかって思うようになってきた。ドイツに留学してAIの研究してる友達は論文を書かないと行けないプレッシャーで泣いてるらしい。正直異常やろと思ってしまう。華やかなAI研究の裏で心がしんどくなってる学生がいっぱいいてそれもめちゃめちゃ悲しい。

### 2022/05/03
* 研究室にきて論文書いた。偉いなあ。
* 明日はドクター・ストレンジの公開日。チケット取ったので相当楽しみ。

### 2022/04/30
* GW一生映画見てる。楽しすぎるなあ。
* 関数列の収束についてまとめた。
[https://yonesuke.github.io/posts/convergence/](https://yonesuke.github.io/posts/convergence/)
* 論文も少し書いた。偉すぎるなあ。
* 局所コンパクト群の持つ性質の素晴らしさを反芻している。必ず左Harr測度を持つから、例えばコンパクトな線形リー群が完全可約になることを示せる。リー群ゼミ楽しい。あとは実数上のルベーグ測度の特徴付として$\mathbb{R}$を平行移動を群作用とする局所コンパクト群として測度を構成する話とか。コンパクト群に関する話が詳しい本として[群の表現論序説](https://www.iwanami.co.jp/book/b265391.html)がある。第0章が学生と教授の会話の形式で話が進んでいくんやけど学生が物分りが良すぎるのが笑ってしまう。
* 今までリー代数のことをリー環と呼ぶのがなんだか嫌やった(掛け算の単位元$1$がなくね？)んやけど、Banach環も本当は環ではないっぽい(要確認)のでそういうものなんかなあと思い直してリー環という言葉も許すことにした。新しいことを知ることによって心が広くなっていく。
* 例えば局所コンパクト群$G$に対して$L^{1}(G)$は畳み込み積に関して複素のBanach環になるらしいけど、一般に$L^{1}(G)$は$1$を持たないのでこれはBanach環って言葉の割りに環っぽくない例の1つになっている。けどやっぱり$1$が存在しないのは色々やりにくいっぽいので、空間を広げて$1$を付け加えたり、あとは*近似的な$1$*ってのを見出すことがあるらしい。例えば局所コンパクト群$G$に対して複素のBanach環$L^{1}(G)$は強い意味で近似的な$1$を持つ、らしい。

### 2022/04/26
* 研究費の交付申請書とかいうのを書かないと行けないらしい。何回申請書書かせるねん。これは先生方の時間泥棒やなあと思ったり。
* 今年度は90万円、来年度は80万円をありがたく使わせていただけるらしい。まあ来年度はいないから意味ないんやけどね〜パソコン買って本を大量に買っていくか。

### 2022/04/21
* 今朝またまた大谷が無双してた。俺も勇気づけられる。
* リー群ゼミの発表やった。$\mathsf{su}(2)$の既約表現を求める部分は個人的に面白いと感じたので細かいとこまで説明した。調和振動子も同じような既約表現の実現の一つとして表せるようにも思えるけども決定的に違うのはエネルギー固有値が可算無限個あること。今やってるゼミの本は有限次元の既約表現しか扱っていないので範疇ではないんやろうなあ。無限次元のリー代数の表現論を学ぶモチベにつながる。[Oscillator representation](https://en.wikipedia.org/wiki/Oscillator_representation)あたりが知りたいことなのかもしれないけどきちんと読まないとわからない。
* SPY FAMILYの１話にシンプレクティック同相写像が出てきてテンションがあがってので見ていくことにする。アニメを追いかけるのは小学生以来な気がする。みんなが見てるからって理由だけでポケモンを見てたのと、日曜の朝に仮面ライダーか何かのあとがプリキュアで妹と惰性で見てたのを覚えてる。

### 2022/04/16
* 朝起きたら大谷が打っててワクチン打って帰ってきたらまた大谷が打ってた。さすが。
* 今日明日は安静にするのでリーマンゼータ関数の本をパラパラ読むことにする。

### 2022/04/15
* 「未来への10カウント」を見た。キムタク相変わらずイケメン。
* バーゼル問題$\sum 1/n^{2}=\pi^{2}/6$の歴史を色々聞いた。そこからリーマンゼータ関数$\zeta(s)=\sum 1/n^{s}$やそのリーマン予想につながる話が面白かった。やっぱ数学ロマンがあっておもろいなあ。
* リーマンゼータ関数の数値計算に関してこんなアルゴリズムがあるらしい。
[https://en.wikipedia.org/wiki/Odlyzko%E2%80%93Sch%C3%B6nhage_algorithm](https://en.wikipedia.org/wiki/Odlyzko%E2%80%93Sch%C3%B6nhage_algorithm)
気が向いたら実装したい。

### 2022/04/14
* ここ数ヶ月何もしたくない状態が続いていて研究をあまりしていなかった。何もしていない状況が嫌で、でもなにかしないと流石にまずいっていう罪悪感があったから好きな数学の勉強でもしよう、みたいな感情がずっと続いていた。けどまあ流石に卒業したいから頑張るかみたいな気持ちで今日はたくさんコードを書いた。自動微分を使って位相応答関数を色々求めることができるパッケージを整備した。だいぶ便利になった気がする。自動微分にはまじで感謝してる。ありがとう。
* 明日は論文を書くのとガウス過程回帰で結合関数を推定するやつを色々とまとめていく。目指せ卒業。
* Netflixのクイーンズ・ギャンビットをN週目してる。だいぶ好き。自分の中のいつ見てもおもろいコンテンツリストの一つに入っている。

### 2022/04/13
* Twitter上でのシニアと若手研究者のレスバを見てると虚無い感情を抱きますねえ〜〜
  * [https://note.com/sendaitribune/n/nf7f0a975486f](https://note.com/sendaitribune/n/nf7f0a975486f)

### 2022/04/10
* やる気がでなさすぎて一生リー群の問題だけ解いてる
* 昨日はモービウス見に行って面白かった。sonyのスパイダーユニバースがこれから広がっていくんやろうなあ。
* 選挙行ってやる気出たらコード書く。
* 　研究日誌というタイトルでありながら中身は研究のこと全然書かない日記になってしまっている。だが気にしない。日記を書いてみようみたいなことを修士のときにも考えて一時期やってたけどすぐ飽きてしまっていたので、今回半年近く続いてるのは自分の中でも衝撃。ドクター終わるまでは続けていこう。
* 　今日のNHKスペシャルで$abc$予想の特集が組まれていて見ざるを得なかった。$abc$予想の証明がされたってニュースは自分が高2のときに流れてきてその当時は足し算と掛け算の関係がうんたらとかいう解説を聞いてもふーんって感じだった。同級生の数学大好き友達(この友達は京大理学部に入って数学科に進学した。登山部に入って留年してたけど今は何をしているのだろうか。)が$abc$予想の話を聞いて「この予想が正しかったらフェルマーの最終定理がある数よりも大きい$n$でかんたんに示すことができる」って言っててすごいんやなあと思ったのを覚えている(というか正確には自分にはちんぷんかんぷんな話を友達はすぐさま理解して色々その先を考えることができているということに対する羨望というか嫉妬というかそういった感情を抱いたことを覚えている。)

  $abc$予想の証明に対する疑義はずっとついて回っているらしいけどRIMS側が出版するに至っているというのが今の状況なのかな。(RIMSの論文誌PRIMSに投稿して出版されたらしいけどPRIMSの編集長自身が望月さんっていうのはどうなんだろうか？という気持ちにはさせられる。)ピーターショルツさんが論文の疑問点を洗うために1週間京都に滞在して望月さんと議論したけど結局議論は平行線のままやったらしい。そのときの議論をまとめたノートが[Why $abc$ is still a conjecture](https://ncatlab.org/nlab/files/why_abc_is_still_a_conjecture.pdf)として出ている。書いてあることは一ミリもわからない。ちなみにピーターショルツさんは30歳でフィールズ賞を受賞した天才で、$abc$予想の論文が出たときは24歳でその時点で論文に欠陥があることを見抜いたらしい。(自分が24歳のときに何をしていたかを思い出すとまた絶望するのでそれは考えないことにする。)ここらへんの経緯は[これ](https://tar0log.tumblr.com/post/648055627348869120/2018%E5%B9%B49%E6%9C%8820%E6%97%A5quanta-magazine-titans-of-mathematics)が詳しい。ちなみに金曜5限の現代の数学と数理解析って授業に今潜ってて、その授業は数理解析研究所の教員たちが学部生向けに最先端の研究を紹介してくれるっていう神講義になっている。先週その第一回の講義に参加したら雑談の中で「最近だと$abc$予想が証明されたって話がありますが、$abc$予想はちょっと危険なのでここでは話さないでおこうと思います。」って言っててそこはかとない闇を感じた。

### 2022/04/08
* 大谷の開幕を見届けたが負けてしまった。
* 位相応答関数を自動微分を使って求めるプログラムを書いた。何回も使ってるコードはパッケージ化していきたい。
  * [https://github.com/yonesuke/prax](https://github.com/yonesuke/prax)
* 就活が終わっている気がする。というか何もしていないのでぬるっと終わっていった感じがする。あと学振ははよ給料くれ〜4月分は5月に回されるの悲しい。

### 2022/04/06
* タオの測度論のゼミがようやく終わった。M2のときからやっていたのでちょうど3年かかったことになる。演習問題も全部解いたのでかなり力がついた気がする。
* それに合わせて関数列の収束の大小関係についてブログにまとめた。
[https://yonesuke.github.io/posts/convergence/](https://yonesuke.github.io/posts/convergence/)
* tikzをはじめてまともに使った。svg出力してブログに埋め込んだ。それぞれの矢印の証明についてもまとめたい。
* 複素関数が正則であることの定義が本によって違う(微分可能性を入れるか入れないか)のとか毎回頭こんがらがってるから色々とまとめたいなあ。

### 2022/04/01
* 今日は大阪まで家族とご飯を食べに行った。
* あと人生で何回親に会えるのかとか考えると悲しくなった。
* 行き帰りの電車の中で引き続きリー群の問題を解いた。明日書き下そう。
* さくらが綺麗やったからここにも貼っちゃう。

### 2022/03/31
* D2最後の日。
* 講究録を無事に提出した。
* やる気が出なかったけど手を動かしたかったのでリー群の問題を解いた。[第1章](/seminar/2022リー群/第1章)が終わった。

### 2022/03/30
* 立川くんのおかげでtikzjaxが動くようになった！！！マジ感謝。
[https://yonesuke.github.io/posts/tikz/](https://yonesuke.github.io/posts/tikz/)
* webassemblyなんかやってみたくなるねえ。
* 研究室見学に来た京都工芸繊維大学の学部生が桐蔭出身やってまじでテンション上がった。
* 明日締切の講究録結局殆ど書いてないから今日は徹夜で頑張る。
* 明日桐蔭甲子園優勝してくれ。頼む。
* 東芝の新社長のインタビューを読んだ。山本くん頑張れ。 
[https://business.nikkei.com/atcl/gen/19/00387/032800003/](https://business.nikkei.com/atcl/gen/19/00387/032800003/) 


### 2022/03/27
* marpでスライド作ってみた。いい感じかもしれん。beamerよりはものを書くことに集中できる気がする。

### 2022/03/25
* [https://en.wikipedia.org/wiki/List_of_named_matrices](https://en.wikipedia.org/wiki/List_of_named_matrices) これ眺めてるだけで時間過ごせる。

### 2022/03/23
* [https://sha256algorithm.com/](https://sha256algorithm.com/)
  * sha256の流れがめっちゃわかりやすくまとまってる。時間あったら自分でも実装したい。

### 2022/03/22
* [https://yonesuke.github.io/posts/egorov/](https://yonesuke.github.io/posts/egorov/) エゴロフの定理の証明をまとめた。どの本も$E_{N,m}$って集合をいきなり定義しててわけわかめやったけど山本くんと話してるうちに理解したのでそれを書いただけ。
* github actions死んでる気がする。
* wikiではmermaidが書けないのか。
    ```mermaid
    graph TD
        A[/一様収束/] --> B[/各点収束/] --> C[/概収束/]
        A --> D[/ $L^\infty$収束 /] --> E[/概一様収束/]
        D -. 有限測度 .-> F[/$L^1$収束/] --> G[/測度収束/]
        E --> C
        C -. 有限測度 or 優関数 .-> E
        E --> G
        C -. 優関数 .-> F
        D -. outside of a nullset .-> A
    ```

### 2022/03/21
* 今週すること
* 結合関数GP推定
  * gpraxをまとめる
  * adjoint法による位相応答関数推定のjax実装をまとめる
  * 具体的なシステムでgp結合関数推定を行い、まとめる
* 臨界指数NN推定
  * 蔵本モデルで臨界指数をきちんと求める
  * 論文書いていく
* 色々な研究の終わりが見えてきたので次の研究に向けて色々考えてみる
  * 同期しない密なネットワークの探索を何も考えずにjobを作る
* 今週も頑張る

### 2022/03/17
* 朝からコーディングテスト。疲れる。
* [https://devcenter.heroku.com/articles/heroku-postgresql](https://devcenter.heroku.com/articles/heroku-postgresql)
* これを使うようにしたので、pushするたびにスレのデータが消えることはなくなった。

### 2022/03/16
* apiの使い方を学ぼうということでひろゆきに対抗して掲示板作った。
* [https://yoneda-heroku-test.herokuapp.com/board/](https://yoneda-heroku-test.herokuapp.com/board/)
* texが打てる！！
* おれのいろんな界隈の友達が掲示板で喋ってて感動する。
* みんなsqlインジェクション頑張るの笑う。

### 2022/03/15
* dockerで立ち上げているoverleafにはデフォルトで全然パッケージが入っていないので手で入れないといけない。有名パッケージを一括で入れてくれるコマンドを入力した。
  * 参考: [https://blog.tea-soak.org/2021/01/overleaf.html](https://blog.tea-soak.org/2021/01/overleaf.html)
* $[0,1]$区間の一様乱数$X,Y$に対して$X/Y$に一番近い整数が偶数になる確率が$(5-\pi)/4$になるらしい。その証明をした。
  * [https://yonesuke.github.io/posts/uniform-ratio-pi/](https://yonesuke.github.io/posts/uniform-ratio-pi/)

### 2022/03/12
* Łojasiewicz exponentってなんなんか知りたい。蔵本モデルの解析に使ってる論文があった。
[https://doi.org/10.1063/1.4908104](https://doi.org/10.1063/1.4908104)
* 学振の採用手続きと授業料免除の申請の準備をしなあかん。

### 2022/03/11
* バットマン見てきた。おもろすぎる〜〜〜〜
* 関数列が$L^1$関数で抑えられているならば、各点収束列は$L^1$収束列になるのが優収束定理やけど、各点収束列ならば概一様収束列になることも言えるらしい。証明はエゴロフの定理とほぼ同じ。というかエゴロフの定理は有限測度空間で各点収束列が概一様収束列になることやから似たようなことを言ってる。

### 2022/03/08
* 就活〜〜
* wandb便利すぎワロタ

### 2022/03/01
* エントリーシートとかいうのを書いてると一瞬で時間が溶けていく。研究〜〜
* Lie群ゼミはいい調子で進んでいる。時間を見つけて演習問題を解いていきたい。

### 2022/02/21
* 結婚式の招待状の返信を書いた。謎の文化があるのを知らなかったのでググってよかった。
[https://www.weddingpark.net/magazine/8225/](https://www.weddingpark.net/magazine/8225/)
* vicsekモデルの先行研究の論文を読む。
  * [ここ](/user/yoneda/vicsek)にまとめてる。
  * angular noiseとvector noiseがあってvector noiseのほうは有限サイズで不連続転移がよく確認できる。angular noiseは有限サイズの影響が強くて連続転移っぽく見えるけどbinder ratio見てると不連続転移になっているのがわかる、というのが結論なんだろうなあ。

### 2022/02/16
* vicsekモデルの数値計算が終わったのでhuggingfaceにアップロードする。
* スプレッドシートを埋め込むテスト
* `<iframe src="https://docs.google.com/spreadsheets/d/1_WUK6VwAiCx4PJSu6JX_bq1tM4tRBEklO_elbMuE9BQ/edit?usp=sharing" style="width:90%;height:590px;border:0"></iframe>`

### 2022/02/15
* 今日やること
  * GP回帰で結合関数推定
    * 3体の場合の結合関数推定を頑張る
    * kernelの設計の話をかんたんにまとめる
  * NNで臨界指数推定
    * vicsekの実装をスパコンにあげて走らせる
    * huggingfaceにデータをアップロードして読み込めるようにする
    * (jaxだとtfdsを使うのが良さそう？？)
* dysonにnextcloudが立ち上がってるのでそれを手元のパソコンからいつでもアクセスできるようにした。
  * ローカルに`autossh`をインストールする。一度途切れたssh接続を自動で再接続してくれるらしい。
  * 外部からdysonにアクセスできるように`~/.ssh/config`に次のように書き込む。
    ```bash
    Host dyson-nextcloud
      HostName 192.168.1.20
      User yoneda
      ProxyCommand ssh -W %h:%p remote-pascal
      LocalForward 2345 localhost:8090
    ```
    ここでは`remote-pascal`経由で`dyson-nextcloud`に接続できるようにした。また、nextcloudはdysonの8090番に立ち上がってるのでそれをローカルの2345番からつながるようにした。
  * 最後に`autossh`で接続する。
    ```bash
    autossh -fN -M 0 dyson-nextcloud
    ```
    接続を切る場合は
    ```bash
    killall autossh
    ```
    本当はパソコンの起動時にこのコマンドを走らせるようにしたかったけどそこまではしてない。
  * [localhost:2345](localhost:2345)に接続するとnextcloudが立ち上がる。便利。

### 2022/02/14
* 朝に面接を受けた。
* M1 Macでのtensorflow環境を用意してくれてるdockerないの〜〜
* 京大スパコンでsingularity使えるから色々試してみてた。

### 2022/02/12
* 作った。
[https://github.com/yonesuke/RationalNets](https://github.com/yonesuke/RationalNets)

### 2022/02/05
* JAXを推していきたいのでJAXで蔵本モデルを実装した。
  * [https://gist.github.com/yonesuke/9e96ffbe8f35b34683d255eac5bbff0e](https://gist.github.com/yonesuke/9e96ffbe8f35b34683d255eac5bbff0e)

### 2022/02/02
* 午前は書類を片付ける
* 密なネットワークを決めて安定な平衡点を探索する作戦を考えたが失敗した。
  * [Dense Networks That Don’t Synchronize.pdf]
  * [JAX実装](https://gist.github.com/yonesuke/92c88214e0bb12646b608bad8f980df9)

### 2022/02/01
* jaxでNNとかの最適化を組むときはoptaxがいい感じっぽくてずっと使ってたけどjaxoptってのも最近出ててそれはどうなんだろうか。optaxのラッパーみたいな感じで使えるものもあるっぽい。あとLBFGSが入ってるのがでかい。
  <blockquote class="twitter-tweet"><p lang="en" dir="ltr">JAXopt v0.3 has been released! New features: LBFGS, nonlinear conjugate gradient, and Gauss-Newton algorithm! <a href="https://t.co/PyHX081tMe">https://t.co/PyHX081tMe</a></p>&mdash; Mathieu Blondel (@mblondel_ml) <a href="https://twitter.com/mblondel_ml/status/1488166454662946820?ref_src=twsrc%5Etfw">January 31, 2022</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

### 2022/01/31
* 2022年なのに今気づいた。これまでの日付がミスっているが直さない。
* これから対応していかないといけない書類一覧、めんどくさい。
  * [02/07]　科研費(学振)のやつ　
  * [02/10]　奨学金免除申請のやつ
  * [03/11]　来年度授業料免除申請のやつ 
  * [3月末]　学振のやつ
* 今日やること
  * 上の書類のいくらかに目を通す
  * JAXでExactGPとそのBatch学習を実装する、そしてそれのSparseGPとの比較をする
  * Vicsekどうするか考える
* Cholesky分解とかを勝手に微分してくれて感動してる。
  * Cholesky分解の微分に関しては[これ](https://arxiv.org/abs/1602.07527)がまとまってる。
  * 行列の自動微分あたりに関しては[これ](https://arxiv.org/abs/1710.08717)も見つけた。具体例の中にSparseGPも乗ってた。

### 2022/01/28
* ついにメカニカルキーボードを買ってしまった。カチャカチャ言わせて論文を書いていく。
* critical connectivity関連のやりたいことが乱雑しているのでそれを簡単にまとめておく。

### 2022/01/25
* JAFEEと日本物理学会、自分が発表するわけではないが名前が入っている。
  * [http://www.jafee.gr.jp/01rally/conference/pro_56th_2022_0121.pdf](http://www.jafee.gr.jp/01rally/conference/pro_56th_2022_0121.pdf)
    * デジタルおよびバリアオプションに対する Deep Hedging の学習性能の評価, 樋口 智英(野村證券), 米田 亮介(京都大学大学院情報学研究科), 小林 望(野村證券), 藤田 政文(ノムラインターナショナルPLC)
  * [https://onsite.gakkai-web.net/jps/jps_search/2022sp/data2/html/program11.html](https://onsite.gakkai-web.net/jps/jps_search/2022sp/data2/html/program11.html)
    * ニューラルネットワークを用いたスケーリング解析手法, 米田亮介, ○原田健自 (京大情報)
* C++でガウス過程回帰の実装もしたほうが良さそう、pythonだと時間がかかってしまう。

### 2022/01/24
* 実装できた。適当に挙動を見たあとMPI実装もする。
* [https://gist.github.com/yonesuke/6e7b7090ccf5053e1545102a4c9a5f35](https://gist.github.com/yonesuke/6e7b7090ccf5053e1545102a4c9a5f35)

### 2022/01/23
* c++でvicsekを実装するぞ

### 2022/01/21
* 昨日イカゲームをイッキ見したので眠い。
* 正則関数という日本語はあまり適切ではないと常々思っている。英語との対応がややこしいことになっている。

### 2022/01/20
* wordleってのが最近流行ってるのでやってみる
  * 探索の初手でappleって売ったけどpを2回使うのはもったいないので別の初期値を取ったほうが良いかも

### 2022/01/18
* HuggingFaceの使い方わからんすぎわろた
  * [ここらへん](https://huggingface.co/docs/datasets/dataset_script.html)を参考にしてコードを書かないといけない
* 分散が不均一なデータのことをheteroscedasticということを学んだ。
  * Gaussian negative log likelihoodを損失関数に取る
    * pytorchなら[これ](https://pytorch.org/docs/stable/generated/torch.nn.GaussianNLLLoss.html#torch.nn.GaussianNLLLoss)
    * optax,jaxにはない

### 2022/01/15
* HuggingFaceがかなり良さそう。ここにkuramotoとかisingとかviscekのデータを載せていつでも有限サイズスケーリングできるようにする。
  * [https://huggingface.co/datasets/yonesuke/kuramoto](https://huggingface.co/datasets/yonesuke/kuramoto)
* Neural ODEいよいよやっていくぞ！！
  * Lotka Voltera方程式のパラメーター推定は驚くほどうまく行った。振動子の場合もやってみる。

### 2022/01/12
* jax君が勝手に微分してくれるから随伴方程式を解くのがだいぶ楽になった。ありがとうjax。位相応答関数を求めることができた。
* `jax.lax`の便利関数何回も調べてるのでまとめる。
  * `while_loop`
    ```python
    def while_loop(cond_fun, body_fun, init_val):
      val = init_val
      while cond_fun(val):
        val = body_fun(val)
      return val
    ```
  * `cond`
    ```python
    def cond(pred, true_fun, false_fun, *operands):
      if pred:
        return true_fun(*operands)
      else:
        return false_fun(*operands)
    ```
  * `fori_loop`
    ```python
    def fori_loop(lower, upper, body_fun, init_val):
      val = init_val
      for i in range(lower, upper):
        val = body_fun(i, val)
      return val
    ```

### 2022/01/11
* spiking neuron君たちはかなり高速化に成功した。
* jaxのarrayの要素の更新について
  * [https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ndarray.at.html#jax.numpy.ndarray.at](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.ndarray.at.html#jax.numpy.ndarray.at)

### 2022/01/07
* spiking neuronのネットワークを作って時間発展をさせたが時間がかかりすぎるのでいい感じに高速化したい。
* critical connectivityのネットワーク探索のプログラムもはやく書きたい。
* スパイダーマンはよ見に行きたい。

### 2022/01/06
* 八坂神社に初詣に行った。
* neural odeのアイデアを使った妄想をした。上手くいくかはわからない。
* google driveの15GBが厳しいのでnextcloudに全部移行したい。raspberry piでかんたんに動かせるらしい。

### 2022/01/04
* あけおめ
* 気が向いたらVicsek Modelを実装する
  * ノイズの入れ方によって一次転移と二次転移に分かれるらしい。
  * [http://web.mit.edu/8.334/www/grades/projects/projects10/Hernandez-Lopez-Rogelio/dynamics_2.html](http://web.mit.edu/8.334/www/grades/projects/projects10/Hernandez-Lopez-Rogelio/dynamics_2.html)

### 2021/12/29
* strogatzの漸近解析の授業repo [https://github.com/arkabokshi/asymptotics-perturbation-methods](https://github.com/arkabokshi/asymptotics-perturbation-methods)
  * 別ブランチにGitHub actionsで生成されたPDFも貼ってくれてる、ありがとう。
* 今年を振り返ってみると研究らしい研究をほとんでできてないなあ、となって悲しい。
* 来年はドクター最後の年なので頑張りたい。
* 論文を絶対に書く。
  * Levyノイズが入った蔵本モデルの理論解析
  * NNで臨界指数求める
  * GPで結合関数推定
* そして論文を読んで新しい研究を始める。
  * critical connectivityあたり
  * NN/GPでPDE学習

### 2021/12/23
* [https://github.com/google/jax/pull/6053](https://github.com/google/jax/pull/6053)
* ニューラルネットワークで臨界指数学習させるの無事にうまく行った。おめでとう。
  * Binder ratio
  * 磁化
  * susceptibility

### 2021/12/22
* 研究してる〜〜
* jaxで2秒で終わってたコードをpytorchに書き直したら30秒かかって草
* Dataloaderとか使いたいと思ってたけどjaxでdataloader使うサンプルコード上がってたから耐えそう
  * [https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html](https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html)

### 2021/12/14
* インターン終わった〜〜〜〜嬉しすぎる〜〜〜〜〜
* 明日から研究頑張ろう

### 2021/12/09
* インターン8日目終わり
* 労働は非常に疲れることを実感する
* 相当研究滞ってて耐えてない

### 2021/11/30
* [Gallery of named graphs](https://en.wikipedia.org/wiki/Gallery_of_named_graphs)
* 量子渦のjulia実装 [https://docs.qojulia.org/examples/vortex/](https://docs.qojulia.org/examples/vortex/)
* 今日もインターン
  * tensorflowのネットワークを色々いじれるようになった。
  * 金融のことわからんすぎるんよなあ。
* 研究のto-doが溜まってるのでどこかでやりきりたい
  * ガウス過程回帰で結合関数推定
    * 古川くんのODEを調べる
    * spiking neuronsのODEの実装とその結合関数推定
    * gpflowの実装をきれいにまとめる
  * ニューラルネットワークで臨界指数
    * 論文の続きを書く
      * 数値計算結果の図を作る
      * 臨界指数のテーブルを作る
  * 同期しないネットワークの探索
    * 論文を読む
      * [2111.02568](https://arxiv.org/abs/2111.02568)
      * [2111.10059](https://arxiv.org/abs/2111.10059)
     * 色々考えよう

### 2021/11/28
* [Deep hedging](https://arxiv.org/abs/1802.03042) だいぶわかってきた。
  * [pytorch実装](https://github.com/pfnet-research/pfhedge), [tensorflow実装](https://github.com/YuMan-Tam/deep-hedging)がある。

### 2021/11/27
* ガウス過程回帰しようとして固有値が小さすぎてコレスキー分解ができないのN回目で発狂しそう。
* とりあえずガウス過程回帰を使った結合関数推定の進捗を貼っておく。
* ヤクルト日本一！！！！！

### 2021/11/24
* インターンずっと続いてて研究する時間まじでないなあ。ミスった気がする。
* 今週はもうインターンはないので研究する。進捗を生むぞ。
* [TensorFlow確率におけるガウス過程回帰](https://www.tensorflow.org/probability/examples/Gaussian_Process_Regression_In_TFP?hl=ja) は参考になりそう。

### 2021/11/16
* 今日は１日論文を書いた。ここまでの進捗を貼っておく。
* あと山本くんの書きかけの論文を読んだ。頑張れ。
* あとjaxでGPU番号の指定の仕方とメモリの開放の仕方を学んだ。かなり有用な情報。
  ```python
  import os
  import jax
  
  ## デバイスの指定
  os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
  os.environ["CUDA_VISIBLE_DEVICES"]="2"
  ## メモリの開放
  backend = jax.lib.xla_bridge.get_backend()
  for buf in backend.live_buffers(): buf.delete()
  ```
* 次のインターンに向けた論文を読まないと行けない。それも頑張らなあかん。
* もうちょっと書き進めた。

### 2021/11/12
* 1週間のインターンが終わって京都帰還
* 疲れた

### 2021/11/09
* 今日は餃子を食べて帰ってきた(19:30 :clock730:)、疲れた。
* macになれてるからかwindowsキーボードに中々慣れない。
* 明日の非常勤の補講資料を作る、偉すぎ。

### 2021/11/08
* 晩ごはんをごちそうになった。美味しかった。
* 宿に帰ってきた(22:30 :clock1030:)が論文書く気力は相当ない。明日以降も疲れてるやろうなあ。
* Netflixを垂れ流しながら寝る。

### 2021/11/07
* 明日からインターンなのでこの一週間は研究をしない
* インターンから宿に帰ったあとに論文を書く気力があったらそのときは頑張る
  * とりあえず論文の今の現状を添付しておく
* 明日以降どこかで今週の研究室セミナーに参加できない旨をメールする
* 水曜日の非常勤の授業は補講の資料を作る必要がある

### 2021/11/04
* この2日研究してなかったので今日は頑張る。
* 論文を書く
* 数値実験もやってまとめる
* ザック・スナイダー版のジャスティス・リーグを見た。4時間の超大作で面白かった。

### 2021/10/25
* データの生成が間違ってることがわかった
* そこの部分を直して実験し直す
  * 時間刻み幅を固定してデータ点数が増えていったときの結合関数推定の精度を比較していく
  * あとは今は$n_{\mathrm{trial}}=1$だが、こっちを増やしていったときの変化も見てみる
* ハイパラが多いので[configparser](https://docs.python.org/ja/3/library/configparser.html)とかを導入してみる
* ニューラルネットワークを使って臨界指数を求めるやつを書いていく → [main.pdf]

### 2021/10/18
* oh-my-poshを導入した。nerd fontを入れる必要がある
* [condaのチートシート](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf)
* van der polの計算の確認を引き続き行う
* データがなんだか変な感じがする
* 結合した2体のvan der pol振動子の結合関数とFourier, GP回帰による推定結果
* 結合した2体のFitzHugh Nagumo振動子の結合関数とFourier, GP回帰による推定結果

### 2021/10/17
* 日曜日なので研究しない
* 映画[『Our friend』](https://our-friend-movie.com/)を見に行った

### 2021/10/16
* 高須さんの計算をした。

### 2021/10/15
* GP回帰を使った結合関数推定の結果をまとめる
  * 昨日の続き
  * 昨日はFitzHugh-Nagumoの結果をまとめたので今日はvan der Pol振動子について
* 時間があればLevy蔵本の結果を思い出しつつ論文の草案みたいなのを作る
* 夜親とご飯食べに行く。

* van der Polの場合をやってみたら太田さんの結果もずれてることが分かった。データの生成がどこかおかしいのか？？要検討。

### 2021/10/14
* 履歴書&研究成果報告書をまとめる
  * $\LaTeX$で履歴書作成: [https://www.tamacom.com/rireki-j.html](https://www.tamacom.com/rireki-j.html)
* GP回帰を使った結合関数推定の結果をまとめる
  * `matplotlib`で$\LaTeX$を使う方法
    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    plt.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath,bm}"
    plt.rcParams["font.family"] = "serif"

    xs = np.arange(0, 1, 0.01)
    ys = xs ** 2

    plt.figure(figsize=[8, 6])
    plt.rcParams["font.size"] = 15
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.plot(xs, ys)
    plt.xlabel("$x$")
    plt.ylabel("$f(x)=x^{2}$")
    plt.savefig("out.png")
    ```

