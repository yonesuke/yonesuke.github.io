---
title: "決定木を1から実装する"
date: 2023-06-25
draft: false
math: true
authors:
  - yonesuke
---

決定木って名前はよく聞くし`scikit-learn`で簡単に使えてしまうけど、中身を詳しく知っているわけではなかったのできちんと実装してみることにする。
from scratchでの実装にはこの記事が非常に参考になった。

* [https://towardsdatascience.com/implementing-a-decision-tree-from-scratch-f5358ff9c4bb](https://towardsdatascience.com/implementing-a-decision-tree-from-scratch-f5358ff9c4bb)
* [https://darden.hatenablog.com/entry/2016/12/15/222447](https://darden.hatenablog.com/entry/2016/12/15/222447)

<!-- more -->

## 決定木

まず決定木の仕組みについて説明する。
とは言っても懇切丁寧に説明している記事はほかにいくらでもあるので、ここではざっくりとした説明に留める。

### アルゴリズム

決定木における分類問題においては分類を行いたい対象に対する特徴量が`n_features`個与えられており、各特徴量は数で表現されるとする。

* 例えば、`sklearn`内に格納されているirisデータセットには4個の特徴量が用意されている。具体的には、sepal length, sepal width, petal length, petal widthであり、いずれも単位はcmの数になっている。

このとき、決定木は与えられた特徴量たちに対して「特徴量Aは$x$より小さいか？」という質問を行う。この質問に対する答えがYesならば、次に「特徴量Bは$y$より小さいか？」という質問を、答えがNoならば、「特徴量Cは$z$より小さいか？」という質問を行う。
こういった質問を次々に行っていき、最後に「前の質問の答えがYesならば分類クラスはPである」といった具合に分類を行う。これが決定木の一つの大きな流れである。

質問によって分岐が起きていくので、最初の質問を根とする**木**とみなすことができる。
この木のもとでは各質問は木の**ノード**と捉えられ、また分岐の果てに分類問題の答えを出力する所は**葉**とみなすことができる。

### 学習

上で説明した決定木アルゴリズムにおいて、各ノードに配置する「特徴量Aは$x$より小さいか？」という質問を作成する必要がある。これを学習データをもとに決定していく**CART**アルゴリズムを紹介する。

学習の流れは各ノードを根から順に決定していく流れとなるので、再帰的な構造を持つ。
学習データが質問を経てあるノードに到着した段階で`n_samples`個になったとしよう。
このとき、**学習データがそれぞれ持つ分類クラスが質問を通して最も別れるような質問が最適な質問である**と考える。明日晴れるか雨が降るかを知りたいときに、質問にYesと答えれば必ず晴れ、Noと答えれば必ず雨、ということが分かれば、これは最適な質問である、ということである。
これを定量的に評価する方法として、**情報利得**を用いることを考える。
`n_samples`個の学習データのそれぞれのクラスが格納された$y\in\mathbb{R}^{n_\mathrm{samples}}$に対して、クラスの揃い具合を表す情報量$f\colon\mathbb{R}^{N}\to\mathbb{R}$があったとしよう。
この指標は値が小さいほど学習データの分類クラスは揃っていて、値が大きいほどに学習データの分類クラスは全くもってばらけている状況を想定する。
ある質問を通して、`n_left`個の学習データがYesと、`n_right`個の学習データがNoと答えたとしよう。また、これに応じて分類クラス$y$は$y_{\mathrm{left}}$と$y_{\mathrm{right}}$に分けられるとする。
学習データが入力された段階では、情報量は$f(y)$である。
また、質問にYesと答えたときの情報量は$f(y_{\mathrm{left}})$、Noと答えたときの情報量は$f(y_{\mathrm{right}})$となる。
確率的には質問に答えた後の情報量は$\frac{n_{\mathrm{left}}}{n_{\mathrm{samples}}}f(y_{\mathrm{left}}) + \frac{n_{\mathrm{right}}}{n_{\mathrm{samples}}}f(y_{\mathrm{right}})$となる。よって、質問に答える前後で情報利得
$$
f(y) - \left(\frac{n_{\mathrm{left}}}{n_{\mathrm{samples}}}f(y_{\mathrm{left}}) + \frac{n_{\mathrm{right}}}{n_{\mathrm{samples}}}f(y_{\mathrm{right}})\right)
$$
が最大となるような質問が最適である、と考える。
具体的な実装においては、各ノードに対して学習データを入力するたびごとに、まず特徴量を一つ選ぶ。学習データにわたってこの特徴量の値を閾値として情報利得を計算する。これをすべての特徴量と閾値に対して行い、最大の情報利得を取る特徴量と閾値でもってそのノードの質問を確定する、
という流れを取る。

分類クラスの揃い具合を表す代表的な指標として、**エントロピー**と**ジニ不純度**が
ある。`sklearn`はデフォルトでジニ不純度を指定するようである。


## 決定木の実装

はじめに、ノードの実装を行う。
各ノードは質問を決定する特徴量(のインデックス)`self.feature`とその閾値`self.threshold`を持つ。
これにより定まる質問に対して答えがYesであれば左側の子ノード`self.left`を、
Noであれば右側の子ノード`selft.right`を参照するようにする。
ただし、葉ノードである場合には、その特徴量(のインデックス)を保管するようにする。

```python
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    
    def is_leaf(self):
        return self.value is not None
```

次に決定木の実装である。

```python
import numpy as np

class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2, criterion='entropy', random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        if criterion == 'entropy':
            self.criterion = self._entropy
        elif criterion == 'gini':
            self.criterion = self._gini
        else:
            raise ValueError(f"Invalid criterion: criterion '{criterion}' not implemented, use 'entropy' or 'gini'")
        if random_state is not None:
            self.random_state = np.random.RandomState(random_state)
        else:
            self.random_state = np.random.RandomState()

    def _is_finished(self, depth, n_class_labels, n_samples):
        return (depth >= self.max_depth or n_class_labels == 1 or n_samples < self.min_samples_split)
    
    def _entropy(self, y):
        proportions = np.bincount(y) / len(y)
        entropy = -np.sum([p * np.log2(p) for p in proportions if p > 0])
        return entropy
    
    def _gini(self, y):
        proportions = np.bincount(y) / len(y)
        gini = 1 - (proportions ** 2).sum()
        return gini

    def _create_split(self, X, thresh):
        idx_left = X <= thresh
        idx_right = np.logical_not(idx_left)
        return idx_left, idx_right

    def _information_gain(self, X, y, thresh):
        idx_left, idx_right = self._create_split(X, thresh)
        n, n_left, n_right = len(y), len(idx_left), len(idx_right)
        if n_left == 0 or n_right == 0:
            return 0
         
        info_parent = self.criterion(y)
        info_child = (n_left / n) * self.criterion(y[idx_left]) + (n_right / n) * self.criterion(y[idx_right])
        return info_parent - info_child

    def _best_split(self, X, y, features_candidate):
        gain_best = -1

        for feat in features_candidate:
            X_feat = X[:, feat]
            thresholds = np.unique(X_feat)
            for thresh in thresholds:
                gain = self._information_gain(X_feat, y, thresh)
                if gain > gain_best:
                    gain_best, feat_best, thresh_best = gain, feat, thresh
        
        return feat_best, thresh_best
    
    def _build_tree(self, X, y, depth=0):
        n_samples = X.shape[0]
        n_class_labels = len(np.unique(y))

        if self._is_finished(depth, n_class_labels, n_samples):
            most_common_labels = np.argmax(np.bincount(y))
            return Node(value=most_common_labels)

        features_candidate = self.random_state.choice(self.n_features, self.n_features, replace=False)
        feat, thresh = self._best_split(X, y, features_candidate)

        idx_left, idx_right = self._create_split(X[:, feat], thresh)
        child_left = self._build_tree(X[idx_left, :], y[idx_left], depth + 1)
        child_right = self._build_tree(X[idx_right, :], y[idx_right], depth + 1)
        return Node(feat, thresh, child_left, child_right)
    
    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value
        node_next = node.left if x[node.feature] <= node.threshold else node.right
        return self._traverse_tree(x, node_next)
    
    def fit(self, X, y):
        self.n_features = X.shape[1]
        self.root = self._build_tree(X, y)

    def predict(self, X):
        predictions = [self._traverse_tree(x, self.root) for x in X]
        return np.array(predictions)
```

!!! note
    `splitter='best'`の場合は各ノードの質問を構成する特徴量を決めるためにすべての特徴量を試す必要がある。そのため、はじめは`random_state`の指定は特に必要ないのではないか？と思っていた。具体的には45行目において`for feat in features_candidate:`を行う代わりに`for i in range(self.n_samples):`を行えば良いのではないか？ということである。
    自分の中での一つの答えは、毎回`for i in range(self.n_samples):`を行ってしまうと、複数の特徴量(と閾値)で同じ最善な情報利得を得た場合に、必ずインデックスが若いほうの特徴量を選んでしまう、という問題が発生してしまう。
    そのため、インデックスを`np.random.choice(self.n_features, self.n_features, replace=False)`で並び変えることによって複数の最適な情報利得を得た場合にもランダム性が作用して性能向上に繋がる可能性が出る。[このIssue](https://github.com/scikit-learn/scikit-learn/issues/2386)が参考になる。


## 具体例

`sklearn`内のデータを利用して分類問題を解こう。`sklearn.tree.DecisionTreeClassifier`との比較も行う。

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

data = datasets.load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

random_state = 20230625

clf_sklearn = DecisionTreeClassifier(max_depth=5, criterion='gini', splitter='best', random_state=random_state)
clf_sklearn.fit(X_train, y_train)
y_pred_sklearn = clf_sklearn.predict(X_test)
acc_sklearn = (y_pred_sklearn == y_test).sum() / len(y_test)

clf_myown = DecisionTree(max_depth=5, criterion='gini', random_state=random_state)
clf_myown.fit(X_train, y_train)
y_pred_myown = clf_myown.predict(X_test)
acc_myown = (y_pred_myown == y_test).sum() / len(y_test)

print(f'Accuracy(sklearn): {acc_sklearn:.5f}')
print(f'Accuracy(my own):  {acc_myown:.5f}')
```

出力は次のようになる。
```console
Accuracy(sklearn): 0.94737
Accuracy(my own):  0.92105
```

それなりに精度は出ているが`sklearn`には負けている。。。

## まとめ

決定木のアルゴリズムを概説し、1から実装を行った。それなりの精度は出ているが`sklearn`には負けている。splitterの作り方が間違っているのかもしれないが、これは`sklearn`の細かい実装を見る必要がある。

この実装をもとにrandom forestや勾配boosting等の実装も行ってみたい。