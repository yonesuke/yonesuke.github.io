---
title: "SHA-256に用いられる定数が微妙に合わない"
date: 2023-06-18
draft: false
math: true
authors:
    - yonesuke
---

ひょんなことからSHA-256を実装しようと思ったのだが、少し謎なところがあったのでまとめる。
何かわかる人がいたらコメントしてほしい。

## SHA-256
SHA-256はハッシュ関数の一つで、任意の文字列を256ビットのハッシュ値を出力する関数である。
ハッシュ関数である以上そのアルゴリズムは複雑で、出力から元の文字列を取り出すことは困難である。
ただ、この関数自体はターミナルですぐに試すことができる。

```bash
echo -n 'hello, world' | sha256sum
> 09ca7e4eaa6e8ae9c7d261167129184883644d07dfba7cbfbc4c8a2e08360d5b  -
echo -n 'hello, world!' | sha256sum
> 68e656b251e67e8358bef8483ab0d51c6619f3e7a1a9f0e75838d41ff368f728  -
```
もしくは、`openssl`を用いてもよい。
```bash
echo -n 'hello, world' | openssl dgst -sha256
> SHA2-256(stdin)= 09ca7e4eaa6e8ae9c7d261167129184883644d07dfba7cbfbc4c8a2e08360d5b
echo -n 'hello, world!' | openssl dgst -sha256
> SHA2-256(stdin)= 68e656b251e67e8358bef8483ab0d51c6619f3e7a1a9f0e75838d41ff368f728
```

(当たり前だが)いずれも出力結果が同じであることが確認できる。また、文字列を少し変えるだけで全く異なった出力を得ることも見て取れる。

今回の記事ではそのアルゴリズム自体は扱わない。参考になる記事として以下を挙げておく。

* https://en.wikipedia.org/wiki/SHA-2
    * Wikipedia
* https://blog.boot.dev/cryptography/how-sha-2-works-step-by-step-sha-256/
    * アルゴリズムの流れが詳しく書いてある
* https://sha256algorithm.com/
    * SHA-256が内部でどのような計算をするのかをアニメーションできれいに表示してくれる。一見の価値あり。こういったウェブサイトを作れるの尊敬する。

## SHA-256のアルゴリズムの中に登場する定数

SHA-256のアルゴリズムの中には定数がいくつか登場する。
それは、ハッシュ値の初期値として設定されるもので、

> first 32 bits of the fractional parts of the square roots of the first 8 primes 2..19

と定義されている。これらを`h0`から`h7`とおくと、

```
h0 := 0x6a09e667
h1 := 0xbb67ae85
h2 := 0x3c6ef372
h3 := 0xa54ff53a
h4 := 0x510e527f
h5 := 0x9b05688c
h6 := 0x1f83d9ab
h7 := 0x5be0cd19
```

となることがwikipediaや多くのウェブサイトで明記されている。
このように結果だけ与えられてしまうと導出をしたくなるのが人間の性である。
というわけで実際に求めてみた。

## 定数の導出とずれ
C++で実装をしてみた。

* double型の変数を64bitの形で表現するには`std::memcpy`を用いた。
このほかにも共有体`union`を用いる方法があるだろう。ほかに適切な変換方法をご存じの方がいればぜひ教えていただきたい。
* double型はbit表現においては、上から符号1bit、指数11bit、仮数52bitで表現される。今必要なのは仮数部の上32bitである。ただ、bitsetによる64bitの表現では仮数部の一番下の桁から順にbitが表現されるので、`get_fraction`関数で配列の要素の取り方は若干変な形になってしまっている。
* 平方根の計算にはニュートン法を用いた。別に二分法でもなんでも良いとは思う。収束条件にはdouble型のbit表現で指数部すべてと仮数部の上から32bit目までがすべて一致していればok、ということにしている。

以下がそのコードである。

```cpp
#include <iostream>
#include <string>
#include <cstring>
#include <vector>
#include <bitset>

// transform double to bitset<64> format
// input: double
// output: std::bitset<64>
std::bitset<64> double2bit(double x)
{
    uint64_t u;
    std::memcpy(&u, &x, sizeof(u));
    std::bitset<64> b(u);
    return b;
}

// get first 32bits of the fractional part of double in bitset<32> format
// input: double
// output: std::bitset<32>
std::bitset<32> get_fraction(double x)
{
    std::bitset<64> b = double2bit(x);
    std::bitset<32> res;
    for (int i = 0; i < 32; i++) res[31 - i] = b[63 - (i + 12)];
    return res;
}

// calculate the square root using Newton method
// input: double
// output: double
// stop condition: first 32bits of fractional part of square root becomes consistent
double sqrt(double x)
{
    double res = x;
    double res_prev;
    while (true)
    {
        res_prev = res;
        res = (res + x / res) / 2;
        std::bitset<64> b1 = double2bit(res_prev);
        std::bitset<64> b2 = double2bit(res);
        bool is_consistent = true;
        for (int i = 0; i < 44; i++)
        {
            if (b1[63 - i] != b2[63 - i])
            {
                is_consistent = false;
                break;
            }
        }
        if (is_consistent) break;
    }
    return res;
}

int main()
{
    // get first 8 prime numbers
    std::vector<int> primes;
    int n = 2;
    while (primes.size() < 8)
    {
        bool is_prime = true;
        for (int i = 2; i < n; i++)
        {
            if (n % i == 0)
            {
                is_prime = false;
                break;
            }
        }
        if (is_prime) primes.push_back(n);
        n++;
    }

    // calculate the square root
    std::vector<double> sqrt_primes;
    for (auto p: primes) sqrt_primes.push_back(sqrt(p));

    // print out square roots in hex forms
    for (int i=0; i<8; i++)
    {
        std::cout << "h" << i << " := " << std::hex << "0x" << get_fraction(sqrt_primes[i]).to_ulong() << std::endl;
    }
}
```

で、この実行結果は以下のようになる。

```
h0 := 0x6a09e667
h1 := 0xbb67ae85
h2 := 0x1e3779b9
h3 := 0x52a7fa9d
h4 := 0xa887293f
h5 := 0xcd82b446
h6 := 0x7e0f66a
h7 := 0x16f83346
```

**あれ、h2以降が違う値をとってる？？？**
ちょっとよくわからんので2進表記で比較を行ってみる。
同じ値の部分を太字で表示してみることにしてみる次のような表になる。

| | 公式値 | 自身の実装値 |
|:---:|:---:|:---:|
| h0 | **01101010000010011110011001100111** | **01101010000010011110011001100111** |
| h1 | **10111011011001111010111010000101** | **10111011011001111010111010000101** |
| h2 | **0011110001101110111100110111001**0 | 0**0011110001101110111100110111001** |
| h3 | **1010010101001111111101010011101**0 | 0**1010010101001111111101010011101** |
| h4 | **0101000100001110010100100111111**1 | 1**0101000100001110010100100111111** |
| h5 | **1001101100000101011010001000110**0 | 1**1001101100000101011010001000110** |
| h6 | **000111111000001111011001101010**11 | 00**000111111000001111011001101010** |
| h7 | **010110111110000011001101000110**01 | 00**010110111110000011001101000110** |

こうしてみると、h0,h1はずれておらず、h2,h3,h4,h5は1bitのずれ、h6,h7は2bitのずれが生じていることが確認できる。この系統的なずれは平方根について、
$$
2^0 < \sqrt{2} < \sqrt{3} < 2^1 < \sqrt{5} < \sqrt{7} < \sqrt{11} < \sqrt{13} < 2^2 < \sqrt{17} < \sqrt{19} < 2^3
$$
が何かしらの影響を与えているのだろうな、という気がしているが実際のところどうなのかはわからない。
もしかすると、double型の変数のbit表現の仕様が時代によって異なっているとかでその結果このような系統的なずれが生まれてしまっているのかもしれない。
（本当はsha256が初めて提案された論文をみるのよいのかもしれない、どれ？？）

## まとめ
というわけで公式に使われている値と自分で実装してみた値に系統的なずれが生じていることがわかった。
このずれの正体が何なのか気になるので知っている方がいればコメントよろしくお願いします。
