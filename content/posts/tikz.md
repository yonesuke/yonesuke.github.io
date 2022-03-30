---
title: "TikZ test"
date: 2022-03-29T18:59:16+09:00
tikz: true
draft: false
---


tikzのtestです。


{{< tikz "A simple cycle" >}}
\begin{tikzpicture}[scale=1.5,transform shape]
  \def \n {5}
  \def \radius {3cm}
  \def \margin {8} % margin in angles, depends on the radius

  \foreach \s in {1,...,\n}
  {
    \node[draw, circle] at ({360/\n * (\s - 1)}:\radius) {$\s$};
    \draw[->, >=latex] ({360/\n * (\s - 1)+\margin}:\radius) 
      arc ({360/\n * (\s - 1)+\margin}:{360/\n * (\s)-\margin}:\radius);
  }
\end{tikzpicture}
{{< /tikz >}}


いい感じ！！

参考にしたサイト: https://blog.xiupos.net/posts/computer/tikz/

基本的にはこのサイトの通りにすればうまく動いた。ただ、tikzを動かすのにwindow.onloadの機能を使っていて、それが他のwindow.onloadと競合すると上書きされてtikzのコンパイルが走らなくなるので注意が必要。
