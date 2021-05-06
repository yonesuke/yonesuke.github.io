---
title: "LaTeXの設定"
date: 2021-05-05T18:46:29+09:00
draft: false
---

$\LaTeX$で文章を書くときによく行う設定をまとめておきます。

## Overleaf

- Overleafで日本語で文章を書く場合は次の`latexmkrc`というファイルをおいておきます。
    ```perl
    $latex = 'platex';
    $bibtex = 'pbibtex';
    $dvipdf = 'dvipdfmx %O -o %D %S';
    $makeindex = 'mendex -U %O -o %D %S';
    $pdf_mode = 3; 
    ```
    さらにCompilerは`LaTeX`にしておきましょう。

## Beamer

- LaTeXでスライド生成する際に使われるbeamerです。テーマがたくさんあるので好みのものを選ぶと良いと思います。特に`metropolis`と`focus`というテーマが気に入っています。
    ```tex
    \documentclass[dvipdfmx]{beamer}
    \usepackage{bxdpx-beamer}
    \usepackage{pxjahyper}
    \usepackage{minijs}
    \usetheme[numbering=fraction,block=fill,progressbar=frametitle]{metropolis}
    \usefonttheme{professionalfonts}
    \usepackage{appendixnumberbeamer}
    ```
    Appendixに移行する際の区切りのページには次を書くと便利です。
    ```tex
    % スライド終わり
    \appendix
    \begin{frame}[standout]
        APPENDIX
    \end{frame}
    % アペンディクスはじまり
    % \begin{frame}
    % ...
    ```

## 便利Package

- `hyperref`
    次のオプションとともに呼ぶと便利です。
    ```tex
    \usepackage[colorlinks=true,linkcolor=magenta,citecolor=magenta,breaklinks=true]{hyperref}
    ```
    特に`breaklinks=true`はリンクを改行してくれるので重宝します。リンクの色は好みに合わせて変えてください。

- `biblatex`
    次のオプションとともに呼ぶとPR系の論文のように参考文献を表示してくれます。
    ```tex
    \usepackage[style=phys,articletitle=true,biblabel=brackets,chaptertitle=false,pageranges=false,doi=false]{biblatex}
    % bibファイルの読み込み
    \addbibresource{main.bib}
    % ...
    % 本文
    % ...
    % これで出力
    \printbibliography
    ```

## 便利サイト

- [doi2bib](https://www.doi2bib.org/)
    DOIのリンクを入力するとBibTeXが帰ってきます。各出版社が出力するbibファイルを用いてもよいのですが、出版社によって出力のされ方がまちまちです。このサイトだと一貫した出力のされ方になるので結構気に入っています。

