---
title: "ハイパラが増えたらConfigをPydanticで管理し、argparseでsweepまで済ませる"
date: 2026-08-25
slug: pydantic_argparse_cli_sweep
draft: false
math: true
authors:
    - yonesuke
categories:
    - Python
    - Machine Learning
---

機械学習の実験コードを書いていると、ハイパーパラメータがどんどん増えていきます。
そして実験を回し始めると、今度はパラメータを少しずつ変えてsweepしたいという欲求も出てきます。

ここでは、"PydanticでConfigを定義して、argparse経由でCLIから渡せるようにする" という簡単なコードを紹介します。
これにより、パラメータをgridでsweepすることができ、実験の管理が楽になります。

<!-- more -->

## Configをクラスにまとめる

まずはConfigをPydanticの`BaseModel`で定義します。
`dataclass`でもよいのですが、Pydanticにするとバリデーション（`gt`、`le`、`ge`など）が使えて、早い段階でConfgのvalidationを効かせられるので便利です。

```python
from enum import Enum
from pathlib import Path
from pydantic import BaseModel, Field

class OptimizerType(str, Enum):
    adam = "adam"
    adamw = "adamw"
    sgd = "sgd"

class Config(BaseModel):
    lr: float       = Field(default=3e-4, gt=0.0, le=1.0, description="Learning rate")
    epochs: int     = Field(default=10, ge=1, description="Number of training epochs")
    batch_size: int = Field(default=64, ge=1, description="Mini-batch size")
    seed: int       = Field(default=42, description="Random seed for reproducibility")
    hidden_dim: int = Field(default=128, ge=1, description="Hidden layer dimension")
    num_layers: int = Field(default=3, ge=1, le=100, description="Number of layers")
    dropout: float  = Field(default=0.1, ge=0.0, le=1.0, description="Dropout rate")
    optimizer: OptimizerType = Field(default=OptimizerType.adamw, description="Optimizer type")
    weight_decay: float = Field(default=1e-2, ge=0.0, description="Weight decay coefficient")
    dataset: str    = Field(default="mnist", description="Dataset name")
    output_dir: Path = Field(default=Path("outputs"), description="Output directory path")
    experiment_name: str = Field(default="default", description="Experiment name for logging")
    use_amp: bool   = Field(default=True, description="Enable automatic mixed precision")
```

普段の実験ではこの`Config`をそのまま生成して`train(cfg)`に渡します。コードがスッキリし、新しいパラを追加するのもフィールドを一行足すだけです。

## sweepしたくなったら

実験を本格的に回し始めると、「`dropout`を0.1/0.2/0.3で比べたい」とか「`lr`が`1e-3`と`1e-4`のどちらが効くか」といった**パラメータスイープ（parameter sweep）**をしたくなります。

直感的には「指定しなかったパラにはdefault値を使い、指定したパラには複数の候補値の全部の組み合わせを回す」というのが欲しい形でしょう。単一値ならsweepなし、複数値ならgrid sweepという動作にします。

## Pydantic → argparseでブリッジする

Configのフィールドをそのまま`argparse`の引数に移し替え、`nargs="+"`で「複数値を受け取れる」にします。

- 単一値だけ渡す → スイープなしの1回実行
- 複数値でスペース区切りに渡す → 各パラの候補の直積で複数回いっぺんに実行

それを`itertools.product`で展開して、`list[Config]`を作ります。

```python
import itertools
import argparse
from typing import Any
from pydantic import BaseModel
from setuptools._distutils.util import strtobool

def _build_parser(model_cls: type[BaseModel]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ML Training Config (pass multiple values per arg to sweep)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    for name, info in model_cls.model_fields.items():
        flag = f"--{name.replace('_', '-')}"
        ann = info.annotation
        default = info.default
        desc = info.description or name
        default_str = default.value if isinstance(default, Enum) else repr(default)
        help_text = f"{desc} (default: {default_str})"

        kwargs: dict[str, Any] = dict(nargs="+", default=None, help=help_text)
        if isinstance(ann, type) and issubclass(ann, Enum):
            kwargs["type"] = ann
            kwargs["choices"] = [e.value for e in ann]
        elif ann is bool:
            kwargs["type"] = strtobool
        else:
            kwargs["type"] = ann
        parser.add_argument(flag, **kwargs)

    return parser


def parse_configs(model_cls: type[BaseModel] = Config) -> list[Config]:
    parser = _build_parser(model_cls)
    args = parser.parse_args()

    grid: dict[str, list[Any]] = {}
    for name, info in model_cls.model_fields.items():
        raw: list | None = getattr(args, name, None)
        grid[name] = raw if raw is not None else [info.default]

    keys = list(grid.keys())
    return [model_cls(**dict(zip(keys, combo)))
            for combo in itertools.product(*(grid[k] for k in keys))]
```

メインではこう使います。

```python
import itertools

def train(cfg: Config) -> float:
    return cfg.lr * cfg.epochs  # placeholder

if __name__ == "__main__":
    configs = parse_configs()
    print(f"Sweep: {len(configs)} run(s)\n")

    for i, cfg in enumerate(configs, 1):
        loss = train(cfg)
        print(f"[{i}/{len(configs)}] lr={cfg.lr} dropout={cfg.dropout} "
              f"optimizer={cfg.optimizer.value} -> loss={loss:.6f}")
```

## 使い方

```console
# 単一値：スイープなし
uv run main.py --lr 0.001 --epochs 10

# 複数値を与えるとgridでスイープ
uv run main.py --lr 1e-3 1e-4 --dropout 0.1 0.2 0.3

# ヘルプ表示
uv run main.py --help
```

この例では`lr`が2候補、`dropout`が3候補なので直積 `2×3=6` 通りのConfigが自動生成されます。設定の何を変えたのか、どの組み合わせを回したのかがCLIで一目瞭然になります。

## まとめ

この記事ではPydanticでConfigを定義し、argparseでCLIからパラメータをgrid sweepできるようにする方法を紹介しました。
これにより、ハイパーパラメータの管理が容易になり、実験の効率化が図れます。
また、このようなコマンドライン引数を準備する副次的な効果として、
AIエージェントが実験を自動で回す際にも便利なインターフェースになりうると考えています。

それでは。
