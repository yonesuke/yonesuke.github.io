---
title: "JAXで蔵本モデル"
date: 2022-08-26
slug: jax_kuramoto
draft: true
math: true
authors:
    - yonesuke
categories:
    - Physics
    - Python
---

# JAXで蔵本モデル

<!-- more -->

```python
import jax.numpy as jnp
from jax import random, jit
from jax.lax import fori_loop

class AutonomousDiffEq:
    def __init__(self, n_dim: int):
        self.n_dim = n_dim

    def forward(self, state):
        return NotImplementedError

    def runge_kutta(self, state, dt):
        k1 = self.forward(state)
        k2 = self.forward(state + 0.5 * dt * k1)
        k3 = self.forward(state + 0.5 * dt * k2)
        k4 = self.forward(state + dt * k3)
        slope = (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        return state + dt * slope

    def run(self, init_state: jnp.ndarray, dt: float, t_max: float) -> jnp.ndarray:
        n_loops = int(t_max / dt)
        orbits = jnp.zeros((n_loops, n_dim))
        @jit
        def body_fn(i, val):
            state, orbits = val
            state = self.runge_kutta(state, dt)
            orbits = orbits.at[i].set(state)
            return [state, orbits]
        _, orbits = fori_loop(0, n_loops, body_fn, init_val=[init_state, orbits])
        return orbits
```