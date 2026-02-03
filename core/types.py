from typing import NamedTuple

import jax.numpy as jnp

# 6 discrete actions: -x, +x, -y, +y, +z, -z
ACTIONS = jnp.array([
    [-1, 0, 0],
    [1, 0, 0],
    [0, -1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [0, 0, -1],
], dtype=jnp.int32)

ACTIONS_NAMES = ["x_left", "x_right", "y_up", "y_down", "z_forward", "z_backward"]


class GameState(NamedTuple):
    body: jnp.ndarray       # (max_length, 3) int32 — index 0 = head, padded with -1 beyond length
    length: jnp.ndarray     # () int32
    direction: jnp.ndarray  # (3,) int32
    food: jnp.ndarray       # (3,) int32
    alive: jnp.ndarray      # () bool
    eating: jnp.ndarray     # () bool
    step_count: jnp.ndarray # () int32
    score: jnp.ndarray      # () int32
    key: jnp.ndarray        # (2,) uint32  — JAX PRNG key
