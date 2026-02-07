from __future__ import annotations

import jax
import jax.numpy as jnp

from core.engine import reset, step
from core.types import GameState


class VectorizedSnakeEnv:
    """Batched JAX snake environment using vmap.

    All state is kept as JAX arrays with a leading batch dimension.
    No Python loops per step — everything is vectorized and JIT-compiled.
    """

    def __init__(self, num_envs: int, grid_num: int = 10, seed: int = 0):
        self.num_envs = num_envs
        self.grid_num = grid_num
        self.max_length = grid_num ** 3

        self._key = jax.random.PRNGKey(seed)

        self._batch_reset = jax.jit(
            jax.vmap(reset, in_axes=(0, None, None)),
            static_argnums=(1, 2),
        )
        self._batch_step = jax.jit(
            jax.vmap(step, in_axes=(0, 0, None)),
            static_argnums=(2,),
        )
        self._single_reset = jax.jit(reset, static_argnums=(1, 2))

    def reset(self) -> GameState:
        """Reset all environments. Returns batched GameState."""
        self._key, subkey = jax.random.split(self._key)
        keys = jax.random.split(subkey, self.num_envs)
        self._state = self._batch_reset(keys, self.grid_num, self.max_length)
        return self._state

    def step(self, actions: jax.Array) -> tuple[GameState, jax.Array, jax.Array]:
        """Step all environments.

        Args:
            actions: (num_envs,) int32 array of actions.

        Returns:
            (state, rewards, dones) — all batched.
        """
        prev_alive = self._state.alive
        self._state = self._batch_step(self._state, actions, self.grid_num)
        now_alive = self._state.alive
        eating = self._state.eating

        # +1 eat, -1 death, 0 otherwise
        rewards = jnp.where(
            ~prev_alive, 0.0,
            jnp.where(~now_alive, -1.0,
                       jnp.where(eating, 1.0, 0.0))
        )
        dones = ~now_alive

        return self._state, rewards, dones

    def auto_reset(self, actions: jax.Array) -> tuple[GameState, jax.Array, jax.Array]:
        """Step all environments and automatically reset dead ones.

        Args:
            actions: (num_envs,) int32 array of actions.

        Returns:
            (state, rewards, dones) — dones reflect which envs died this step.
        """
        state, rewards, dones = self.step(actions)
        self._state = self._auto_reset_dead(self._state, dones)
        return self._state, rewards, dones

    def _auto_reset_dead(self, state: GameState, dones: jax.Array) -> GameState:
        """Reset envs where dones=True, keep alive ones untouched."""
        self._key, subkey = jax.random.split(self._key)
        keys = jax.random.split(subkey, self.num_envs)
        fresh = self._batch_reset(keys, self.grid_num, self.max_length)

        # Select fresh state for dead envs, keep current for alive
        return jax.tree.map(
            lambda f, s: jnp.where(
                dones.reshape(-1, *([1] * (f.ndim - 1))),
                f, s,
            ),
            fresh, state,
        )
