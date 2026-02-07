from __future__ import annotations

import gymnasium as gym
import numpy as np
import jax
import jax.numpy as jnp
from gymnasium import spaces

from core.engine import reset, step
from core.types import GameState


class SnakeEnv(gym.Env):
    """Gymnasium wrapper around the JAX snake engine.

    Observation: (3, grid_num, grid_num, grid_num) float32
        Channel 0: head (1-hot)
        Channel 1: body including head (1-hot up to length)
        Channel 2: food (1-hot)

    Action: Discrete(6) — see core.types.ACTIONS for mapping.

    Rewards: +1 eat, -1 death, 0 otherwise.
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, grid_num: int = 10, render_mode: str | None = None, seed: int = 0):
        super().__init__()
        self.grid_num = grid_num
        self.max_length = grid_num ** 3
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(3, grid_num, grid_num, grid_num),
            dtype=np.float32,
        )

        self._key = jax.random.PRNGKey(seed)
        self._state: GameState | None = None
        self._renderer = None

        # JIT-compiled functions
        self._reset_fn = jax.jit(reset, static_argnums=(1, 2))
        self._step_fn = jax.jit(step, static_argnums=(2,))

    def _get_obs(self) -> np.ndarray:
        s = self._state
        g = self.grid_num
        obs = np.zeros((3, g, g, g), dtype=np.float32)

        if not bool(s.alive):
            return obs

        body_np = np.asarray(s.body)
        length = int(s.length)

        # Channel 0: head
        h = body_np[0]
        obs[0, h[0], h[1], h[2]] = 1.0

        # Channel 1: body (all segments up to length)
        for i in range(length):
            b = body_np[i]
            obs[1, b[0], b[1], b[2]] = 1.0

        # Channel 2: food
        f = np.asarray(s.food)
        obs[2, f[0], f[1], f[2]] = 1.0

        return obs

    def _get_info(self) -> dict:
        return {
            "score": int(self._state.score),
            "length": int(self._state.length),
            "step_count": int(self._state.step_count),
            "alive": bool(self._state.alive),
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._key = jax.random.PRNGKey(seed)
        self._key, subkey = jax.random.split(self._key)
        self._state = self._reset_fn(subkey, self.grid_num, self.max_length)
        return self._get_obs(), self._get_info()

    def step(self, action: int):
        prev_alive = bool(self._state.alive)
        self._state = self._step_fn(self._state, jnp.int32(action), self.grid_num)
        now_alive = bool(self._state.alive)
        eating = bool(self._state.eating)

        if not prev_alive:
            # Already dead — no-op
            reward = 0.0
            terminated = True
        elif not now_alive:
            reward = -1.0
            terminated = True
        elif eating:
            reward = 1.0
            terminated = False
        else:
            reward = 0.0
            terminated = False

        truncated = False
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        if self.render_mode == "human":
            if self._renderer is None:
                from rendering.renderer import Renderer
                self._renderer = Renderer(self.grid_num)
            self._renderer.update(self._state)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
