import jax
import jax.numpy as jnp

from core.types import GameState, ACTIONS
from core.utils import _spawn_food, _move_body, _check_out_of_bounds, _check_self_collision


def reset(key: jax.Array, grid_num: int, max_length: int) -> GameState:
    """Initialize a fresh game state with random head position, direction, and food."""
    key, k1, k2, k3 = jax.random.split(key, 4)

    # Random head position in [0, grid_num)^3
    head = jax.random.randint(k1, shape=(3,), minval=0, maxval=grid_num)

    # Random direction (one of 6)
    dir_idx = jax.random.randint(k2, shape=(), minval=0, maxval=6)
    direction = ACTIONS[dir_idx]

    # Body array: head at index 0, rest padded with -1
    body = jnp.full((max_length, 3), -1, dtype=jnp.int32)
    body = body.at[0].set(head)

    length = jnp.int32(1)

    # Spawn food away from head
    key, food = _spawn_food(k3, body, length, grid_num)

    return GameState(
        body=body,
        length=length,
        direction=direction,
        food=food,
        alive=jnp.bool_(True),
        eating=jnp.bool_(False),
        step_count=jnp.int32(0),
        score=jnp.int32(0),
        key=key,
    )


def step(state: GameState, action: jax.Array, grid_num: int) -> GameState:
    """Advance the game by one step given an action (int in [0, 5]).

    Handles direction assignment (blocking 180-degree reversal), movement,
    collision detection, food eating, and respawning.
    """
    # Map action int to direction vector
    new_dir = ACTIONS[action]

    # Block 180-degree reversal when length > 1:
    # if new_dir + current_dir == 0 on all axes, keep current direction
    is_reversal = jnp.all(new_dir + state.direction == 0)
    is_long = state.length > 1
    direction = jnp.where(is_reversal & is_long, state.direction, new_dir)

    # Compute new head
    new_head = state.body[0] + direction

    # Move body
    new_body, new_length = _move_body(state.body, new_head, state.length, state.eating)

    # Check collisions
    oob = _check_out_of_bounds(new_head, grid_num)
    self_coll = _check_self_collision(new_body, new_length)
    alive = state.alive & ~oob & ~self_coll

    # Check food collision
    eating = jnp.all(new_head == state.food) & alive
    score = state.score + eating.astype(jnp.int32)

    # Spawn new food if eating (otherwise keep current food)
    key, new_food = _spawn_food(state.key, new_body, new_length, grid_num)
    food = jnp.where(eating, new_food, state.food)
    # Only consume the key if we actually spawned
    key = jnp.where(eating, key, state.key)

    return GameState(
        body=new_body,
        length=new_length,
        direction=direction,
        food=food,
        alive=alive,
        eating=eating,
        step_count=state.step_count + 1,
        score=score,
        key=key,
    )
