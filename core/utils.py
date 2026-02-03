import jax
import jax.numpy as jnp


def _spawn_food(key: jax.Array, body: jax.Array, length: jax.Array, grid_num: int) -> tuple[jax.Array, jax.Array]:
    """Sample a random empty cell for food placement using the Gumbel-max trick.

    Args:
        key: PRNG key.
        body: (max_length, 3) int32 body positions.
        length: () int32 current snake length.
        grid_num: size of grid per axis.

    Returns:
        (new_key, food_position) where food_position is (3,) int32.
    """
    total = grid_num ** 3
    key, subkey = jax.random.split(key)

    # Flat indices occupied by the snake
    flat_body = body[:, 0] * grid_num * grid_num + body[:, 1] * grid_num + body[:, 2]

    # Mask: True for occupied cells
    occupied = jnp.zeros(total, dtype=jnp.bool_)
    # Only mark indices up to `length` as occupied
    mask = jnp.arange(body.shape[0]) < length
    occupied = occupied.at[flat_body].set(mask)

    # Gumbel-max trick: logits 0 for empty, -1e9 for occupied → uniform sample over empty
    logits = jnp.where(occupied, -1e9, 0.0)
    gumbel_noise = jax.random.gumbel(subkey, shape=(total,))
    flat_idx = jnp.argmax(logits + gumbel_noise)

    # Convert flat index back to (x, y, z)
    x = flat_idx // (grid_num * grid_num)
    y = (flat_idx % (grid_num * grid_num)) // grid_num
    z = flat_idx % grid_num
    food = jnp.array([x, y, z], dtype=jnp.int32)

    return key, food


def _move_body(body: jax.Array, new_head: jax.Array, length: jax.Array, eating: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Shift body array forward and insert new head at index 0.

    Args:
        body: (max_length, 3) current body.
        new_head: (3,) new head position.
        length: () current length.
        eating: () bool whether snake just ate.

    Returns:
        (new_body, new_length).
    """
    # Roll body forward: each segment takes position of the one before it
    new_body = jnp.roll(body, 1, axis=0)
    new_body = new_body.at[0].set(new_head)

    # New length: increment if was eating
    new_length = length + eating.astype(jnp.int32)

    # If not eating, zero out the tail beyond new_length (clear the vacated slot)
    # The slot at index `new_length` should be zeroed when not growing
    clear_mask = jnp.arange(body.shape[0]) >= new_length
    new_body = jnp.where(clear_mask[:, None], -1, new_body)

    return new_body, new_length


def _check_out_of_bounds(head: jax.Array, grid_num: int) -> jax.Array:
    """Check whether head is outside [0, grid_num).

    Returns:
        () bool, True if out of bounds.
    """
    return jnp.any((head < 0) | (head >= grid_num))


def _check_self_collision(body: jax.Array, length: jax.Array) -> jax.Array:
    """Check whether head (body[0]) collides with any body segment body[1:length].

    Returns:
        () bool, True if self-collision detected.
    """
    head = body[0]
    # Compare head against every other segment
    matches = jnp.all(body[1:] == head, axis=1)
    # Only count segments within current length
    valid = jnp.arange(1, body.shape[0]) < length
    return jnp.any(matches & valid)
