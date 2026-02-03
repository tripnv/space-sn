"""Pure-JAX search algorithms (BFS, DFS, Best-First, A*).

All plan functions are JIT-compilable and vmap-able. They operate on
raw JAX arrays and return action sequences without side effects.
"""

from __future__ import annotations

from collections import deque
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from core.types import GameState, ACTIONS

# ── Direction mapping ──────────────────────────────────────────────────────────
# ACTIONS[i] gives the (dx, dy, dz) for action i.
# We need the reverse: given a delta, find the action index.
# Build a lookup via dot-product matching against the ACTIONS table.

_DIRS = ACTIONS  # (6, 3) int32


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _build_occupied_grid(body: jnp.ndarray, length: jnp.ndarray, grid_num: int) -> jnp.ndarray:
    """Build a (G,G,G) bool occupancy grid from the snake body."""
    occupied = jnp.zeros((grid_num, grid_num, grid_num), dtype=jnp.bool_)

    def _mark(i, occ):
        pos = body[i]
        safe = jnp.clip(pos, 0, grid_num - 1)
        valid = jnp.all((pos >= 0) & (pos < grid_num))
        return occ.at[safe[0], safe[1], safe[2]].set(
            occ[safe[0], safe[1], safe[2]] | valid
        )

    return jax.lax.fori_loop(0, length, _mark, occupied)


def _delta_to_action(delta: jnp.ndarray) -> jnp.ndarray:
    """Convert a (3,) direction delta to an action index (scalar int32)."""
    # Match against ACTIONS table
    match = jnp.all(_DIRS == delta, axis=1)  # (6,) bool
    return jnp.argmax(match).astype(jnp.int32)


def _reconstruct_path(
    goal: jnp.ndarray,
    head: jnp.ndarray,
    parent: jnp.ndarray,
    grid_num: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Trace parent pointers goal→head and return (actions, num_actions).

    actions: (max_path,) int32 — action indices, valid up to num_actions
    num_actions: () int32
    """
    max_path = grid_num ** 3

    # First, collect the path in reverse (goal → head)
    rev_actions = jnp.zeros(max_path, dtype=jnp.int32)

    def _trace(carry):
        pos, rev_acts, idx = carry
        p = parent[pos[0], pos[1], pos[2]]
        delta = pos - p
        act = _delta_to_action(delta)
        rev_acts = rev_acts.at[idx].set(act)
        return (p, rev_acts, idx + 1)

    def _cond(carry):
        pos, _, _ = carry
        return ~jnp.all(pos == head)

    init = (goal, rev_actions, jnp.int32(0))
    final_pos, rev_acts, count = jax.lax.while_loop(_cond, _trace, init)

    # Reverse the path
    actions = jnp.zeros(max_path, dtype=jnp.int32)
    def _reverse(i, acts):
        acts = acts.at[i].set(rev_acts[count - 1 - i])
        return acts
    actions = jax.lax.fori_loop(0, count, _reverse, actions)

    return actions, count


def _jax_fallback(head: jnp.ndarray, occupied: jnp.ndarray, grid_num: int) -> jnp.ndarray:
    """Return the first safe action, or 0 if none."""
    def _check(i, best):
        found, act = best
        neighbor = head + _DIRS[i]
        safe = jnp.clip(neighbor, 0, grid_num - 1)
        in_bounds = jnp.all((neighbor >= 0) & (neighbor < grid_num))
        free = ~occupied[safe[0], safe[1], safe[2]]
        is_valid = in_bounds & free & ~found
        act = jnp.where(is_valid, i, act)
        found = found | (in_bounds & free)
        return (found, act)

    _, action = jax.lax.fori_loop(0, 6, _check, (jnp.bool_(False), jnp.int32(0)))
    return action


def _euclidean_heuristic(pos: jnp.ndarray, goal: jnp.ndarray) -> jnp.ndarray:
    diff = (pos - goal).astype(jnp.float32)
    return jnp.sqrt(jnp.sum(diff ** 2))


def _manhattan_heuristic(pos: jnp.ndarray, goal: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(jnp.abs(pos - goal)).astype(jnp.float32)


# ── BFS ────────────────────────────────────────────────────────────────────────

@partial(jax.jit, static_argnums=(3,))
def bfs_plan(
    head: jnp.ndarray,
    goal: jnp.ndarray,
    occupied: jnp.ndarray,
    grid_num: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """BFS search returning (actions, num_actions, found)."""
    cap = grid_num ** 3

    # Queue: FIFO via flat array
    queue = jnp.zeros((cap, 3), dtype=jnp.int32)
    queue = queue.at[0].set(head)
    front = jnp.int32(0)
    back = jnp.int32(1)

    # Visited: treat occupied cells as pre-visited
    visited = occupied.copy()
    visited = visited.at[head[0], head[1], head[2]].set(True)

    # Parent pointers (init to own position = "no parent")
    parent = jnp.zeros((grid_num, grid_num, grid_num, 3), dtype=jnp.int32)
    # Initialize each cell's parent to itself (sentinel)
    ix = jnp.arange(grid_num)
    gx, gy, gz = jnp.meshgrid(ix, ix, ix, indexing='ij')
    parent = parent.at[:, :, :, 0].set(gx)
    parent = parent.at[:, :, :, 1].set(gy)
    parent = parent.at[:, :, :, 2].set(gz)

    found = jnp.bool_(False)
    found_goal = goal.copy()

    def _body(carry):
        queue, front, back, visited, parent, found, found_goal = carry
        current = queue[front]
        front = front + 1

        def _expand(j, inner):
            q, bk, vis, par, fnd, fg = inner
            neighbor = current + _DIRS[j]
            safe = jnp.clip(neighbor, 0, grid_num - 1)
            in_bounds = jnp.all((neighbor >= 0) & (neighbor < grid_num))
            not_visited = ~vis[safe[0], safe[1], safe[2]]
            valid = in_bounds & not_visited & ~fnd

            # Update visited
            vis = vis.at[safe[0], safe[1], safe[2]].set(
                vis[safe[0], safe[1], safe[2]] | valid
            )
            # Update parent
            par = par.at[safe[0], safe[1], safe[2]].set(
                jnp.where(valid, current, par[safe[0], safe[1], safe[2]])
            )
            # Add to queue
            q = q.at[bk].set(jnp.where(valid, neighbor, q[bk]))
            bk = jnp.where(valid, bk + 1, bk)
            # Goal check
            is_goal = jnp.all(neighbor == goal) & in_bounds
            fnd = fnd | (is_goal & valid)
            fg = jnp.where(is_goal & valid, neighbor, fg)
            return (q, bk, vis, par, fnd, fg)

        queue, back, visited, parent, found, found_goal = jax.lax.fori_loop(
            0, 6, _expand, (queue, back, visited, parent, found, found_goal)
        )
        return (queue, front, back, visited, parent, found, found_goal)

    def _cond(carry):
        _, front, back, _, _, found, _ = carry
        return (front < back) & ~found

    init = (queue, front, back, visited, parent, found, found_goal)
    queue, front, back, visited, parent, found, found_goal = jax.lax.while_loop(
        _cond, _body, init
    )

    actions, num_actions = _reconstruct_path(found_goal, head, parent, grid_num)
    # If not found, num_actions = 0
    num_actions = jnp.where(found, num_actions, jnp.int32(0))
    return actions, num_actions, found


# ── DFS ────────────────────────────────────────────────────────────────────────

@partial(jax.jit, static_argnums=(3,))
def dfs_plan(
    head: jnp.ndarray,
    goal: jnp.ndarray,
    occupied: jnp.ndarray,
    grid_num: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """DFS search returning (actions, num_actions, found)."""
    cap = grid_num ** 3

    # Stack: LIFO via flat array
    stack = jnp.zeros((cap, 3), dtype=jnp.int32)
    stack = stack.at[0].set(head)
    top = jnp.int32(1)

    # Visited: treat occupied cells as pre-visited
    visited = occupied.copy()
    visited = visited.at[head[0], head[1], head[2]].set(True)

    parent = jnp.zeros((grid_num, grid_num, grid_num, 3), dtype=jnp.int32)
    ix = jnp.arange(grid_num)
    gx, gy, gz = jnp.meshgrid(ix, ix, ix, indexing='ij')
    parent = parent.at[:, :, :, 0].set(gx)
    parent = parent.at[:, :, :, 1].set(gy)
    parent = parent.at[:, :, :, 2].set(gz)

    found = jnp.bool_(False)
    found_goal = goal.copy()

    def _body(carry):
        stack, top, visited, parent, found, found_goal = carry
        top = top - 1
        current = stack[top]

        def _expand(j, inner):
            stk, tp, vis, par, fnd, fg = inner
            neighbor = current + _DIRS[j]
            safe = jnp.clip(neighbor, 0, grid_num - 1)
            in_bounds = jnp.all((neighbor >= 0) & (neighbor < grid_num))
            not_visited = ~vis[safe[0], safe[1], safe[2]]
            valid = in_bounds & not_visited & ~fnd

            vis = vis.at[safe[0], safe[1], safe[2]].set(
                vis[safe[0], safe[1], safe[2]] | valid
            )
            par = par.at[safe[0], safe[1], safe[2]].set(
                jnp.where(valid, current, par[safe[0], safe[1], safe[2]])
            )
            stk = stk.at[tp].set(jnp.where(valid, neighbor, stk[tp]))
            tp = jnp.where(valid, tp + 1, tp)
            is_goal = jnp.all(neighbor == goal) & in_bounds
            fnd = fnd | (is_goal & valid)
            fg = jnp.where(is_goal & valid, neighbor, fg)
            return (stk, tp, vis, par, fnd, fg)

        stack, top, visited, parent, found, found_goal = jax.lax.fori_loop(
            0, 6, _expand, (stack, top, visited, parent, found, found_goal)
        )
        return (stack, top, visited, parent, found, found_goal)

    def _cond(carry):
        _, top, _, _, found, _ = carry
        return (top > 0) & ~found

    init = (stack, top, visited, parent, found, found_goal)
    stack, top, visited, parent, found, found_goal = jax.lax.while_loop(
        _cond, _body, init
    )

    actions, num_actions = _reconstruct_path(found_goal, head, parent, grid_num)
    num_actions = jnp.where(found, num_actions, jnp.int32(0))
    return actions, num_actions, found


# ── Best-First ─────────────────────────────────────────────────────────────────

@partial(jax.jit, static_argnums=(3,))
def best_first_plan(
    head: jnp.ndarray,
    goal: jnp.ndarray,
    occupied: jnp.ndarray,
    grid_num: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Best-First (greedy) search returning (actions, num_actions, found)."""
    cap = grid_num ** 3

    # Flat-array priority queue: costs + positions
    costs = jnp.full(cap, jnp.inf, dtype=jnp.float32)
    positions = jnp.zeros((cap, 3), dtype=jnp.int32)
    costs = costs.at[0].set(_euclidean_heuristic(head, goal))
    positions = positions.at[0].set(head)
    pq_size = jnp.int32(1)

    # Closed set
    closed = occupied.copy()
    closed = closed.at[head[0], head[1], head[2]].set(False)  # head must be expandable

    parent = jnp.zeros((grid_num, grid_num, grid_num, 3), dtype=jnp.int32)
    ix = jnp.arange(grid_num)
    gx, gy, gz = jnp.meshgrid(ix, ix, ix, indexing='ij')
    parent = parent.at[:, :, :, 0].set(gx)
    parent = parent.at[:, :, :, 1].set(gy)
    parent = parent.at[:, :, :, 2].set(gz)

    found = jnp.bool_(False)
    found_goal = goal.copy()

    def _body(carry):
        costs, positions, pq_size, closed, parent, found, found_goal = carry

        # Pop min
        idx = jnp.argmin(costs)
        current = positions[idx]
        costs = costs.at[idx].set(jnp.inf)

        # Check if already closed (lazy dedup)
        safe_cur = jnp.clip(current, 0, grid_num - 1)
        already_closed = closed[safe_cur[0], safe_cur[1], safe_cur[2]]

        # Goal check on pop
        is_goal = jnp.all(current == goal)
        found = found | (is_goal & ~already_closed)
        found_goal = jnp.where(is_goal & ~already_closed, current, found_goal)

        # Mark closed
        closed = closed.at[safe_cur[0], safe_cur[1], safe_cur[2]].set(True)

        def _expand(j, inner):
            c, pos, sz, cl, par, fnd, fg = inner
            neighbor = current + _DIRS[j]
            safe = jnp.clip(neighbor, 0, grid_num - 1)
            in_bounds = jnp.all((neighbor >= 0) & (neighbor < grid_num))
            not_closed = ~cl[safe[0], safe[1], safe[2]]
            valid = in_bounds & not_closed & ~already_closed & ~fnd

            h = _euclidean_heuristic(neighbor, goal)
            c = c.at[sz].set(jnp.where(valid, h, c[sz]))
            pos = pos.at[sz].set(jnp.where(valid, neighbor, pos[sz]))
            par = par.at[safe[0], safe[1], safe[2]].set(
                jnp.where(
                    valid & ~cl[safe[0], safe[1], safe[2]],
                    current,
                    par[safe[0], safe[1], safe[2]],
                )
            )
            sz = jnp.where(valid, sz + 1, sz)
            return (c, pos, sz, cl, par, fnd, fg)

        costs, positions, pq_size, closed, parent, found, found_goal = jax.lax.fori_loop(
            0, 6, _expand, (costs, positions, pq_size, closed, parent, found, found_goal)
        )
        return (costs, positions, pq_size, closed, parent, found, found_goal)

    def _cond(carry):
        costs, _, _, _, _, found, _ = carry
        has_elements = jnp.any(costs < jnp.inf)
        return has_elements & ~found

    init = (costs, positions, pq_size, closed, parent, found, found_goal)
    costs, positions, pq_size, closed, parent, found, found_goal = jax.lax.while_loop(
        _cond, _body, init
    )

    actions, num_actions = _reconstruct_path(found_goal, head, parent, grid_num)
    num_actions = jnp.where(found, num_actions, jnp.int32(0))
    return actions, num_actions, found


# ── A* ─────────────────────────────────────────────────────────────────────────

@partial(jax.jit, static_argnums=(3,))
def astar_plan(
    head: jnp.ndarray,
    goal: jnp.ndarray,
    occupied: jnp.ndarray,
    grid_num: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """A* search returning (actions, num_actions, found)."""
    cap = grid_num ** 3

    # Flat-array PQ
    costs = jnp.full(cap, jnp.inf, dtype=jnp.float32)  # f-values
    positions = jnp.zeros((cap, 3), dtype=jnp.int32)
    costs = costs.at[0].set(_manhattan_heuristic(head, goal))
    positions = positions.at[0].set(head)
    pq_size = jnp.int32(1)

    # g-cost grid
    g_cost = jnp.full((grid_num, grid_num, grid_num), jnp.inf, dtype=jnp.float32)
    g_cost = g_cost.at[head[0], head[1], head[2]].set(0.0)

    closed = occupied.copy()
    closed = closed.at[head[0], head[1], head[2]].set(False)  # head must be expandable

    parent = jnp.zeros((grid_num, grid_num, grid_num, 3), dtype=jnp.int32)
    ix = jnp.arange(grid_num)
    gx, gy, gz = jnp.meshgrid(ix, ix, ix, indexing='ij')
    parent = parent.at[:, :, :, 0].set(gx)
    parent = parent.at[:, :, :, 1].set(gy)
    parent = parent.at[:, :, :, 2].set(gz)

    found = jnp.bool_(False)
    found_goal = goal.copy()

    def _body(carry):
        costs, positions, pq_size, g_cost, closed, parent, found, found_goal = carry

        # Pop min f-value
        idx = jnp.argmin(costs)
        current = positions[idx]
        costs = costs.at[idx].set(jnp.inf)

        safe_cur = jnp.clip(current, 0, grid_num - 1)
        already_closed = closed[safe_cur[0], safe_cur[1], safe_cur[2]]

        # Goal check on pop
        is_goal = jnp.all(current == goal)
        found = found | (is_goal & ~already_closed)
        found_goal = jnp.where(is_goal & ~already_closed, current, found_goal)

        closed = closed.at[safe_cur[0], safe_cur[1], safe_cur[2]].set(True)
        current_g = g_cost[safe_cur[0], safe_cur[1], safe_cur[2]]

        def _expand(j, inner):
            c, pos, sz, gc, cl, par, fnd, fg = inner
            neighbor = current + _DIRS[j]
            safe = jnp.clip(neighbor, 0, grid_num - 1)
            in_bounds = jnp.all((neighbor >= 0) & (neighbor < grid_num))
            not_closed = ~cl[safe[0], safe[1], safe[2]]

            new_g = current_g + 1.0
            old_g = gc[safe[0], safe[1], safe[2]]
            improves = new_g < old_g
            valid = in_bounds & not_closed & ~already_closed & ~fnd & improves

            gc = gc.at[safe[0], safe[1], safe[2]].set(
                jnp.where(valid, new_g, gc[safe[0], safe[1], safe[2]])
            )
            f = new_g + _manhattan_heuristic(neighbor, goal)
            c = c.at[sz].set(jnp.where(valid, f, c[sz]))
            pos = pos.at[sz].set(jnp.where(valid, neighbor, pos[sz]))
            par = par.at[safe[0], safe[1], safe[2]].set(
                jnp.where(valid, current, par[safe[0], safe[1], safe[2]])
            )
            sz = jnp.where(valid, sz + 1, sz)
            return (c, pos, sz, gc, cl, par, fnd, fg)

        costs, positions, pq_size, g_cost, closed, parent, found, found_goal = jax.lax.fori_loop(
            0, 6, _expand,
            (costs, positions, pq_size, g_cost, closed, parent, found, found_goal),
        )
        return (costs, positions, pq_size, g_cost, closed, parent, found, found_goal)

    def _cond(carry):
        costs, _, _, _, _, _, found, _ = carry
        has_elements = jnp.any(costs < jnp.inf)
        return has_elements & ~found

    init = (costs, positions, pq_size, g_cost, closed, parent, found, found_goal)
    result = jax.lax.while_loop(_cond, _body, init)
    costs, positions, pq_size, g_cost, closed, parent, found, found_goal = result

    actions, num_actions = _reconstruct_path(found_goal, head, parent, grid_num)
    num_actions = jnp.where(found, num_actions, jnp.int32(0))
    return actions, num_actions, found


# ── Wrapper classes ────────────────────────────────────────────────────────────

class _JAXSearchBase:
    """Shared wrapper logic: builds occupied grid, manages action queue."""

    _plan_fn = None  # overridden by subclass

    def __init__(self, grid_num: int = 10):
        self.grid_num = grid_num
        self._action_queue: deque[int] = deque()
        # Pre-JIT the plan function (first call triggers compilation)
        self._plan_jit = jax.jit(self._plan_fn, static_argnums=(3,))

    def act(self, state: GameState) -> int:
        if not self._action_queue:
            self._replan(state)
        if self._action_queue:
            return self._action_queue.popleft()
        # Fallback
        head = jnp.asarray(state.body[0], dtype=jnp.int32)
        occupied = _build_occupied_grid(state.body, state.length, self.grid_num)
        return int(_jax_fallback(head, occupied, self.grid_num))

    def _replan(self, state: GameState):
        head = jnp.asarray(state.body[0], dtype=jnp.int32)
        goal = jnp.asarray(state.food, dtype=jnp.int32)
        occupied = _build_occupied_grid(state.body, state.length, self.grid_num)

        actions, num_actions, found = self._plan_jit(head, goal, occupied, self.grid_num)

        if bool(found):
            n = int(num_actions)
            act_np = np.asarray(actions[:n])
            self._action_queue = deque(int(a) for a in act_np)


class JAXBFSAgent(_JAXSearchBase):
    agent_type = "JAX-BFS"
    _plan_fn = staticmethod(bfs_plan)


class JAXDFSAgent(_JAXSearchBase):
    agent_type = "JAX-DFS"
    _plan_fn = staticmethod(dfs_plan)


class JAXBestFirstAgent(_JAXSearchBase):
    agent_type = "JAX-BEST-FIRST"
    _plan_fn = staticmethod(best_first_plan)


class JAXAStarAgent(_JAXSearchBase):
    agent_type = "JAX-ASTAR"
    _plan_fn = staticmethod(astar_plan)
