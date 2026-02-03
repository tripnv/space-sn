from __future__ import annotations

import heapq
from collections import deque
from itertools import product

import numpy as np

from core.types import GameState, ACTIONS

# Direction tuple → action index lookup
_DIR_TO_ACTION = {tuple(int(x) for x in ACTIONS[i]): i for i in range(6)}


def _build_adjacency(grid_num: int) -> dict[tuple, list[tuple]]:
    directions = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, 1), (0, 0, -1)]
    adj: dict[tuple, list[tuple]] = {}
    for pos in product(range(grid_num), repeat=3):
        adj[pos] = []
        for d in directions:
            adj[pos].append((pos[0] + d[0], pos[1] + d[1], pos[2] + d[2]))
    return adj


class _Node:
    __slots__ = ("position", "parent", "action")

    def __init__(self, position: tuple, parent: _Node | None = None):
        self.position = position
        self.parent = parent
        self.action: int | None = None
        if parent is not None:
            dx = position[0] - parent.position[0]
            dy = position[1] - parent.position[1]
            dz = position[2] - parent.position[2]
            self.action = _DIR_TO_ACTION[(dx, dy, dz)]

    def __eq__(self, other):
        return isinstance(other, _Node) and self.position == other.position

    def __hash__(self):
        return hash(self.position)

    def __lt__(self, other):
        return self.position < other.position

    def __repr__(self):
        return f"Node{self.position}"


def _unwrap_path(node: _Node) -> deque[int]:
    actions: deque[int] = deque()
    while node.parent is not None:
        actions.appendleft(node.action)
        node = node.parent
    return actions


def _valid(pos: tuple, grid_num: int) -> bool:
    return all(0 <= c < grid_num for c in pos)


class _SearchBase:
    """Shared logic for all graph-search agents."""

    def __init__(self, grid_num: int = 10):
        self.grid_num = grid_num
        self._adj = _build_adjacency(grid_num)
        self._action_queue: deque[int] = deque()

    def act(self, state: GameState) -> int:
        if not self._action_queue:
            self._action_queue = self._plan(state)
        if self._action_queue:
            return self._action_queue.popleft()
        # Fallback: pick any non-occupied neighbour, or just go somewhere to die
        return self._fallback(state)

    def _plan(self, state: GameState) -> deque[int]:
        raise NotImplementedError

    def _occupied_set(self, state: GameState) -> set[tuple]:
        body = np.asarray(state.body)
        length = int(state.length)
        return {tuple(int(c) for c in body[i]) for i in range(length)}

    def _fallback(self, state: GameState) -> int:
        head = tuple(int(x) for x in np.asarray(state.body[0]))
        occupied = self._occupied_set(state)
        for neighbour in self._adj[head]:
            if _valid(neighbour, self.grid_num) and neighbour not in occupied:
                dx = neighbour[0] - head[0]
                dy = neighbour[1] - head[1]
                dz = neighbour[2] - head[2]
                return _DIR_TO_ACTION[(dx, dy, dz)]
        return 0  # no safe move — will die


class BFSAgent(_SearchBase):
    agent_type = "BFS"

    def _plan(self, state: GameState) -> deque[int]:
        head = tuple(int(x) for x in np.asarray(state.body[0]))
        goal = tuple(int(x) for x in np.asarray(state.food))
        occupied = self._occupied_set(state)

        start = _Node(head)
        if head == goal:
            return deque()

        frontier: deque[_Node] = deque([start])
        explored: set[_Node] = set()

        while frontier:
            current = frontier.popleft()
            explored.add(current)

            for neighbour_pos in self._adj[current.position]:
                child = _Node(neighbour_pos, parent=current)
                if (
                    _valid(neighbour_pos, self.grid_num)
                    and neighbour_pos not in occupied
                    and child not in explored
                    and child not in frontier
                ):
                    if neighbour_pos == goal:
                        return _unwrap_path(child)
                    frontier.append(child)

        return deque()


class DFSAgent(_SearchBase):
    agent_type = "DFS"

    def _plan(self, state: GameState) -> deque[int]:
        head = tuple(int(x) for x in np.asarray(state.body[0]))
        goal = tuple(int(x) for x in np.asarray(state.food))
        occupied = self._occupied_set(state)

        start = _Node(head)
        if head == goal:
            return deque()

        frontier: list[_Node] = [start]
        explored: set[_Node] = set()

        while frontier:
            current = frontier.pop()
            if current in explored:
                continue
            if current.position == goal:
                return _unwrap_path(current)
            explored.add(current)

            for neighbour_pos in self._adj[current.position]:
                child = _Node(neighbour_pos, parent=current)
                if (
                    _valid(neighbour_pos, self.grid_num)
                    and neighbour_pos not in occupied
                    and child not in explored
                ):
                    frontier.append(child)

        return deque()


class BestFirstAgent(_SearchBase):
    agent_type = "BEST-FIRST"

    def _plan(self, state: GameState) -> deque[int]:
        head = tuple(int(x) for x in np.asarray(state.body[0]))
        goal = tuple(int(x) for x in np.asarray(state.food))
        occupied = self._occupied_set(state)

        start = _Node(head)
        if head == goal:
            return deque()

        counter = 0
        open_set: list[tuple[float, int, _Node]] = [
            (self._heuristic(head, goal), counter, start)
        ]
        closed: set[tuple] = set()

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current.position == goal:
                return _unwrap_path(current)

            if current.position in closed:
                continue
            closed.add(current.position)

            for neighbour_pos in self._adj[current.position]:
                if (
                    not _valid(neighbour_pos, self.grid_num)
                    or neighbour_pos in occupied
                    or neighbour_pos in closed
                ):
                    continue

                child = _Node(neighbour_pos, parent=current)
                counter += 1
                heapq.heappush(
                    open_set, (self._heuristic(neighbour_pos, goal), counter, child)
                )

        return deque()

    @staticmethod
    def _heuristic(pos: tuple, goal: tuple) -> float:
        return sum((a - b) ** 2 for a, b in zip(pos, goal)) ** 0.5


class AStarAgent(_SearchBase):
    agent_type = "ASTAR"

    def _plan(self, state: GameState) -> deque[int]:
        head = tuple(int(x) for x in np.asarray(state.body[0]))
        goal = tuple(int(x) for x in np.asarray(state.food))
        occupied = self._occupied_set(state)

        start = _Node(head)
        if head == goal:
            return deque()

        # g_cost tracks actual path cost (number of steps) for each visited position
        g_cost: dict[tuple, int] = {head: 0}
        # Priority queue entries: (f_cost, tie_breaker, node)
        counter = 0
        open_set: list[tuple[float, int, _Node]] = [
            (self._heuristic(head, goal), counter, start)
        ]
        closed: set[tuple] = set()

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current.position == goal:
                return _unwrap_path(current)

            if current.position in closed:
                continue
            closed.add(current.position)

            current_g = g_cost[current.position]

            for neighbour_pos in self._adj[current.position]:
                if (
                    not _valid(neighbour_pos, self.grid_num)
                    or neighbour_pos in occupied
                    or neighbour_pos in closed
                ):
                    continue

                new_g = current_g + 1
                if new_g < g_cost.get(neighbour_pos, float("inf")):
                    g_cost[neighbour_pos] = new_g
                    f = new_g + self._heuristic(neighbour_pos, goal)
                    child = _Node(neighbour_pos, parent=current)
                    counter += 1
                    heapq.heappush(open_set, (f, counter, child))

        return deque()

    @staticmethod
    def _heuristic(pos: tuple, goal: tuple) -> int:
        """Manhattan distance — admissible and consistent for 6-direction 3D grid."""
        return sum(abs(a - b) for a, b in zip(pos, goal))
