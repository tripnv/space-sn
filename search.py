from itertools import product
from typing import Any, List
from collections import deque

GRID_NUM = 10


class Node:
    def __init__(self, position: tuple, parent=None, action=None, path_cost=0) -> None:
        self.position = position
        self.parent = parent
        self.action = action
        self.depth = path_cost
        if self.parent:
            self.depth += 1

    def __repr__(self) -> str:
        return f"Node{self.position}"

    def __eq__(self, __value) -> bool:
        """Check whether the position of the Node equals that of another node"""
        return isinstance(__value, Node) and self.position == __value.position

    def expand(self):
        """List all nodes one step away given the position of the node"""
        return [Node(neighbour) for neighbour in ADJACENCY_DICT[self.position]]

    def __hash__(self) -> int:
        return hash(self.position)

    def add_action(self) -> None:
        """Assign action that needs to be taken to get from parent position to child position."""
        child_x, child_y, child_z = self.position
        parent_x, parent_y, parent_z = self.parent.position

        self.action = (child_x - parent_x, child_y - parent_y, child_z - parent_z)


class Agent:
    def __init__(self, agent_type: str) -> None:
        self.agent_type = agent_type.upper()

    def position_occupied(self, node):
        for block in self.occupied_positions:
            if block.__eq__(node):
                return True
        return False

    def invalid_position(self, node):
        x, y, z = node.position
        if x < 0 or x > GRID_NUM - 1:
            return True
        if y < 0 or y > GRID_NUM - 1:
            return True
        if z < 0 or z > GRID_NUM - 1:
            return True
        return False

    def generate_path(
        self,
        start_position: tuple,
        end_position: tuple,
        occupied_positions: List[tuple],
    ):
        """Generate the sequence of actions that connect the start position to the end position such that the occupied squares are avoided."""
        self.occupied_positions = [Node(block) for block in occupied_positions]

        # Sceanrio 1
        if self.agent_type == "BFS":
            search_final_node = self.bfs(start_position, end_position)

        elif self.agent_type == "DFS":
            search_final_node = self.dfs(start_position, end_position)
            if search_final_node == None:
                search_final_node = self.dfs(
                    start_position, self.occupied_positions[-1]
                )
        elif self.agent_type == "A*":
            raise NotImplementedError

        else:
            raise NotImplementedError

        if search_final_node == None:
            search_final_node = self.select_available_position(start_position)

        return self.unwrap_path(search_final_node)

    def select_available_position(self, start_position):
        """Return the first available child node; if no such child has been found select the first adjacent position"""
        node = Node(start_position)
        for child_position in ADJACENCY_DICT[node.position]:
            child_node = Node(child_position, parent=node)
            if not self.position_occupied(child_node):
                child_node.add_action()
                return child_node

        # Case for no available positions; basically, killing the snake
        child_node = Node(ADJACENCY_DICT[node.position][0], parent=node)
        child_node.add_action()
        return child_node

    def bfs(self, start_position, end_position):
        """Breadth first search"""
        node = Node(start_position)
        goal = Node(end_position)

        # Test goal state
        if goal.__eq__(node):
            return node

        frontier = deque([node])
        explored = set()
        while frontier:
            node = frontier.popleft()
            explored.add(node)

            # Nodes that are reachable in one step
            for child_position in ADJACENCY_DICT[node.position]:
                child_node = Node(child_position, parent=node)
                if (
                    not self.position_occupied(child_node)
                    and not self.invalid_position(child_node)
                    and child_node not in explored
                    and child_node not in frontier
                ):
                    child_node.add_action()
                    if goal.__eq__(child_node):
                        return child_node
                    frontier.append(child_node)
        return None

    def dfs(self, start_position, end_position):
        """Depth first search"""
        node = Node(start_position)
        goal = Node(end_position)

        if goal.__eq__(node):
            return node
        frontier = [node]
        explored = set()
        while frontier:
            node = frontier.pop()

            if goal.__eq__(node):
                return node

            explored.add(node)

            for child_position in ADJACENCY_DICT[node.position]:
                child_node = Node(child_position, parent=node)
                if (
                    not self.position_occupied(child_node)
                    and not self.invalid_position(child_node)
                    and child_node not in explored
                    and child_node not in frontier
                ):
                    child_node.add_action()
                    frontier.append(child_node)
        return None

    def unwrap_path(self, node):
        """Given a result node, generate a list of actions that connects"""
        action_queue = deque()
        while node.parent:
            action_queue.appendleft(node.action)
            node = node.parent
        return action_queue


ADJACENCY_DICT = {key: [] for key in product(range(GRID_NUM), repeat=3)}
# More structured adjacency dict creation, meaning that the adjacent positions are ordered
for pos, content in ADJACENCY_DICT.items():
    content.append(tuple(map(lambda i, j: i - j, pos, (1, 0, 0))))
    content.append(tuple(map(lambda i, j: i - j, pos, (-1, 0, 0))))
    content.append(tuple(map(lambda i, j: i - j, pos, (0, 1, 0))))
    content.append(tuple(map(lambda i, j: i - j, pos, (0, -1, 0))))
    content.append(tuple(map(lambda i, j: i - j, pos, (0, 0, 1))))
    content.append(tuple(map(lambda i, j: i - j, pos, (0, 0, -1))))


# Alternative adj
# def create_adjacency_dict(units_across_dim):
#     space = list(product(range(units_across_dim), repeat=3))
#     adjacency_dict = {pos: [] for pos in space}
#     for pos in space:
#         x, y, z = pos

#         subset = []
#         for i in [-1, 1]:
#             subset.append((x + i, y, z))
#             subset.append((x, y + i, z))
#             subset.append((x, y, z + 1))

#         for coords in subset:
#             # Check validity
#             x_, y_, z_ = coords
#             if (
#                 (x_ >= 0)
#                 and (y_ >= 0)
#                 and (z_ >= 0)
#                 and (x_ < units_across_dim)
#                 and (y_ < units_across_dim)
#                 and (z_ < units_across_dim)
#             ):
#                 if coords not in adjacency_dict[pos]:
#                     adjacency_dict[pos].append(coords)
#                 if pos not in adjacency_dict[coords]:
#                     adjacency_dict[coords].append(pos)

#     return adjacency_dict
