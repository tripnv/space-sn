from itertools import product
from typing import Any, List
from collections import deque


GRID_NUM = 10


class Node:
    """
    Represents a node with a specific position in a 3D grid.

    :param position: Tuple representing a 3D position in the grid.
    :param parent: Parent node of the current node.
    :param action: Action taken to reach this node from its parent.
    :param gx: Distance from a start position
    :param hx: Distance from an end position
    """

    def __init__(self, position: tuple, parent=None, action=None, gx=0, hx=0) -> None:
        self.position = position
        self.parent = parent
        self.action = action
        self.depth = 0
        if self.parent:
            self.depth += 1

    def __repr__(self) -> str:
        """
        String representation of the Node object.

        :return: String representing the node's position.
        """
        return f"Node{self.position}"

    def __eq__(self, __value) -> bool:
        """
        Check if the position of the Node is equal to that of another node.

        :param __value: The other node to compare against.
        :return: True if positions are the same, False otherwise.
        """
        return isinstance(__value, Node) and self.position == __value.position

    def expand(self):
        """
        List all nodes one step away from the current node's position.

        :return: List of adjacent nodes.
        """
        return [Node(neighbour) for neighbour in ADJACENCY_DICT[self.position]]

    def __hash__(self) -> int:
        """
        Compute hash of the node based on its position.

        :return: Hash value of the node.
        """
        return hash(self.position)

    def add_action(self) -> None:
        """
        Assign the action taken to move from the parent node to the current node.
        Modifies self.action.
        """
        child_x, child_y, child_z = self.position
        parent_x, parent_y, parent_z = self.parent.position

        self.action = (child_x - parent_x, child_y - parent_y, child_z - parent_z)


class Agent:
    """
    Represents an agent capable of pathfinding in a 3D grid.

    :param agent_type: Type of agent (e.g., BFS, DFS, A*).
    """

    def __init__(self, agent_type: str) -> None:
        self.agent_type = agent_type.upper()

    def position_occupied(self, node):
        """
        Check if a given position is occupied. Basically checking for the grid cubes occupied by the snake itself.
        In a future version could include obstacles as well.

        :param node: Node representing the position to check.
        :return: True if position is occupied, False otherwise.
        """
        for block in self.occupied_positions:
            if block.__eq__(node):
                return True
        return False

    def invalid_position(self, node):
        """
        Check if a given position is invalid (outside of grid limits).

        :param node: Node representing the position to check.
        :return: True if position is invalid, False otherwise.
        """
        x, y, z = node.position
        if x < 0 or x > GRID_NUM - 1:
            return True
        if y < 0 or y > GRID_NUM - 1:
            return True
        if z < 0 or z > GRID_NUM - 1:
            return True
        return False

    def select_available_position(self, start_position):
        """Return the first available child node; if no such child has been found select the first adjacent position

        :param start_position: Position to start the search from.
        :return: Available child node closest to the start position.
        """
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
        """
        Perform breadth-first search to find a path in the grid.

        :param start_position: Starting position tuple.
        :param end_position: Ending position tuple.
        :return: Result node after performing BFS.
        """
        current_node = Node(start_position)
        goal_node = Node(end_position)

        # Test goal_node state
        if goal_node.__eq__(current_node):
            return current_node

        frontier = deque([current_node])
        explored = set()
        while frontier:
            current_node = frontier.popleft()
            explored.add(current_node)

            # Nodes that are reachable in one step
            for child_position in ADJACENCY_DICT[current_node.position]:
                child_node = Node(child_position, parent=current_node)
                if (
                    not self.position_occupied(child_node)
                    and not self.invalid_position(child_node)
                    and child_node not in explored
                    and child_node not in frontier
                ):
                    child_node.add_action()
                    if goal_node.__eq__(child_node):
                        return child_node
                    frontier.append(child_node)
        return None

    def dfs(self, start_position, end_position):
        """
        Perform depth-first search to find a path in the grid.

        :param start_position: Starting position tuple.
        :param end_position: Ending position tuple.
        :return: Result node after performing DFS.
        """
        current_node = Node(start_position)
        goal_node = Node(end_position)

        if goal_node.__eq__(current_node):
            return current_node
        frontier = [current_node]
        explored = set()
        while frontier:
            current_node = frontier.pop()

            if goal_node.__eq__(current_node):
                return current_node

            explored.add(current_node)

            for child_position in ADJACENCY_DICT[current_node.position]:
                child_node = Node(child_position, parent=current_node)
                if (
                    not self.position_occupied(child_node)
                    and not self.invalid_position(child_node)
                    and child_node not in explored
                    and child_node not in frontier
                ):
                    child_node.add_action()
                    frontier.append(child_node)
        return None

    def calculate_euclidean_distance(self, node_a, node_b):
        """
        Calculate the Euclidean distance between two nodes in 3D space.

        :param node_a: First node coordinates.
        :param node_b: Second node coordinates.
        :return: Euclidean distance between the two nodes.
        """
        x_a, y_a, z_a = node_a.position
        x_b, y_b, z_b = node_b.position

        distance = ((x_b - x_a) ** 2 + (y_b - y_a) ** 2 + (z_b - z_a) ** 2) ** 0.5

        return distance

    def calculate_manhattan_distance(self, node_a, node_b):
        """
        Calculate the Manhattan distance between two nodes in 3D space.

        :param node_a: First node coordinates.
        :param node_b: Second node coordinates.
        :return: Manhattan distance between the two nodes.
        """
        x_a, y_a, z_a = node_a.position
        x_b, y_b, z_b = node_b.position

        distance = abs(x_b - x_a) + abs(y_b - y_a) + abs(z_b - z_a)
        return distance

    def fx(self, start_node, current_node, end_node):
        """
        F(x) = g(x) + h(x)
        where g(x) returns the distance of current_node from the start_node
        and h(x) returns the distance to the end_node
        """
        gx = self.calculate_euclidean_distance(start_node, current_node)
        hx = self.calculate_euclidean_distance(current_node, end_node)
        return gx + hx

    def fx_greedy(self, _, current_node, end_node):
        hx = self.calculate_euclidean_distance(current_node, end_node)
        return hx

    def best_first_search(self, start_position, end_position, heuristic):
        """
        Perform best-first search to find a path in the grid recursively.

        :param start_position: Starting position tuple.
        :param end_position: Ending position tuple.
        :return: Result node after performing BFS.
        """
        start_node = Node(start_position)
        goal_node = Node(end_position)

        if goal_node.__eq__(start_node):
            return start_node

        explored = set()
        return self._recursive_best_first(
            start_node, start_node, goal_node, explored, heuristic
        )

    def _recursive_best_first(
        self, start_node, current_node, goal_node, explored, heuristic
    ):
        if goal_node.__eq__(current_node):
            return current_node

        explored.add(current_node)

        # Generate child nodes
        children = []
        for child_position in ADJACENCY_DICT[current_node.position]:
            child_node = Node(child_position, parent=current_node)

            if (
                not self.position_occupied(child_node)
                and not self.invalid_position(child_node)
                and child_node not in explored
            ):
                child_node.add_action()
                children.append(child_node)

        if not children:
            return None

        # Sort child nodes based on their heuristic value, with the lowest first
        children.sort(
            # key=lambda node: self.calculate_manhattan_distance(node, goal_node)
            key=lambda child_node: heuristic(start_node, child_node, goal_node)
        )

        # Recursively explore the node with the lowest heuristic value first
        return self._recursive_best_first(
            start_node, children[0], goal_node, explored, heuristic
        )

    def a_star(self, start_position, end_position):
        # Best first search with an fx = gx + hx heuristic
        return self.best_first_search(start_position, end_position, heuristic=self.fx)

    def unwrap_path(self, node):
        """
        Extract the path from a result node back to the start.

        :param node: Result node to extract the path from.
        :return: List of actions to move from the start to the result node.
        """
        action_queue = deque()
        while node.parent:
            action_queue.appendleft(node.action)
            node = node.parent
        return action_queue

    def generate_path(
        self,
        start_position: tuple,
        end_position: tuple,
        occupied_positions: List[tuple],
    ):
        """
        Generate a sequence of actions from start to end avoiding occupied positions.
        
        :param start_position: Starting position tuple.
        :param end_position: Ending position tuple.
        :param occupied_positions: List of positions that are occupied.
        :return: List of actions to move from start to end.
        """ """Generate the sequence of actions that connect the start position to the end position such that the occupied squares are avoided."""
        self.occupied_positions = [Node(block) for block in occupied_positions]
        # Agent scenarios
        if self.agent_type == "BFS":
            search_final_node = self.bfs(start_position, end_position)

        elif self.agent_type == "DFS":
            search_final_node = self.dfs(start_position, end_position)
            # if search_final_node == None:
            #     search_final_node = self.dfs(
            #         start_position, self.occupied_positions[-1]
            #     )

        elif self.agent_type == "BEST-FIRST":
            search_final_node = self.best_first_search(
                start_position, end_position, self.fx_greedy
            )

        elif self.agent_type == "ASTAR":
            search_final_node = self.a_star(
                start_position,
                end_position,
            )
        else:
            raise NotImplementedError

        if search_final_node == None:
            search_final_node = self.select_available_position(start_position)

        return self.unwrap_path(search_final_node)


ADJACENCY_DICT = {key: [] for key in product(range(GRID_NUM), repeat=3)}
# More structured adjacency dict creation, meaning that the adjacent positions are ordered
for pos, content in ADJACENCY_DICT.items():
    content.append(tuple(map(lambda i, j: i - j, pos, (1, 0, 0))))
    content.append(tuple(map(lambda i, j: i - j, pos, (-1, 0, 0))))
    content.append(tuple(map(lambda i, j: i - j, pos, (0, 1, 0))))
    content.append(tuple(map(lambda i, j: i - j, pos, (0, -1, 0))))
    content.append(tuple(map(lambda i, j: i - j, pos, (0, 0, 1))))
    content.append(tuple(map(lambda i, j: i - j, pos, (0, 0, -1))))


# Alternative adj, order of adjacent positions is not consistent
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
