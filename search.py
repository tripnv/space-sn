from itertools import product


ADJACENCY_DICT = {key: [] for key in product(range(10), repeat=3)}

for pos, content in ADJACENCY_DICT.items():
    content.append(tuple(map(lambda i, j: i - j, pos, (1, 0, 0))))
    content.append(tuple(map(lambda i, j: i - j, pos, (-1, 0, 0))))
    content.append(tuple(map(lambda i, j: i - j, pos, (0, 1, 0))))
    content.append(tuple(map(lambda i, j: i - j, pos, (0, -1, 0))))
    content.append(tuple(map(lambda i, j: i - j, pos, (0, 0, 1))))
    content.append(tuple(map(lambda i, j: i - j, pos, (0, 0, -1))))


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
