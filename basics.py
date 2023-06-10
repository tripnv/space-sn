# %%
import numpy as np
from py5 import Sketch


# %%
# Basic building block
class Block:
    def __init__(self, x, y, z) -> None:
        self.x = x
        self.y = y
        self.z = z

    def as_numpy(self) -> np.array:
        return np.array([self.x, self.y, self.z])

    def update(self, new_coordinates: np.array) -> None:
        self.x = new_coordinates[0]
        self.y = new_coordinates[1]
        self.z = new_coordinates[2]


# %%
class Snake:
    directions = dict(
        x_left=(-1, 0, 0),
        x_right=(1, 0, 0),
        y_up=(0, -1, 0),
        y_down=(0, 1, 0),
        z_forward=(0, 0, 1),
        z_backward=(0, 0, -1),
    )

    def __init__(self, initial_block) -> None:
        self.head = initial_block
        self.head_direction = self.directions[
            np.random.choice(list(self.directions.keys()))
        ]
        self.status = True
        self.length = 1
        self.tail = []

    def add_tail_block(self, block):
        self.tail.append(block)

    def check_status(self):
        # Out of bounds check
        head_coordinates = self.head.as_numpy()
        if not ((0 <= head_coordinates) & (head_coordinates < GRID_NUM)).all():
            self.status = False

        # Self-bite check ??
        # Hesitating about the case when it's right behind itself
        # Check probably needs to be split
        # Self bite check before update, oob after update


screen_height = 1914
screen_width = 2104

middle_y = 1914 // 2

colors = {"red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255)}

GRID_NUM = 10

#
UNIT_SIZE = 50
# Since for the render is based on the center of the box
UNIT_HALF = UNIT_SIZE // 2
ARENA_SIZE = GRID_NUM * UNIT_SIZE

start_position = middle_y


# global block
# block = Block(start_position + UNIT_HALF, start_position + UNIT_HALF, UNIT_HALF)
# snake = Snake()


class SpaceSnake(Sketch):
    def __init__(self, grid_num) -> None:
        super().__init__()
        self.space_representation = np.zeros((grid_num, grid_num, grid_num))
        # self.snake = Snake(self.generate_empty_block())
        self.snake = Snake(Block(0, 9, 9))

    def generate_empty_block(self):
        """Given a 3d board representation return the coordinates of the empty block"""
        empty_positions = np.where(self.space_representation == 0)
        num_candidate_positions = empty_positions[0].shape[0]

        random_index = np.random.choice(num_candidate_positions, 1)

        empty_block = Block(
            x=empty_positions[0][random_index],
            y=empty_positions[1][random_index],
            z=empty_positions[2][random_index],
        )
        return empty_block

    def update_space_representation(self):
        ...

    def settings(self):
        self.size(1914, 2104, self.P3D)
        self.smooth(4)

    def setup(self):
        self.frame_rate(60)
        self.rect_mode(2)
        camera = self.camera()

    def draw(self):
        # c1 = self.color(0)
        # self.fill(c1)
        # self.stroke_weight(10)
        self.background(255)
        grid_size = self.draw_arena(start_position, ARENA_SIZE, GRID_NUM)

        color = self.color(0, 255, 0, 25)
        # Render head
        self.draw_location_support()

        self.push()
        self.fill(color)
        self.draw_support_lines()

        # self.translate(self.snake.head.x, self.snake.head.y, self.snake.head.z)

        self.translate(
            start_position + self.snake.head.x * UNIT_SIZE + UNIT_HALF,
            start_position + self.snake.head.y * UNIT_SIZE + UNIT_HALF,
            self.snake.head.z * UNIT_SIZE + UNIT_HALF,
        )
        # self.fill(color)
        # self.no_fill()
        self.no_stroke()
        self.box(UNIT_SIZE - 2)

        self.pop()

        # Render tail

    def draw_arena(self, starting_distance, arena_size, grid_num):
        # Centre point
        self.stroke_weight(10)
        self.point(starting_distance, starting_distance, 0)

        self.fill(UNIT_SIZE)
        self.text_size(32)
        self.text(
            str(self.snake.head.as_numpy()),
            starting_distance + 20,
            starting_distance - 20,
            0,
        )

        self.text(
            f"Alive: {self.snake.status}",
            starting_distance + 350,
            starting_distance - 20,
            0,
        )

        min_x = starting_distance
        max_x = starting_distance + arena_size
        min_y = starting_distance
        max_y = starting_distance + arena_size
        min_z = 0
        max_z = arena_size

        self.stroke_weight(3)

        # X axis
        self.stroke(self.color(*colors["blue"]))
        self.line(min_x, min_y, min_z, max_x, min_y, min_z)
        # Y axis
        self.stroke(self.color(*colors["red"]))
        self.line(min_x, min_y, min_z, min_x, max_y, min_z)

        self.stroke(self.color(*colors["green"]))
        self.line(min_x, min_y, min_z, min_x, min_y, max_z)

        grid_size = arena_size // grid_num

        grids_xy = np.linspace(min_x, max_x, grid_num + 1)
        grids_z = np.linspace(min_z, max_z, grid_num + 1)

        # Y plane
        self.stroke_weight(1)
        self.stroke(200)
        for y in grids_xy:
            self.line(min_x, y, min_z, max_x, y, min_z)

        for x in grids_xy:
            self.line(x, min_y, min_z, x, max_y, min_z)

        # Z Plane
        for y in grids_xy:
            self.line(max_x, y, min_z, max_x, y, max_z)

        for z in grids_z:
            self.line(max_x, min_y, z, max_x, max_y, z)

        # X Plane
        for x in grids_xy:
            self.line(x, max_y, min_z, x, max_y, max_z)

        for z in grids_z:
            self.line(min_x, max_y, z, max_x, max_y, z)

        return grid_size

    def draw_support_lines(self):
        self.line(
            start_position + self.snake.head.x * UNIT_SIZE + UNIT_HALF,
            start_position + self.snake.head.y * UNIT_SIZE + UNIT_HALF,
            self.snake.head.z * UNIT_SIZE + UNIT_HALF,
            start_position + ARENA_SIZE,
            start_position + self.snake.head.y * UNIT_SIZE + UNIT_HALF,
            self.snake.head.z * UNIT_SIZE + UNIT_HALF,
        )

        self.line(
            start_position + self.snake.head.x * UNIT_SIZE + UNIT_HALF,
            start_position + self.snake.head.y * UNIT_SIZE + UNIT_HALF,
            self.snake.head.z * UNIT_SIZE + UNIT_HALF,
            start_position + self.snake.head.x * UNIT_SIZE + UNIT_HALF,
            start_position + ARENA_SIZE,
            self.snake.head.z * UNIT_SIZE + UNIT_HALF,
        )

        self.line(
            start_position + self.snake.head.x * UNIT_SIZE + UNIT_HALF,
            start_position + self.snake.head.y * UNIT_SIZE + UNIT_HALF,
            self.snake.head.z,
            start_position + self.snake.head.x * UNIT_SIZE + UNIT_HALF,
            start_position + self.snake.head.y * UNIT_SIZE + UNIT_HALF,
            self.snake.head.z * UNIT_SIZE + UNIT_HALF,
        )

    def draw_location_support(self):
        c = self.color(0, 255, 0, 25)
        self.fill(c)
        self.push()
        self.translate(
            start_position + self.snake.head.x * UNIT_SIZE + UNIT_HALF,
            start_position + self.snake.head.y * UNIT_SIZE + UNIT_HALF,
            0,
        )
        self.box(UNIT_SIZE, UNIT_SIZE, 1)
        self.pop()

        self.push()
        self.translate(
            start_position + self.snake.head.x * UNIT_SIZE + UNIT_HALF,
            start_position + ARENA_SIZE,
            self.snake.head.z * UNIT_SIZE + UNIT_HALF,
        )
        self.box(UNIT_SIZE, 1, UNIT_SIZE)
        self.pop()

        self.push()
        self.translate(
            start_position + ARENA_SIZE,
            start_position + self.snake.head.y * UNIT_SIZE + UNIT_HALF,
            self.snake.head.z * UNIT_SIZE + UNIT_HALF,
        )
        self.box(1, UNIT_SIZE, UNIT_SIZE)
        self.pop()

    def key_pressed(self):
        if self.key_code == self.UP:
            # block.y -= UNIT_SIZE
            self.snake.head_direction = self.snake.directions["y_up"]
        elif self.key_code == self.DOWN:
            # block.y += UNIT_SIZE
            self.snake.head_direction = self.snake.directions["y_down"]
        elif self.key_code == self.LEFT:
            # block.x -= UNIT_SIZE
            self.snake.head_direction = self.snake.directions["x_left"]
        elif self.key_code == self.RIGHT:
            # block.x += UNIT_SIZE
            self.snake.head_direction = self.snake.directions["x_right"]
        elif self.key == "w":
            # block.z -= UNIT_SIZE
            self.snake.head_direction = self.snake.directions["z_backward"]
        elif self.key == "s":
            # block.z += UNIT_SIZE
            self.snake.head_direction = self.snake.directions["z_forward"]

        self.snake.head.update(self.snake.head.as_numpy() + self.snake.head_direction)
        self.snake.check_status()


test = SpaceSnake(GRID_NUM)
test.run_sketch()
