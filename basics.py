# %%
from typing import List
from search import Node, ADJACENCY_DICT, Agent
import numpy as np
from py5 import Sketch
from itertools import product
from random import choice
from collections import deque


# Basic building block
class Block:
    def __init__(self, x, y, z) -> None:
        self.x = x
        self.y = y
        self.z = z

    def as_numpy(self) -> np.array:
        return np.array([self.x, self.y, self.z])

    def as_list(self) -> list:
        return [self.x, self.y, self.z]

    def as_tuple(self) -> tuple:
        return (self.x, self.y, self.z)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Block):
            raise ValueError
        return (self.x, self.y, self.z) == (other.x, other.y, other.z)

    def __repr__(self) -> str:
        return "%s %s %s" % (self.x, self.y, self.z)

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
        self.status = True  # Alive status
        self.state = 0  #
        self.length = 1
        self.tail = []

    def snake_blocks_as_list(self):
        if self.tail:
            return [self.head.as_tuple(), *[block.as_tuple() for block in self.tail]]
        else:
            return [self.head.as_tuple()]

    def assign_direction(self, action):
        self.head_direction = action

    def update_snake(self):
        head_previous_position = Block(self.head.x, self.head.y, self.head.z)
        self.head.update(self.head.as_numpy() + self.head_direction)
        if self.length > 1:
            if self.state == 0:
                self.tail.insert(0, head_previous_position)
                self.tail.pop()
            else:
                self.tail.insert(0, head_previous_position)
                self.state = 0
                self.length += 1

        else:
            if self.state == 1:
                self.tail.append(head_previous_position)
                self.state = 0
                self.length += 1

        self.check_self_collision()
        self.check_out_of_bounds()

    def check_out_of_bounds(self):
        # Out of bounds check
        head_coordinates = self.head.as_numpy()
        if not ((0 <= head_coordinates) & (head_coordinates < GRID_NUM)).all():
            self.status = False

    def check_self_collision(self):
        collision = False
        if self.length > 1:
            tail_len = self.length - 1
            tail_idx = 0
            while (not collision) and tail_idx < tail_len:
                if self.head.__eq__(self.tail[tail_idx]):
                    self.status = False
                tail_idx += 1


screen_height = 1914
screen_width = 2104

middle_y = 1914 // 2

# colors = {"red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255)}
FRAME_RATE = 120
FRAME_COUNT_DIVISOR = 15

GRID_NUM = 10
UNIT_SIZE = 50
UNIT_HALF = UNIT_SIZE // 2
ARENA_SIZE = GRID_NUM * UNIT_SIZE

start_position = middle_y
POSSIBLE_COORDINATES = set(product(range(GRID_NUM), repeat=3))
# ADJACENT_COORDS = create_adjacency_dict(GRID_NUM)


# %%
class Environment(Sketch):
    def __init__(self, agent=None) -> None:
        super().__init__()
        self.space_representation = np.zeros((GRID_NUM, GRID_NUM, GRID_NUM))
        # self.snake = Snake(self.generate_empty_block())
        self.snake = Snake(Block(5, 5, 5))
        self.food = self.generate_empty_block()
        self.action_queue = []
        self.agent = agent

    def reset(self):
        self.space_representation = np.zeros((GRID_NUM, GRID_NUM, GRID_NUM))
        self.snake = Snake(Block(5, 5, 5))
        # self.update_space_representation()
        self.food = self.generate_empty_block()

    def generate_empty_block(self):
        if self.snake.tail:
            tail_set = set([block.as_tuple() for block in self.snake.tail])
            tail_set.add(self.snake.head.as_tuple())
        else:
            tail_set = set(self.snake.head.as_tuple())
        empty_x, empty_y, empty_z = choice(
            list(POSSIBLE_COORDINATES.difference(tail_set))
        )
        empty_block = Block(empty_x, empty_y, empty_z)

        return empty_block

    def step(self, action):
        self.snake.assign_direction(action)
        self.snake.update_snake()
        self.check_food_collision()

    def select_direction(self):
        if self.agent:
            if self.action_queue:
                action = self.action_queue.popleft()
                self.step(action)
            else:
                self.action_queue = self.agent.generate_path(
                    self.snake.head.as_tuple(),
                    self.food.as_tuple(),
                    self.snake.snake_blocks_as_list(),
                )

        else:
            if self.frame_count % FRAME_COUNT_DIVISOR == 0:
                self.step(self.snake.head_direction)

    def check_food_collision(self):
        if self.snake.head.__eq__(self.food):
            self.snake.state = 1  # Signaling that the snake ate an apple
            self.food = self.generate_empty_block()

    def headless_run(self):
        while self.snake.status:
            self.select_direction()

        return self.snake.length

    def settings(self):
        self.size(1914, 2104, self.P3D)
        self.smooth(4)

    def setup(self):
        if self.agent:
            self.frame_rate(FRAME_RATE)
        else:
            self.frame_rate(30)
        self.rect_mode(2)
        camera = self.camera()

    def draw(self):
        if self.snake.status == False:
            self.reset()

        self.select_direction()

        c_green_25 = self.color(0, 255, 0, 25)
        c_green_50 = self.color(200, 255, 0, 50)
        c_red_50 = self.color(255, 0, 0, 50)
        c_red_25 = self.color(255, 0, 0, 25)

        self.background(255)
        self.draw_arena(start_position, ARENA_SIZE, GRID_NUM)

        # Render head
        self.draw_location_support(self.snake.head, c_green_25)
        self.draw_support_lines_head()
        self.draw_block(self.snake.head, 45, c_green_25, False)

        # Render tail
        for tail_block in self.snake.tail:
            self.draw_block(tail_block, 45, c_green_50, False)

        # Render food
        self.draw_block(self.food, 45, c_red_50, False)
        self.draw_location_support(self.food, c_red_25)

    def draw_block(self, block, block_size, color, no_strokes):
        self.push()
        if color:
            self.fill(color)
        if no_strokes:
            self.no_stroke()

        self.translate(
            start_position + block.x * UNIT_SIZE + UNIT_HALF,
            start_position + block.y * UNIT_SIZE + UNIT_HALF,
            block.z * UNIT_SIZE + UNIT_HALF,
        )

        self.box(block_size)

        self.pop()

    def display_info(self, offset):
        self.push()
        self.fill(self.color(0, 0, 0))
        self.text_size(32)
        self.text(
            f"""Head: {self.snake.head.__repr__()}
            """,
            offset,
            offset,
            0,
        )

        self.text(
            f"Alive: {self.snake.status}",
            offset,
            offset + UNIT_SIZE * 2,
            0,
        )

        self.text(
            f"Length: {self.snake.length}",
            offset,
            offset + UNIT_SIZE,
            0,
        )

        self.text(
            f"Ate: {self.snake.status}",
            offset,
            offset + UNIT_SIZE * 3,
            0,
        )

        self.pop()

    def draw_arena(self, starting_distance, arena_size, grid_num):
        # Centre point
        # self.stroke_weight(10)
        # self.point(starting_distance, starting_distance, 0)
        min_x = starting_distance
        max_x = starting_distance + arena_size
        min_y = starting_distance
        max_y = starting_distance + arena_size
        min_z = 0
        max_z = arena_size

        grid_size = arena_size // grid_num

        self.push()
        self.stroke_weight(0.5)

        # # X axis
        # self.stroke(self.color(*colors["blue"]))
        # self.line(min_x, min_y, min_z, max_x, min_y, min_z)
        # # Y axis
        # self.stroke(self.color(*colors["red"]))
        # self.line(min_x, min_y, min_z, min_x, max_y, min_z)
        # # Z axis
        # self.stroke(self.color(*colors["green"]))
        # self.line(min_x, min_y, min_z, min_x, min_y, max_z)

        grids_xy = np.linspace(min_x, max_x, grid_num + 1)
        grids_z = np.linspace(min_z, max_z, grid_num + 1)

        # Y plane
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

        self.pop()

        self.display_info(start_position - 250)

    def draw_support_lines_head(self):
        self.push()
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
        self.pop()

    def draw_location_support(self, block, color):
        self.fill(color)
        self.push()
        self.translate(
            start_position + block.x * UNIT_SIZE + UNIT_HALF,
            start_position + block.y * UNIT_SIZE + UNIT_HALF,
            0,
        )
        self.no_stroke()
        self.box(UNIT_SIZE, UNIT_SIZE, 0)
        self.pop()

        self.push()
        self.translate(
            start_position + block.x * UNIT_SIZE + UNIT_HALF,
            start_position + ARENA_SIZE,
            block.z * UNIT_SIZE + UNIT_HALF,
        )
        self.no_stroke()
        self.box(UNIT_SIZE, 0, UNIT_SIZE)
        self.pop()

        self.push()
        self.translate(
            start_position + ARENA_SIZE,
            start_position + block.y * UNIT_SIZE + UNIT_HALF,
            block.z * UNIT_SIZE + UNIT_HALF,
        )
        self.no_stroke()
        self.box(0, UNIT_SIZE, UNIT_SIZE)
        self.pop()

    def key_pressed(self):
        if self.key_code == self.UP:
            self.snake.head_direction = self.snake.directions["y_up"]
        elif self.key_code == self.DOWN:
            self.snake.head_direction = self.snake.directions["y_down"]
        elif self.key_code == self.LEFT:
            self.snake.head_direction = self.snake.directions["x_left"]
        elif self.key_code == self.RIGHT:
            self.snake.head_direction = self.snake.directions["x_right"]
        elif self.key == "w":
            self.snake.head_direction = self.snake.directions["z_backward"]
        elif self.key == "s":
            self.snake.head_direction = self.snake.directions["z_forward"]


# %%
agent = Agent("DFS")
environment = Environment(agent)


# %%
environment.run_sketch()
# %%
# score = environment.headless_run()
# print(score)
