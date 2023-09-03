# %%
import time
from shutil import rmtree
import os
from typing import List
from search import Agent
import numpy as np
from py5 import Sketch
from itertools import product
from random import choice
from datetime import datetime
from PIL import Image
import imageio
from yaml import safe_load

with open("./config.yaml", "r") as f:
    config = safe_load(f)


config["environment"]["render_options"]["start_position"] = (
    config["window"]["height"] // 2
)

config["environment"]["render_options"]["unit_size_half"] = (
    config["environment"]["render_options"]["unit_size"] // 2
)
config["environment"]["render_options"]["arena_size"] = (
    config["environment"]["grid_num"]
    * config["environment"]["render_options"]["unit_size"]
)

candidate_coordinates = set(product(range(config["environment"]["grid_num"]), repeat=3))


# Basic building block for rendering
class Block:
    """
    Represents a basic block structure with 3D coordinates.
    """

    def __init__(self, x, y, z) -> None:
        """
        Initialize a Block object.

        :param x: The x-coordinate of the block.
        :param y: The y-coordinate of the block.
        :param z: The z-coordinate of the block.
        """
        self.x = x
        self.y = y
        self.z = z

    def as_numpy(self) -> np.array:
        """
        Converts the block's coordinates into a numpy array.

        :return: Numpy array of coordinates [x, y, z].
        """
        return np.array([self.x, self.y, self.z])

    def as_list(self) -> list:
        """
        Converts the block's coordinates into a list.

        :return: List of coordinates [x, y, z].
        """
        return [self.x, self.y, self.z]

    def as_tuple(self) -> tuple:
        """
        Converts the block's coordinates into a tuple.

        :return: Tuple of coordinates (x, y, z).
        """
        return (self.x, self.y, self.z)

    def __eq__(self, other: object) -> bool:
        """
        Checks if the current block's coordinates are equal to another block's coordinates.

        :param other: another Block object.
        :return: True if coordinates are the same, otherwise False.
        """
        if not isinstance(other, Block):
            raise ValueError
        return (self.x, self.y, self.z) == (other.x, other.y, other.z)

    def __repr__(self) -> str:
        """
        Provides a string representation of the block.

        :return: String representation of the block's coordinates "x y z".
        """
        return "%s %s %s" % (self.x, self.y, self.z)

    def update(self, new_coordinates: np.array) -> None:
        """
        Updates the block's coordinates using a numpy array.

        :param new_coordinates: A numpy array containing new coordinates [new_x, new_y, new_z].
        """
        self.x = new_coordinates[0]
        self.y = new_coordinates[1]
        self.z = new_coordinates[2]


# %%
class Snake:

    """
    Represents a 3D snake in a game environment, capable of moving in 6 directions.

    Attributes:
        directions (dict): Dictionary mapping the direction names to their corresponding coordinate shifts.
        head (Block): The front block of the snake representing its head.
        head_direction (tuple): Direction in which the snake's head is currently moving.
        status (bool): Alive status of the snake.
        state (bool): Signaling the eating state (0 or 1).
        length (int): Current length of the snake.
        tail (list[Block]): List of blocks representing the snake's tail.
    """

    directions = dict(
        x_left=(-1, 0, 0),
        x_right=(1, 0, 0),
        y_up=(0, -1, 0),
        y_down=(0, 1, 0),
        z_forward=(0, 0, 1),
        z_backward=(0, 0, -1),
    )

    def __init__(self, initial_block, grid_num) -> None:
        """
        Initialize the Snake object with an initial block.

        :param initial_block: The initial block representing the snake's head.
        """
        self.head = initial_block
        self.head_direction: tuple = self.directions[
            np.random.choice(list(self.directions.keys()))
        ]
        self._arena_grid_num = grid_num
        # Alive status
        self.status: bool = True
        # Signaling eating
        self.state: bool = 0
        # Snake length
        self.length: int = 1
        # List of snake blocks
        self.tail: List[Block] = []

    def snake_blocks_as_list(self) -> List[tuple]:
        """
        Return the snake blocks as a list of tuples.

        :return: List[tuple] representation of the snake's blocks.
        """
        if self.tail:
            return [self.head.as_tuple(), *[block.as_tuple() for block in self.tail]]
        else:
            return [self.head.as_tuple()]

    def assign_direction(self, action: tuple) -> None:
        """
        Assign a new direction to the snake head based on the given action.

        :param action: Tuple representing the new direction.
        """
        self.head_direction = action

    def update_snake(self) -> None:
        """Proceed with an environment step; updates the head and tail block coordinates based on the head direction"""

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

    def check_out_of_bounds(self) -> None:
        """
        Check if the snake's head has gone out of the game's grid.
        If it has, set the snake's status to dead.
        """
        head_coordinates = self.head.as_numpy()
        if not (
            (0 <= head_coordinates) & (head_coordinates < self._arena_grid_num)
        ).all():
            self.status = False

    def check_self_collision(self) -> None:
        """
        Check for collision between the updated snake's head and any part of its tail.
        If there's a collision, set the snake's status to dead.
        """
        collision = False
        if self.length > 1:
            tail_len = self.length - 1
            tail_idx = 0
            while (not collision) and tail_idx < tail_len:
                if self.head.__eq__(self.tail[tail_idx]):
                    self.status = False
                tail_idx += 1


# %%
class Environment(Sketch):
    """
    Represents the environment in which the snake operates, handling its movement, food spawning, and interactions.
    """

    def __init__(self, agent: Agent = None, **config) -> None:
        """
        Initialize the environment with optional agent and frame rate.
        :param agent: An autonomous agent that guides the snake. Default is None.
        """
        super().__init__()
        # self.space_representation = np.zeros((self._grid_num, self.grid_num, self.grid_num))
        # self.snake = Snake(self.generate_empty_block())

        # Sequnce of actions generated by the agent, that will be executed step-by-step
        self.action_queue = []

        # Initialize autonomous agent
        self.agent = agent
        self._origon = config["environment"]["render_options"]["start_position"]
        self._grid_num = config["environment"]["grid_num"]
        self._display_frame_rate = config["environment"]["frame_rate"]
        self._render_video = config["environment"]["video"]["render_video"]
        self._video_output_folder_path = config["environment"]["video"]["output_folder"]
        self._frame_count_divisor = config["environment"]["frame_count_divisor"]
        self._render_frame_rate = config["environment"]["video"]["frame_rate"]
        self._render_arena_size = config["environment"]["render_options"]["arena_size"]
        self._render_unit_size = config["environment"]["render_options"][
            "render_unit_size"
        ]
        self._unit_block_stroke = config["environment"]["render_options"][
            "block_stroke"
        ]
        self._unit_size = config["environment"]["render_options"]["unit_size"]
        self._unit_size_half = config["environment"]["render_options"]["unit_size_half"]
        self._info_offset_x = config["environment"]["info"]["offset_x"]
        self._info_offset_y = config["environment"]["info"]["offset_y"]
        self._snake_primary_color = self.color(0, 255, 0, 25)
        self._snake_secondary_color = self.color(200, 255, 0, 50)
        self._food_primary_color = self.color(255, 0, 0, 50)
        self._food_secondary_color = self.color(255, 0, 0, 25)
        self._unit_stroke_weight = config["environment"]["render_options"][
            "unit_stroke_weight"
        ]
        self._grid_stroke_weight = 0.25
        self._some_stroke_value = 200
        # Logging metrics
        self.max_sl = 0
        self.min_sl = 100

        self.snake = Snake(Block(5, 5, 5), self._grid_num)
        self.food = self.generate_empty_block()

        if self._render_video:
            # Create a folder for the rendered images
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if not os.path.exists(self._video_output_folder_path):
                os.makedirs(self._video_output_folder_path)
            self._run_name = f"{self.agent.agent_type.lower()}_{timestamp}"
            self._run_export_path = os.path.join(
                self._video_output_folder_path, self._run_name
            )

            os.makedirs(self._run_export_path)

    def reset(self):
        """
        Reset the environment to an initial state.
        """
        # self.space_representation = np.zeros((self._grid_num, self.grid_num, self.grid_num))
        self.snake = Snake(Block(5, 5, 5))
        # self.update_space_representation()
        self.food = self.generate_empty_block()

    def generate_empty_block(self):
        """
        Generate a block position in the grid that is currently empty.

        :return: Block object representing an empty block in the grid.
        """
        if self.snake.tail:
            tail_set = set([block.as_tuple() for block in self.snake.tail])
            tail_set.add(self.snake.head.as_tuple())
        else:
            tail_set = set(self.snake.head.as_tuple())
        empty_x, empty_y, empty_z = choice(
            list(candidate_coordinates.difference(tail_set))
        )
        empty_block = Block(empty_x, empty_y, empty_z)

        return empty_block

    def step(self, action):
        """
        Execute one step of the environment based on a provided action.

        :param action: Action to be performed by the snake.
        """
        self.snake.assign_direction(action)
        self.snake.update_snake()
        self.check_food_collision()

    def select_direction(self):
        """
        Determine and select the direction of the snake based on the agent type (manual or autonomous).
        """
        if self.agent.agent_type != "HUMAN":
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
            if self.frame_count % self._frame_count_divisor == 0:
                self.step(self.snake.head_direction)

    def check_food_collision(self):
        """
        Check if the snake has collided with the food and handle the collision.
        """
        if self.snake.head.__eq__(self.food):
            self.snake.state = 1  # Signaling that the snake ate an apple
            self.food = self.generate_empty_block()

    def run_headless(self):
        """
        Execute the snake's movement without rendering the graphical interface.

        :return: Length of the snake after it has stopped moving.
        """
        while self.snake.status:
            self.select_direction()

        return self.snake.length

    def generate_video(self):
        """
        Convert images in a folder into a video.

        Parameters:
        - img_folder: The path to the folder containing images.
        - output_video: The path to the output video file.
        - fps: Frames per second for the resulting video.
        """

        def delete_folder(path):
            """
            Delete a folder and its contents.

            Parameters:
            - path: The path to the folder to be deleted.
            """
            # Check if the directory exists
            if not os.path.exists(path):
                print(f"The directory {path} does not exist.")
                return

            # Use shutil.rmtree() to remove the directory and its contents
            rmtree(path)
            print(f"Directory {path} and its contents have been removed.")

        img_folder = self._run_export_path
        output_video = self._run_export_path + ".mp4"

        images = sorted(
            [
                os.path.join(img_folder, img)
                for img in os.listdir(img_folder)
                if img.endswith((".png", ".jpg", ".jpeg"))
            ]
        )

        if not images:
            raise ValueError("No images found in the specified directory!")

        with imageio.get_writer(
            output_video, mode="I", fps=self._render_frame_rate
        ) as writer:
            for img_path in images:
                img = Image.open(img_path)
                img_array = np.array(img)

                writer.append_data(img_array)

        print(f"Video {output_video} created successfully!")
        delete_folder(self._run_export_path)

    def save_frame_to_disk(self):
        self.save_frame(
            filename=f"{self._run_export_path}/frame_#####.png",
            drop_alpha=False,
            use_thread=True,
            format="png",
        )

    def settings(self):
        """
        Define the visual settings for rendering the environment.
        """
        self.size(1914, 2104, self.P3D)
        self.smooth(8)

    def setup(self):
        """
        Set up the environment and initialize the graphical elements.
        """

        self.frame_rate(self._display_frame_rate)
        self.rect_mode(2)
        camera = self.camera()

        self.start_time = time.time()

    def draw(self):
        """
        Render the environment, snake, food, and other graphical elements.
        """

        if self._render_video:
            self.save_frame_to_disk()
        self.select_direction()

        self.background(255)
        self.draw_arena()

        self.draw_snake_head(self.snake)
        self.draw_snake_tail(self.snake)

        self.draw_food()

        self.display_info(
            self._origon - self._info_offset_x,
            self._origon - self._info_offset_y,
        )
        if self.snake.status == False:
            if self._render_video:
                self.generate_video()
            self.exit_sketch()

    def draw_snake_head(self, snake):
        # Render head
        self.draw_location_support(snake.head, self._snake_primary_color)
        self.draw_support_lines_head(snake)
        self.draw_block(
            snake.head,
            self._snake_primary_color,
        )

    def draw_snake_tail(self, snake):
        # Render tail
        for tail_block in snake.tail:
            self.draw_block(
                tail_block,
                self._snake_secondary_color,
            )

    def draw_food(self):
        # Render food
        self.draw_block(
            self.food,
            self._food_primary_color,
        )
        self.draw_location_support(self.food, self._food_secondary_color)

    def draw_block(self, block, block_color):
        """
        Draw a block at a specific position with specified visuals.

        :param block: Block object to be rendered.
        :param block_size: Size of the block to be drawn.
        :param color: Color to fill the block with.
        :param no_strokes: If True, the block will have no outline.
        """
        self.push()
        self.fill(block_color)
        if self._unit_block_stroke:
            self.no_stroke()

        self.translate(
            self._origon + block.x * self._unit_size + self._unit_size_half,
            self._origon + block.y * self._unit_size + self._unit_size_half,
            block.z * self._unit_size + self._unit_size_half,
        )
        self.stroke_weight(self._unit_stroke_weight)
        self.box(self._render_unit_size)

        self.pop()

    def display_info(self, offset_x, offset_y):
        """
        Display the environment's metrics on the screen.

        :param offset: Positional offset to begin displaying metrics.
        """
        fps = self.frame_count // (time.time() - self.start_time)
        snake_length = self.snake.length
        max_steps_per_length = self.max_sl
        steps_per_length = self.frame_count // snake_length

        if steps_per_length > max_steps_per_length:
            self.max_sl = steps_per_length

        metrics = {
            "agent-type": self.agent.agent_type,
            "frame count": self.frame_count,
            "fps": fps,
            "snake length": snake_length,
            "steps/length": steps_per_length,
            "max steps/length": self.max_sl,
        }

        for i, (text, value) in enumerate(metrics.items()):
            self.push()
            self.fill(self.color(100, 100, 100))
            self.text_size(24)
            self.text(
                f"""{text}: {value:.2f}"""
                if not text == "agent-type"
                else f"""{text}: {value}""",
                offset_x - self._unit_size,
                offset_y + i * self._unit_size,
                0,
            )

            self.pop()

    def draw_arena(self):
        """
        Render the 3D arena grid in which the snake operates.
        """

        min_x = self._origon
        max_x = self._origon + self._render_arena_size
        min_y = self._origon
        max_y = self._origon + self._render_arena_size
        min_z = 0
        max_z = self._render_arena_size

        self.push()
        self.stroke_weight(self._grid_stroke_weight)

        grids_xy = np.linspace(min_x, max_x, self._grid_num + 1)
        grids_z = np.linspace(min_z, max_z, self._grid_num + 1)

        # Y plane
        self.stroke(self._some_stroke_value)

        # for y in grids_xy:
        #     self.line(min_x, y, min_z, max_x, y, min_z)

        # for x in grids_xy:
        #     self.line(x, min_y, min_z, x, max_y, min_z)

        # # Z Plane
        # for y in grids_xy:
        #     self.line(max_x, y, min_z, max_x, y, max_z)

        # for z in grids_z:
        #     self.line(max_x, min_y, z, max_x, max_y, z)

        # # X Plane
        # for x in grids_xy:
        #     self.line(x, max_y, min_z, x, max_y, max_z)

        # for z in grids_z:
        #     self.line(min_x, max_y, z, max_x, max_y, z)

        for x, y, z in zip(grids_xy, grids_xy, grids_z):
            self.line(min_x, y, min_z, max_x, y, min_z)
            self.line(x, min_y, min_z, x, max_y, min_z)
            self.line(max_x, y, min_z, max_x, y, max_z)
            self.line(max_x, min_y, z, max_x, max_y, z)
            self.line(x, max_y, min_z, x, max_y, max_z)
            self.line(min_x, max_y, z, max_x, max_y, z)

        self.pop()

    def draw_support_lines_head(self, snake):
        """
        Draw supporting visual lines that extend from the snake's head to the arena boundaries.
        """
        self.push()
        self.line(
            self._origon + snake.head.x * self._unit_size + self._unit_size_half,
            self._origon + snake.head.y * self._unit_size + self._unit_size_half,
            snake.head.z * self._unit_size + self._unit_size_half,
            self._origon + self._render_arena_size,
            self._origon + snake.head.y * self._unit_size + self._unit_size_half,
            snake.head.z * self._unit_size + self._unit_size_half,
        )

        self.line(
            self._origon + snake.head.x * self._unit_size + self._unit_size_half,
            self._origon + snake.head.y * self._unit_size + self._unit_size_half,
            snake.head.z * self._unit_size + self._unit_size_half,
            self._origon + snake.head.x * self._unit_size + self._unit_size_half,
            self._origon + self._render_arena_size,
            snake.head.z * self._unit_size + self._unit_size_half,
        )

        self.line(
            self._origon + snake.head.x * self._unit_size + self._unit_size_half,
            self._origon + snake.head.y * self._unit_size + self._unit_size_half,
            self.snake.head.z,
            self._origon + snake.head.x * self._unit_size + self._unit_size_half,
            self._origon + snake.head.y * self._unit_size + self._unit_size_half,
            snake.head.z * self._unit_size + self._unit_size_half,
        )
        self.pop()

    def draw_location_support(self, block, color):
        """
        Draw supporting visuals to highlight a specific block's position.
        The tiles on the arena margins light up.

        :param block: Block object to be highlighted.
        :param color: Color to highlight the block with.
        """
        self.fill(color)
        self.push()
        self.translate(
            self._origon + block.x * self._unit_size + self._unit_size_half,
            self._origon + block.y * self._unit_size + self._unit_size_half,
            0,
        )
        self.no_stroke()
        self.box(
            self._unit_size,
            self._unit_size,
            0,
        )
        self.pop()

        self.push()
        self.translate(
            self._origon + block.x * self._unit_size + self._unit_size_half,
            self._origon + self._render_arena_size,
            block.z * self._unit_size + self._unit_size_half,
        )
        self.no_stroke()
        self.box(
            self._unit_size,
            0,
            self._unit_size,
        )
        self.pop()

        self.push()
        self.translate(
            self._origon + self._render_arena_size,
            self._origon + block.y * self._unit_size + self._unit_size_half,
            block.z * self._unit_size + self._unit_size_half,
        )
        self.no_stroke()
        self.box(
            0,
            self._unit_size,
            self._unit_size,
        )
        self.pop()

    def key_pressed(self):
        """
        Handle key press events to control the direction of the snake manually.
        """
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

        elif self.key == "q":
            if self._render_video:
                self.generate_video()

            self.exit_sketch()
