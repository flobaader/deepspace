import random
from typing import Tuple, List, Any, Union
import matplotlib.pyplot as plt

__author__ = "Florian Baader"

import numpy as np
from .game import Game

actions = {0: 'up', 1: 'down', 2: 'idle'}
forbidden_moves = [(0, 1), (1, 0)]


class Pong(Game):
    maxScore: int
    PAD_WIDTH: int
    PAD_HEIGHT: int
    paddle2_pos: List[float]
    paddle1_pos: List[float]
    ball_vel: List[Union[int, Any]]
    game_over: bool
    # The controllable paddle
    paddle2_vel: int
    # The game paddle
    paddle1_vel: int
    l_score: int
    r_score: int
    ball_pos: List[int]
    scored: bool
    grid_size: int
    WIDTH: int
    HEIGHT: int
    snake_length: int

    def __init__(self, width=30, height=20, pad_width=1, pad_height=4, maxScore=10, live_view=False, print_score=False):
        self.print_score = print_score
        self.live_view = live_view
        self.maxScore = maxScore
        self.PAD_WIDTH = pad_width
        self.PAD_HEIGHT = pad_height
        self.HEIGHT = height
        self.WIDTH = width
        self.paddle1_pos = [0, self.HEIGHT / 2]
        self.paddle2_pos = [self.WIDTH - 1, self.HEIGHT / 2]
        self.reset()
        self.state_changed = True

    def reset(self):
        self.ball_pos = [0, 0]
        self.l_score = 0
        self.r_score = 0
        self.paddle1_vel = 0
        self.paddle2_vel = 0
        self.game_over = False
        # Random first ball
        self.ball_init(np.random.randint(2) == 0)

    # helper function that spawns a ball, returns a position vector and a velocity vector
    # if right is True, spawn to the right, else spawn to the left
    def ball_init(self, right):
        self.ball_pos = [self.WIDTH / 2, self.HEIGHT / 2]
        horz = random.randrange(2, 4)
        vert = random.randrange(1, 3)
        if not right:
            horz = - horz
        self.ball_vel = [horz, -vert]

    @property
    def name(self):
        return "Pong"

    @property
    def nb_actions(self):
        return 3

    def move_dummy(self):
        if self.ball_pos[1] > self.paddle1_pos[1] + self.PAD_HEIGHT // 2:
            self.paddle1_vel = 8
        elif self.ball_pos[1] < self.paddle1_pos[1] + self.PAD_HEIGHT // 2:
            self.paddle1_vel = -8
        else:
            self.paddle1_vel = 0

    def update_positions(self):
        # update paddle's vertical position, keep paddle on the screen
        self.move_paddle(self.paddle1_pos, self.paddle1_vel)
        self.move_paddle(self.paddle2_pos, self.paddle2_vel)

        # update ball
        self.ball_pos[0] += self.ball_vel[0]
        self.ball_pos[1] += self.ball_vel[1]



        # ball collision check on top and bottom walls
        if self.ball_pos[1] <= 0 or self.ball_pos[1] >= self.HEIGHT - 1:
            self.ball_vel[1] = - self.ball_vel[1]

        # ball collision check on gutters or paddles
        if self.ball_pos[0] <= 0:
            self.paddle_collision(x_cor=0, y_start=self.paddle1_pos[1], y_end=self.paddle1_pos[1] + self.PAD_HEIGHT)
        if self.ball_pos[0] >= self.WIDTH - 1:
            self.paddle_collision(x_cor=self.WIDTH - 1, y_start=self.paddle2_pos[1],
                                  y_end=self.paddle2_pos[1] + self.PAD_HEIGHT)

        self.fix_ball()

    def paddle_collision(self, x_cor, y_start, y_end):
        if int(self.ball_pos[1]) in range(int(y_start), int(y_end), 1):
            self.ball_vel[0] = -1.1 * self.ball_vel[0]
            self.ball_vel[1] *= 1.1
        else:
            # Scored
            if x_cor == 0:
                # Right player
                self.l_score += 1
                self.ball_init(True)
            else:
                self.r_score += 1
                self.ball_init(False)
            if self.print_score:
                print("Score: ", self.l_score, " : ", self.r_score)

    def move_paddle(self, paddle_pos, paddle_vel):
        next_pos = paddle_pos[1] + paddle_vel
        if 0 < next_pos + self.PAD_HEIGHT < self.HEIGHT:
            paddle_pos[1] += paddle_vel

    def play(self, action):
        assert action in range(3), "Invalid action."
        if action == 0:
            self.paddle2_vel = -8
        elif action == 1:
            self.paddle2_vel = 8
        else:
            self.paddle2_vel = 0
        self.move_dummy()
        self.update_positions()

    def get_state(self):
        canvas = np.zeros((self.WIDTH, self.HEIGHT))

        # Draw Ball
        canvas[int(self.ball_pos[0]), int(self.ball_pos[1])] = 1

        canvas[
        int(self.paddle1_pos[0]): int(self.paddle1_pos[0] + self.PAD_WIDTH),
        int(self.paddle1_pos[1]): int(self.paddle1_pos[1] + self.PAD_HEIGHT),
        ] = -.5

        canvas[
        int(self.paddle2_pos[0]): int(self.paddle2_pos[0] + self.PAD_WIDTH),
        int(self.paddle2_pos[1]): int(self.paddle2_pos[1] + self.PAD_HEIGHT),
        ] = .5

        ca = canvas.transpose()
        if self.live_view:
            plt.imshow(ca)
            plt.pause(0.01)
        return ca

    def get_score(self):
        return self.l_score

    def up(self):
        self.play(0)

    def down(self):
        self.play(1)

    def idle(self):
        self.play(3)

    def is_over(self):
        is_over = self.l_score == self.maxScore or self.r_score == self.maxScore
        if is_over:
            if self.is_won():
                print("The AI won the game!")
            else:
                print("The dummy won!")
        return is_over

    def is_won(self):
        return self.r_score == self.maxScore

    def fix_ball(self):
        if self.ball_pos[0] > self.WIDTH - 1:
            self.ball_pos[0] = self.WIDTH - 1
        if self.ball_pos[0] < 0:
            self.ball_pos[0] = 0
        if self.ball_pos[1] > self.HEIGHT - 1:
            self.ball_pos[1] = self.HEIGHT - 1
        if self.ball_pos[1] < 0:
            self.ball_pos[1] = 0
