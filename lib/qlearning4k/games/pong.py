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
    HALF_PAD_HEIGHT: float
    HALF_PAD_WIDTH: float
    game_over: bool
    BALL_RADIUS: int
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

    def __init__(self, width=30, height=20, pad_width=2, pad_height=5, maxScore=10):
        self.maxScore = maxScore
        self.PAD_WIDTH = pad_width
        self.PAD_HEIGHT = pad_height
        self.HALF_PAD_WIDTH = pad_width // 2
        self.HALF_PAD_HEIGHT = pad_height // 2
        self.HEIGHT = height
        self.WIDTH = width
        self.paddle1_pos = [self.HALF_PAD_WIDTH - 1, self.HEIGHT // 2]
        self.paddle2_pos = [self.WIDTH - 1 - self.HALF_PAD_WIDTH, self.HEIGHT // 2]
        self.BALL_RADIUS = 2
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
        self.ball_pos = [self.WIDTH // 2, self.HEIGHT // 2]
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


    def move_PC(self):
        if self.ball_pos[1] > self.paddle1_pos[1]:
            self.paddle1_vel = 8
        else:
            self.paddle1_vel = -8


    def updatePositions(self):
        # update paddle's vertical position, keep paddle on the screen
        if self.HALF_PAD_HEIGHT < self.paddle1_pos[1] < self.HEIGHT - self.HALF_PAD_HEIGHT:
            self.paddle1_pos[1] += self.paddle1_vel
        elif self.paddle1_pos[1] == self.HALF_PAD_HEIGHT and self.paddle1_vel > 0:
            self.paddle1_pos[1] += self.paddle1_vel
        elif self.paddle1_pos[1] == self.HEIGHT - self.HALF_PAD_HEIGHT and self.paddle1_vel < 0:
            self.paddle1_pos[1] += self.paddle1_vel

        if self.HALF_PAD_HEIGHT < self.paddle2_pos[1] < self.HEIGHT - self.HALF_PAD_HEIGHT:
            self.paddle2_pos[1] += self.paddle2_vel
        elif self.paddle2_pos[1] == self.HALF_PAD_HEIGHT and self.paddle2_vel > 0:
            self.paddle2_pos[1] += self.paddle2_vel
        elif self.paddle2_pos[1] == self.HEIGHT - self.HALF_PAD_HEIGHT and self.paddle2_vel < 0:
            self.paddle2_pos[1] += self.paddle2_vel

        # update ball
        self.ball_pos[0] += int(self.ball_vel[0])
        self.ball_pos[1] += int(self.ball_vel[1])

        # ball collision check on top and bottom walls
        if int(self.ball_pos[1]) <= self.BALL_RADIUS:
            self.ball_vel[1] = - self.ball_vel[1]
        if int(self.ball_pos[1]) >= self.HEIGHT + 1 - self.BALL_RADIUS:
            self.ball_vel[1] = -self.ball_vel[1]

        # ball collison check on gutters or paddles
        if int(self.ball_pos[0]) <= self.BALL_RADIUS + self.PAD_WIDTH and int(self.ball_pos[1]) in range(
                self.paddle1_pos[1] - self.HALF_PAD_HEIGHT,
                self.paddle1_pos[1] + self.HALF_PAD_HEIGHT,
                1):
            self.ball_vel[0] = -self.ball_vel[0]
            self.ball_vel[0] *= 1.1
            self.ball_vel[1] *= 1.1
        elif int(self.ball_pos[0]) <= self.BALL_RADIUS + self.PAD_WIDTH:
            self.r_score += 1
            self.ball_init(True)

        if int(self.ball_pos[0]) >= self.WIDTH + 1 - self.BALL_RADIUS - self.PAD_WIDTH and int(
                self.ball_pos[1]) in range(
            self.paddle2_pos[1] - self.HALF_PAD_HEIGHT, self.paddle2_pos[1] + self.HALF_PAD_HEIGHT, 1):
            self.ball_vel[0] = -self.ball_vel[0]
            self.ball_vel[0] *= 1.1
            self.ball_vel[1] *= 1.1
        elif int(self.ball_pos[0]) >= self.WIDTH + 1 - self.BALL_RADIUS - self.PAD_WIDTH:
            self.l_score += 1
            self.ball_init(False)

    def play(self, action):
        assert action in range(3), "Invalid action."
        if action == 0:
            self.paddle2_vel = -8
        elif action == 1:
            self.paddle2_vel = 8
        else:
            self.paddle2_vel = 0
        self.move_PC()
        self.updatePositions()

    def get_state(self):
        canvas = np.zeros((self.WIDTH, self.HEIGHT))

        canvas[
        self.ball_pos[0]: self.ball_pos[0] + self.BALL_RADIUS,
        self.ball_pos[1]: self.ball_pos[1] + self.BALL_RADIUS
        ] = 1

        canvas[
        self.paddle1_pos[0]: self.paddle1_pos[0] + self.PAD_WIDTH,
        self.paddle1_pos[1]: self.paddle1_pos[1] + self.PAD_HEIGHT
        ] = -.5

        canvas[
        self.paddle2_pos[0]: self.paddle2_pos[0] + self.PAD_WIDTH,
        self.paddle2_pos[1]: self.paddle2_pos[1] + self.PAD_HEIGHT
        ] = .5

        ca = canvas.transpose()
        #plt.imshow(ca)
        #plt.pause(0.01)
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
        return self.l_score == self.maxScore or self.r_score == self.maxScore

    def is_won(self):
        return self.l_score == self.maxScore
