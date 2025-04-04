import numpy as np
import random
import math
from scipy.ndimage import label
from scipy.signal import convolve2d
from typing import Tuple, List, Callable


class Mesh:
    """
    A mesh is a 2D array
    """

    def __init__(self, dimx: int, dimy: int):
        self.dimx = dimx
        self.dimy = dimy
        self.data = np.zeros((dimx, dimy))


class Cell(Mesh):

    # convolution kernels
    kernel_star = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    connected_structure = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

    def __init__(self, dimx: int, dimy: int, pattern: np.ndarray,
                 init_pos: Tuple[int, int], sense_fn: Callable[[int, int], float],
                 maze: np.ndarray):
        super().__init__(dimx, dimy)
        self.data[init_pos[0]:init_pos[0]+pattern.shape[0],
                  init_pos[1]:init_pos[1]+pattern.shape[1]] = pattern
        self.sense_fn = sense_fn
        self.maze = maze

    def update(self) -> None:

        # identify the boundarys and inner shells
        bound_conv = convolve2d(self.data, Cell.kernel_star, mode='same',
                                boundary='fill')
        outer_idxs = np.argwhere(bound_conv > 0).tolist()
        inner_idxs = np.argwhere(bound_conv < 0).tolist()

        # sort based on sensing
        sorted_outer_idxs = sorted(
            outer_idxs, key=lambda xy: self.sense_fn(xy[0], xy[1]), reverse=True)
        sorted_inner_idxs = sorted(inner_idxs,
                                   key=lambda xy: self.sense_fn(xy[0], xy[1]))

        # take some percent of the outer and inner points
        to_fill_coor = random.choices(sorted_outer_idxs, weights=[
            0.8**i for i in range(len(sorted_outer_idxs))])[0]
        to_remove_coor = random.choices(sorted_inner_idxs, weights=[
            0.8**i for i in range(len(sorted_inner_idxs))])[0]

        # check if the cell is in the maze
        while self.maze[to_fill_coor[0], to_fill_coor[1]] == 1:
            to_fill_coor = random.choices(sorted_outer_idxs, weights=[
                0.8**i for i in range(len(sorted_outer_idxs))])[0]

        # update a copy of cells
        data_copy = self.data.copy()
        data_copy[to_fill_coor[0], to_fill_coor[1]] = 1
        data_copy[to_remove_coor[0], to_remove_coor[1]] = 0

        # check that all points are connected to the rest of the region
        # i.e at least a one in the region
        _, num_features = label(data_copy, structure=Cell.connected_structure)

        # commit if there are no points that are alone
        if num_features == 1:
            self.data = data_copy


class Scent(Mesh):

    dx = 0.1
    diffusion_rate = 0.1
    dt = 0.9 * (dx**2 / (4*diffusion_rate))
    decay_rate = 0

    def __init__(self, dimx: int, dimy: int, food_pos: List[Tuple[int, int]],
                 maze: np.ndarray):
        super().__init__(dimx, dimy)
        for pos in food_pos:
            self.data[pos[0], pos[1]] = 500

        self.food_pos = food_pos
        self.maze = maze

    def update(self) -> None:
        # diffuse the food scent using Fick's law
        for i in range(self.dimx):
            for j in range(self.dimy):
                if self.maze[i, j] == 1:
                    continue

                x_left = max(i-1, 0)
                x_right = min(i+1, self.dimx-1)
                y_top = max(j-1, 0)
                y_bottom = min(j+1, self.dimy-1)

                laplacian = 0
                # calculate the laplacian using the finite difference method
                if self.maze[x_left, j] == 0:
                    laplacian += self.data[x_left, j] - self.data[i, j]
                if self.maze[x_right, j] == 0:
                    laplacian += self.data[x_right, j] - self.data[i, j]
                if self.maze[i, y_top] == 0:
                    laplacian += self.data[i, y_top] - self.data[i, j]
                if self.maze[i, y_bottom] == 0:
                    laplacian += self.data[i, y_bottom] - self.data[i, j]
                laplacian /= Scent.dx**2
                
                # diffusion based on Fick's law
                self.data[i, j] += self.diffusion_rate * laplacian * self.dt
                self.data[i, j] -= self.decay_rate * self.dt * self.data[i, j]

                self.data[i, j] = max(self.data[i, j], 0)

    def get_scent(self, x: int, y: int) -> float:

        # return the scent at the given position
        return self.data[x, y]
