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
                 maze: np.ndarray, fill_avail_cell: Callable[[int, int], bool],
                 fill_avail_scent: Callable[[int, int], bool],
                 mold_D: float, mold_dx: float, mold_dt: float, mold_decay_rate: float):
        super().__init__(dimx, dimy)
        self.data[init_pos[0]:init_pos[0]+pattern.shape[0],
                  init_pos[1]:init_pos[1]+pattern.shape[1]] = pattern
        self.x_min, self.x_max, self.y_min, self.y_max = self.compute_minmax_coor()
        self.maze = maze
        self.fill_avail = fill_avail_cell
        self.sense_fn = sense_fn

        self.mold_history = Scent(dimx, dimy, mold_D, mold_dx, mold_dt,
                                  mold_decay_rate, maze, fill_avail_scent)
        self.explore_fn = lambda x, y: -1 * self.mold_history.get_scent(x, y)
        self.threshold = 0.01
        self.is_exploring = False

    def compute_minmax_coor(self) -> Tuple[int, int, int, int]:
        """
        Compute the boundary of the cell in the maze
        """
        # get the indices of the cell in the maze
        cell_indices = np.argwhere(self.data > 0)
        x_min, x_max = cell_indices[:, 0].min(), cell_indices[:, 0].max()
        y_min, y_max = cell_indices[:, 1].min(), cell_indices[:, 1].max()

        return x_min, x_max, y_min, y_max

    def compute_boundary(self) -> Tuple[List, List]:
        """
        Identify the boundarys and inner shells
        convolve only the area around the cell to save compute
        """
        reduced_data = self.data.copy()
        reduced_data = reduced_data[max(self.x_min-1, 0):min(self.x_max+2, self.dimx),
                                    max(self.y_min-1, 0):min(self.y_max+2, self.dimy)]
        bound_conv = convolve2d(reduced_data, Cell.kernel_star, mode='same',
                                boundary='fill')
        offset_x = max(self.x_min-1, 0)
        offset_y = max(self.y_min-1, 0)
        outer_idxs = np.argwhere(bound_conv > 0).tolist()
        outer_idxs = [(int(i[0]+offset_x), int(i[1]+offset_y))
                      for i in outer_idxs]
        inner_idxs = np.argwhere(bound_conv < 0).tolist()
        inner_idxs = [(int(i[0]+offset_x), int(i[1]+offset_y))
                      for i in inner_idxs]

        return outer_idxs, inner_idxs

    def explore_fn(self, x: int, y: int) -> float:
        return x*self.explore_dir[0] + y*self.explore_dir[1]

    def pick_removal(self, inner_idxs: List[Tuple[int, int]]) -> Tuple[int, int]:
        """
        Pick the inner pixel to remove from the cell
        """
        fn = self.explore_fn if self.is_exploring else self.sense_fn
        sorted_inner_idxs = sorted(inner_idxs, key=lambda xy: fn(xy[0], xy[1]))

        to_remove_coor = random.choices(sorted_inner_idxs, weights=[
            0.8**i for i in range(len(sorted_inner_idxs))])[0]

        return to_remove_coor

    def pick_fill(self, outer_idxs: List[Tuple[int, int]]) -> Tuple[int, int]:
        """
        Pick the outer pixel to fill the cell
        """
        # remove outer indices that are walled by the maze
        outer_idxs = [idx for idx in outer_idxs
                      if self.fill_avail(idx[0], idx[1])]

        fn = self.explore_fn if self.is_exploring else self.sense_fn
        sorted_outer_idxs = sorted(outer_idxs, key=lambda xy: fn(xy[0], xy[1]),
                                   reverse=True)

        to_fill_coor = random.choices(sorted_outer_idxs, weights=[
            0.8**i for i in range(len(sorted_outer_idxs))])[0]

        # check if the cell can be filled in the maze
        while not self.fill_avail(to_fill_coor[0], to_fill_coor[1]):
            to_fill_coor = random.choices(sorted_outer_idxs, weights=[
                0.8**i for i in range(len(sorted_outer_idxs))])[0]

        return to_fill_coor

    def sense_exists(self, outer_idxs: List) -> bool:
        """
        Whether there is a point on the outer boundary that has a scent
        """
        for x, y in outer_idxs:
            if self.sense_fn(x, y) > self.threshold:
                return True

        return False

    def update(self) -> None:

        # get the boundary and inner points
        outer_idxs, inner_idxs = self.compute_boundary()

        if self.sense_exists(outer_idxs):
            self.is_exploring = False
        else:
            self.is_exploring = True

        # pick a removeal and a fill point
        to_remove_coor = self.pick_removal(inner_idxs)
        to_fill_coor = self.pick_fill(outer_idxs)

        # update a copy of cells
        data_copy = self.data.copy()
        data_copy[to_fill_coor[0], to_fill_coor[1]] = 1
        data_copy[to_remove_coor[0], to_remove_coor[1]] = 0

        # check that all points are connected, there exists a sequence of jumps
        # from a one to another one between any two points
        _, num_features = label(data_copy, structure=Cell.connected_structure)

        # commit if there are no points that are alone
        if num_features == 1:
            self.data = data_copy
            self.x_min, self.x_max, self.y_min, self.y_max = self.compute_minmax_coor()

            # every point that you are active add to mold history
            filled_idxs = np.argwhere(self.data > 0).tolist()
            for x, y in filled_idxs:
                self.mold_history.drop_particle(x, y, 1)
            self.mold_history.update()

        # print(self.mold_history.data)


class Scent(Mesh):

    def __init__(self, dimx: int, dimy: int,
                 diffusion_rate: float, dx: float, dt: float, decay_rate: float,
                 maze: np.ndarray, fill_avail: Callable[[int, int], bool]):
        super().__init__(dimx, dimy)
        self.maze = maze
        self.fill_avail = fill_avail

        self.diffusion_rate = diffusion_rate
        self.dx = dx
        self.dt = dt
        self.decay_rate = decay_rate

    def compute_laplacian(self, x: int, y: int) -> float:
        """
        Compute the second derivative wrt space using finite difference method
        """

        x_left = max(x-1, 0)
        x_right = min(x+1, self.dimx-1)
        y_top = max(y-1, 0)
        y_bottom = min(y+1, self.dimy-1)

        # calculate the laplacian using the finite difference method
        laplacian = 0
        if self.maze[x_left, y] == 0:
            laplacian += self.data[x_left, y] - self.data[x, y]
        if self.maze[x_right, y] == 0:
            laplacian += self.data[x_right, y] - self.data[x, y]
        if self.maze[x, y_top] == 0:
            laplacian += self.data[x, y_top] - self.data[x, y]
        if self.maze[x, y_bottom] == 0:
            laplacian += self.data[x, y_bottom] - self.data[x, y]
        laplacian /= self.dx**2

        return laplacian

    def drop_particle(self, x: int, y: int, amount: float) -> None:
        """
        Drop a particle at the given position
        """
        if self.fill_avail(x, y):
            self.data[x, y] += amount

    def update(self) -> None:
        # diffuse the food scent using Fick's law
        for x in range(self.dimx):
            for y in range(self.dimy):
                if not self.fill_avail(x, y):
                    continue

                laplacian = self.compute_laplacian(x, y)
                self.data[x, y] += self.diffusion_rate * laplacian * self.dt
                self.data[x, y] -= self.decay_rate * self.dt * self.data[x, y]
                self.data[x, y] = max(self.data[x, y], 0)

    def get_scent(self, x: int, y: int) -> float:
        return self.data[x, y]
