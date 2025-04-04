from Mesh import Cell, Scent
import pygame
import numpy as np
import math
import random
from typing import List, Tuple


class Game:

    col_alive = [50, 20, 215]
    background = [0, 0, 0]
    col_grid = [30, 30, 60]

    cell_pattern = np.ones((5, 5))

    food_D = 0.5
    food_decay = 0
    food_dx = 0.1
    food_dt = 0.9 * (food_dx**2 / (4*food_D)) if food_D > 0 else 1e-5

    mold_D = 0.001
    mold_decay = 0.01
    mold_dx = 0.1
    mold_dt = 0.9 * (mold_dx**2 / (4*mold_D)) if mold_D > 0 else 1e-5

    def cell_beyond_grid(dimx: int, dimy: int, pos: Tuple[int, int]) -> bool:
        return (pos[0] < 0) | (pos[0]+Game.cell_pattern.shape[0] >= dimx) | \
            (pos[1] < 0) | (pos[1]+Game.cell_pattern.shape[1] >= dimy)

    def cell_over_maze(maze: np.ndarray, pos: Tuple[int, int]) -> bool:
        if Game.cell_beyond_grid(maze.shape[0], maze.shape[1], pos):
            return True
        grid = np.zeros(maze.shape)
        grid[pos[0]:pos[0]+Game.cell_pattern.shape[0],
             pos[1]:pos[1]+Game.cell_pattern.shape[1]] = Game.cell_pattern

        return (grid * maze > 0).any()

    def generate_maze(dimx: int, dimy: int) -> np.ndarray:
        width = 9
        maze = np.zeros((dimx, dimy))

        # draw walls every 10 pixels
        for i in range(0, dimx, width+1):
            maze[:, i] = 1
        maze[:, -1] = 1
        for j in range(0, dimy, width+1):
            maze[j, :] = 1
        maze[-1, :] = 1

        # use backtrack algorithm to generate maze
        stack = [(0, 0)]
        visited = np.zeros((dimx//(width+1), dimy//(width+1)), dtype=bool)
        visited[0, 0] = True
        while len(stack) > 0:
            x, y = stack[-1]
            neighbors = []
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x+dx, y+dy
                if 0 <= nx < dimx//(width+1) and 0 <= ny < dimy//(width+1) and not visited[nx, ny]:
                    neighbors.append((nx, ny))
            if len(neighbors) > 0:
                nx, ny = random.choice(neighbors)
                visited[nx, ny] = True
                # remove the wall between current cell and chosen neighbor
                if nx - x == -1:
                    maze[x*(width+1):x*(width+1)+1, y *
                         (width+1)+1:(y+1)*(width+1)] = 0
                elif nx - x == 1:
                    maze[(x+1)*(width+1):(x+1)*(width+1)+1,
                         y*(width+1)+1:(y+1)*(width+1)] = 0
                elif ny - y == -1:
                    maze[x*(width+1)+1:(x+1)*(width+1),
                         y*(width+1):y*(width+1)+1] = 0
                elif ny - y == 1:
                    maze[x*(width+1)+1:(x+1)*(width+1), (y+1)
                         * (width+1):(y+1)*(width+1)+1] = 0
                stack.append((nx, ny))
            else:
                stack.pop()

        return maze

    def __init__(self, num_cells: int, food_pos: List[Tuple[int, int]],
                 dimx: int = 100, dimy: int = 100, cellsize: float = 8):

        self.dimx, self.dimy = dimx, dimy
        self.cellsize = cellsize
        self.maze = Game.generate_maze(self.dimx, self.dimy)
        # self.maze = np.zeros((self.dimx, self.dimy))

        self.food_scent = Scent(self.dimx, self.dimy,
                                Game.food_D, Game.food_dx, Game.food_dt,
                                Game.food_decay, self.maze, self.avail_to_fill_scent)
        for (x, y) in food_pos:
            self.food_scent.drop_particle(x, y, 500)

        # define the cells
        self.cells = []
        all_init_pos = []
        for _ in range(num_cells):
            init_pos = (np.random.randint(0, dimx), np.random.randint(0, dimy))
            while (init_pos in all_init_pos) | \
                    (init_pos in food_pos) |\
                    Game.cell_over_maze(self.maze, init_pos) | \
                    Game.cell_beyond_grid(dimx, dimy, init_pos):

                init_pos = (np.random.randint(0, dimx),
                            np.random.randint(0, dimy))

            self.cells.append(Cell(self.dimx, self.dimy, Game.cell_pattern,
                                   init_pos, self.food_scent.get_scent,
                                   self.maze, self.avail_to_fill_cell,
                                   self.avail_to_fill_scent, Game.mold_D,
                                   Game.mold_dx, Game.mold_dt, Game.mold_decay))
            all_init_pos.append(init_pos)

        # initalize pygame
        pygame.init()
        self.surface = pygame.display.set_mode(
            (self.dimx*self.cellsize, self.dimy*self.cellsize))
        pygame.display.set_caption("Cellular Automata")

    def avail_to_fill_cell(self, x: int, y: int) -> bool:
        """
        Determines if the pixel could be filled by a cell
        """
        # out of boundary
        if x < 0 | x >= self.dimx | y < 0 | y >= self.dimy:
            return False

        # maze wall
        if self.maze[x, y] == 1:
            return False

        # another cell is occupying the pixel
        for cell in self.cells:
            if cell.data[x, y] > 0:
                return False

        return True

    def avail_to_fill_scent(self, x: int, y: int) -> bool:
        """
        Determines if the pixel could be filled by a scent
        """
        # out of boundary
        if x < 0 | x >= self.dimx | y < 0 | y >= self.dimy:
            return False

        # maze wall
        if self.maze[x, y] == 1:
            return False

        return True

    def update(self):

        # move the cells
        for cell in self.cells:
            cell.update()

        # update the food scent
        self.food_scent.update()

        # draw the grid on the surface
        self.surface.fill(Game.background)
        for i in range(self.dimx):
            pygame.draw.line(self.surface, Game.col_grid,
                             (i, 0), (i, self.dimy))
        for j in range(self.dimy):
            pygame.draw.line(self.surface, Game.col_grid,
                             (0, j), (self.dimx, j))

        # superpose all the cells into a single grid, identify the cell by the number
        cells_grid = np.zeros((self.dimx, self.dimy))
        mold_hist = np.zeros((self.dimx, self.dimy))
        for i, cell in enumerate(self.cells):
            cells_grid += (i+1) * \
                cell.data if not cell.is_exploring else -(i+1)*cell.data
            mold_hist += (i+1) * cell.mold_history.data

        # draw the cells on the surface
        for i in range(self.dimx):
            for j in range(self.dimy):
                if self.maze[i, j] == 1:
                    pygame.draw.rect(self.surface, [255, 255, 255],
                                     (j*self.cellsize, i*self.cellsize,
                                      self.cellsize, self.cellsize))
                    continue

                color = np.array([0, 0, 0, 0])
                if cells_grid[i, j] > 0:
                    color = [0, min(50*cells_grid[i, j], 255), 255, 1]
                elif cells_grid[i, j] < 0:
                    color = [min(50*-cells_grid[i, j], 255), 0, 255, 1]
                else:
                    if self.food_scent.data[i, j] > 0:
                        color += np.array([
                            0, int(math.tanh(self.food_scent.data[i, j]) * 200), 0, 0])

                    if mold_hist[i, j] > 0:
                        c = int(math.tanh(mold_hist[i, j]) * 50)
                        color += np.array([c, 0, c, 0])
                    color[3] = 0.6
                pygame.draw.rect(self.surface, color,
                                 (j*self.cellsize, i*self.cellsize,
                                  self.cellsize, self.cellsize))

        pygame.display.update()
