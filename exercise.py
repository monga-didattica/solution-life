import numpy as np


def zero_border(grid: np.ndarray) -> np.ndarray:
    """Change the grid such that the border is all zeroes.

    >>> g = np.ones((10, 10), dtype=int)
    >>> g = zero_border(g)
    >>> (g[[0, -1], :] == 0).all() and (g[:, [0, -1]] == 0).all()
    True

    """
    ris = grid.copy()
    ris[[0, -1], :] = 0
    ris[:, [0, -1]] = 0
    return ris


def neighbour_count(grid: np.ndarray, x: int, y: int) -> int:
    """Compute the number of alive neighbours.

    >>> g = np.ones((5,5), dtype=int)
    >>> g = zero_border(g)
    >>> neighbour_count(g, 2, 2)
    8

    >>> g = np.zeros((5,5), dtype=int)
    >>> neighbour_count(g, 2, 2)
    0

    >>> g = np.array([[0,0,0,0],
    ...               [0,1,1,0],
    ...               [0,0,1,0],
    ...               [0,0,0,0]], dtype=int)
    >>> neighbour_count(g, 2, 2)
    2

    """
    assert (grid[[0, -1], :] == 0).all() and (grid[:, [0, -1]] == 0).all()
    ris = 0
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if not (dx == 0 and dy == 0):
                ris = ris + grid[x+dx, y+dy]
    return ris


def neighbour_matrix(grid: np.ndarray) -> np.ndarray:
    """Compute the matrix of neighbours.

    >>> g = np.array([[0,0,0,0,0,0],
    ...               [0,0,1,0,0,0],
    ...               [0,0,0,1,0,0],
    ...               [0,1,1,1,0,0],
    ...               [0,0,0,0,0,0],
    ...               [0,0,0,0,0,0]], dtype=int)
    >>> neighbour_matrix(g)
    array([[0, 0, 0, 0, 0, 0],
           [0, 1, 1, 2, 1, 0],
           [0, 3, 5, 3, 2, 0],
           [0, 1, 3, 2, 2, 0],
           [0, 2, 3, 2, 1, 0],
           [0, 0, 0, 0, 0, 0]])

    """
    assert (grid[[0, -1], :] == 0).all() and (grid[:, [0, -1]] == 0).all()
    ris = np.zeros_like(grid)
    for i in range(1, grid.shape[0]-1):
        for j in range(1, grid.shape[1]-1):
            ris[i, j] = neighbour_count(grid, i, j)
    return ris


def neighbour_matrix_v(grid: np.ndarray) -> np.ndarray:
    """Compute the matrix of neighbours without using loops.

    >>> rng = np.random.default_rng(seed=42)
    >>> g = rng.integers(0, 2, size=(10,20))
    >>> g = zero_border(g)
    >>> (neighbour_matrix_v(g) == neighbour_matrix(g)).all()
    True

    """
    assert (grid[[0, -1], :] == 0).all() and (grid[:, [0, -1]] == 0).all()
    ris = np.zeros_like(grid)
    ris[1:-1, 1:-1] = (grid[2:, 1:-1] + grid[2:, 2:] + grid[1:-1, 2:] +
                       grid[:-2, 2:] + grid[:-2, 1:-1] + grid[:-2, :-2] +
                       grid[1:-1, :-2] + grid[2:, :-2])
    return ris


def new_generation(grid: np.ndarray) -> np.ndarray:
    """Compute new generation.

    >>> g = np.array([[0,0,0,0,0,0],
    ...               [0,0,1,0,0,0],
    ...               [0,0,0,1,0,0],
    ...               [0,1,1,1,0,0],
    ...               [0,0,0,0,0,0],
    ...               [0,0,0,0,0,0]], dtype=int)
    >>> new_generation(g)
    array([[0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0],
           [0, 1, 0, 1, 0, 0],
           [0, 0, 1, 1, 0, 0],
           [0, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0]])

    """
    assert (grid[[0, -1], :] == 0).all() and (grid[:, [0, -1]] == 0).all()
    ris = grid.copy()
    neighs = neighbour_matrix(grid)
    for i in range(1, grid.shape[0]-1):
        for j in range(1, grid.shape[1]-1):
            if grid[i, j] == 1:
                if neighs[i, j] < 2 or neighs[i, j] > 3:
                    ris[i, j] = 0
            else:
                if neighs[i, j] == 3:
                    ris[i, j] = 1
    return ris


def new_generation_v(grid: np.ndarray) -> np.ndarray:
    """Compute new generation without using loops.

    >>> rng = np.random.default_rng(seed=42)
    >>> g = rng.integers(0, 2, size=(10,20))
    >>> g = zero_border(g)
    >>> (new_generation_v(g) == new_generation(g)).all()
    True
    """
    assert (grid[[0, -1], :] == 0).all() and (grid[:, [0, -1]] == 0).all()
    ris = grid.copy()
    neighs = neighbour_matrix_v(grid)

    # overpopulation
    ris[(grid == 1) & (neighs > 3)] = 0
    # underpopulation
    ris[(grid == 1) & (neighs < 2)] = 0
    # reproduction
    ris[(grid == 0) & (neighs == 3)] = 1
    return ris


import matplotlib.pyplot as plt  # type: ignore

SIZE = 50
rng = np.random.default_rng(seed=42)
G = rng.integers(0, 2, size=(SIZE,2*SIZE))
G = zero_border(G)

fig, ax = plt.subplots()
ax.imshow(G, cmap=plt.cm.gray_r, vmin=0, vmax=1)
ax.set_xticks([])
_ = ax.set_yticks([])

fig, ax = plt.subplots()
im = ax.imshow(new_generation(G), cmap=plt.cm.gray_r, vmin=0, vmax=1)
ax.set_xticks([])
_ = ax.set_yticks([])

from matplotlib.animation import FuncAnimation  # type: ignore


class Life:
    def __init__(self, grid: np.ndarray):
        self.grid = grid
        self.fig, ax = plt.subplots()
        self.im = ax.imshow(self.grid, cmap=plt.cm.gray_r, vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])

    def update(self, frame: int):
        self.grid = new_generation_v(self.grid)
        self.im.set_data(self.grid)

    def animate(self, frames: int):
        self.animation = FuncAnimation(self.fig, self.update, frames=frames)


life = Life(G)
life.animate(100)

from IPython.display import HTML  # type: ignore
HTML(life.animation.to_jshtml())
