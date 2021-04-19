import numpy as np # type: ignore

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


def neighbours_count(grid: np.ndarray, x: int, y: int) -> int:
    """Compute the number of alive neighbours.

    >>> g = np.ones((5,5), dtype=int)
    >>> g = zero_border(g)
    >>> neighbours_count(g, 2, 2)
    8

    >>> g = np.zeros((5,5), dtype=int)
    >>> neighbours_count(g, 2, 2)
    0

    >>> g = np.array([[0,0,0,0],
    ...               [0,1,1,0],
    ...               [0,0,1,0],
    ...               [0,0,0,0]], dtype=int)
    >>> neighbours_count(g, 2, 2)
    2

    """
    assert (grid[[0, -1], :] == 0).all() and (grid[:, [0, -1]] == 0).all()
    ris = 0
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if not (dx == 0 and dy == 0):
                ris = ris + grid[x+dx, y+dy]
    return ris

def neighbours_matrix(grid: np.ndarray) -> np.array:
    """Compute the matrix of neighbours.

    >>> g = np.array([[0,0,0,0,0,0],
    ...               [0,0,1,0,0,0],
    ...               [0,0,0,1,0,0],
    ...               [0,1,1,1,0,0],
    ...               [0,0,0,0,0,0],
    ...               [0,0,0,0,0,0]], dtype=int)
    >>> neighbours_matrix(g)
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
            ris[i, j] = neighbours_count(grid, i, j)
    return ris

def neighbours_matrix_v(grid: np.ndarray) -> np.array:
    """Compute the matrix of neighbours without using loops.

    >>> rng = np.random.default_rng(seed=42)
    >>> g = rng.integers(0, 2, size=(10,20))
    >>> g = zero_border(g)
    >>> (neighbours_matrix_v(g) == neighbours_matrix(g)).all()
    True

    """
    assert (grid[[0, -1], :] == 0).all() and (grid[:, [0, -1]] == 0).all()
    ris = np.zeros_like(grid)
    ris[1:-1, 1:-1] = grid[2:,1:-1] + grid[2:,2:] + grid[1:-1,2:] + \
                      grid[:-2,2:] + grid[:-2,1:-1] + grid[:-2,:-2] + \
                      grid[1:-1,:-2] + grid[2:,:-2]
    return ris

def new_generation(grid: np.ndarray) -> np.array:
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
    neighs = neighbours_matrix(grid)
    for i in range(1, grid.shape[0]-1):
        for j in range(1, grid.shape[1]-1):
            if grid[i, j] == 1:
                if neighs[i, j] < 2 or neighs[i, j] > 3:
                    ris[i, j] = 0
            else:
                if neighs[i, j] == 3:
                    ris[i, j] = 1
    return ris

def new_generation_v(grid: np.ndarray) -> np.array:
    """Compute new generation without using loops.

    >>> rng = np.random.default_rng(seed=42)
    >>> g = rng.integers(0, 2, size=(10,20))
    >>> g = zero_border(g)
    >>> (new_generation_v(g) == new_generation(g)).all()
    True
    """
    assert (grid[[0, -1], :] == 0).all() and (grid[:, [0, -1]] == 0).all()
    ris = grid.copy()
    neighs = neighbours_matrix_v(grid)

    # overpopulation
    ris[(grid == 1) & (neighs > 3)] = 0
    # underpopulation
    ris[(grid == 1) & (neighs < 2)] = 0
    # reproduction
    ris[(grid == 0) & (neighs == 3)] = 1
    return ris


if __name__ == '__main__':
    import matplotlib.pyplot as plt # type: ignore
    from matplotlib.animation import FuncAnimation # type: ignore

    SIZE = 1000

    rng = np.random.default_rng(seed=42)
    G = rng.integers(0, 2, size=(SIZE,2*SIZE))
    G = zero_border(G)


    fig, ax = plt.subplots()
    im = ax.imshow(G, cmap=plt.cm.gray_r, vmin=0, vmax=1)
    plt.xticks([])
    plt.yticks([])

    def update(frame: int) -> None:
        global G, im
        if SIZE >= 1000:
            G = new_generation_v(G)
        else:
            G = new_generation(G)
        im.set_data(G)

    animation = FuncAnimation(fig, update, frames=1000)
    if SIZE >= 1000:
        animation.save('solved.mp4')
    plt.show()
