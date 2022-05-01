# The Game of Life

The code for this homework is already written, but it is slow. 

Your goal is to write a new version of the key functions *without using loops*,
but using instead numpy vectorization (`neighbour_matrix` and `new_generation`,
look at the TODO comments): then try it with `SIZE = 1000`.


This project is designed to work within a Jupyter notebook, but to keep
everything clean it synchronizes it with a plain `.py` file. In order this to
work it needs [Jupytext](https://jupytext.readthedocs.io/en/latest/), therefore
remember to setup correctly the virtual environment.

1. set up a virtual environment as usual (for example with pipenv)

2. install all the dependencies with `pip install -r requirements.txt`

3. launch the notebook within the virtual environment: `jupyter notebook`

You can see the animation also before writing the fast versions, but be sure to
use a small `SIZE`.

You can check locally the type hints with `mypy exercise.py` and the doctests
with `python -m doctest exercise.py`.




