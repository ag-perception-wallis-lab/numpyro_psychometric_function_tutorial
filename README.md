# A tutorial on psychometric functions using Numpyro

Thomas Wallis (thomas.wallis@tu-darmstadt.de).

Prepared initially for a 90-min tutorial at the "Workshop on Visual Representations", Schloss Weitenburg near Starzach, March 5th -- 8th 2024.

WARNING: this code is currently not well tested, and should be used at your own risk.

## Environment management

I'm using [Poetry](https://python-poetry.org/) for environment management. The environment is specified in the `pyproject.toml` file. After installing Poetry on your system, you should be able to initialise a virtual environment by typing `poetry install` in the project directory. (By default this creates a `.venv` directory in your project directory). If you have troubles with specific versions, try deleting the `poetry.lock` file and trying `poetry install` again.

## Literate programming

I'm using [Quarto](https://quarto.org) for the notebook-like thing. 
The `.qml` file can be run interactively in VS Code (much like a notebook), but one can then render out to a variety of formats by running `quarto render` from the command line (with the environment activated).

## Basic directory structure

- `data` contains... the data
- `docs` contains documents / slides 
- `results` can be used to save outputs (e.g. fitted model objects)
- `scripts` contains anything that is "run" as a script -- including notebook-like objects.
- `src` contains source code that can be imported into e.g. scripts.
- `tests` aspirationally contains tests for code in `src`.