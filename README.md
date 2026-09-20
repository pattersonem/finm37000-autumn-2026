## FINM 37000 - Futures and Related Derivates

This repo contains a Python package and course material for FINM 37000 offered in
Autumn 2026 at the University of Chicago.

### The finm37000 package

Use of a `venv` is **strongly** encouraged. Different IDEs and systems will manage these differently.
Refer to documentation for your environment. In addition to the Python dependencies,
some notebooks in the course use [graphviz](https://graphviz.org/), which has a Python wrapper but requires
the underlying binary to exist on your system. It is not essential for most
of the material here.

**DO NOT SKIP SETTING UP YOUR `venv`** The following directions assume you are working in a virtual
environment, but it is incumbent on you to make sure you are using yours because the instructions
make no attempt to cover the system and environment differences that can arise.

You can install the package directly from `github`:

```
# For public repo
python -m pip install git+https://github.com/pattersonem/finm37000-autumn-2026.git
```

Install for development:
```
git clone git@github.com:pattersonem/finm37000-autumn-2026.git
cd finm37000-autumn-2026
uv pip install . --group demo
# pip >= 25.2
# python -m pip install --group demo -e .
# Earlier pip
# python -m pip install -e ".[demo]"
```

### Using Jupyter notebook in a virtual environment reminder

Your virtual environment needs to be registered with `ipykernel`. For example, from the command line you
can register it like this

    python -m ipykernel install --user --name=finm37000 --display-name="FINM 37000"

### Test

```
python -m pytest
```

### Lint

```
python -m ruff check
```

### Format

```
python -m ruff format
```

### Type Check

```
python -m mypy .
```

Package source is available in the `src` directory with specifications in `pyproject.toml`.

There are is a `Makefile`, which can batch some of that, but it mixes some idioms (pip and uv)
and separates installation from building, so you need to run with an active `venv` for
it to work and should not expect it to build your `venv` automatically (but `make dev` can help).

