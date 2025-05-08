import nox

PYTHON_VERSION = "3.11"
locations = "methods"

@nox.session(python=PYTHON_VERSION)
def tests(session):
    """Run pytest tests."""
    session.run("poetry", "install", external=True)
    session.run("poetry", "run", "pytest", external=True)


@nox.session(python=PYTHON_VERSION)
def linting(session):
    """Run ruff for code style."""
    session.run("poetry", "install", external=True)
    session.run("poetry", "run", "ruff", "check", locations, external=True)
    

@nox.session(python=PYTHON_VERSION)
def documentation(session):
    """Build documentation using Sphinx."""
    session.run("poetry", "install", external=True)
    session.run("poetry", "run", "sphinx-build", "-M", "html", "docs/source/", "docs/build/", external=True)


@nox.session(python=PYTHON_VERSION)
def typechecks(session):
    """Run static type checking with mypy."""
    session.run("poetry", "install", external=True)
    session.run("poetry", "run", "mypy", locations, external=True)