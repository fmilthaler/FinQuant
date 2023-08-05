# Contributing to FinQuant
First of all, thank you for your interest in FinQuant and wanting to contribute. Your help is much appreciated.

Here are some guidelines for contributing to FinQuant.

## Reporting Bugs and Issues
Before creating a new issue/bug report, check the list of existing [issues](https://github.com/fmilthaler/FinQuant/issues). When raising a new [issue](https://github.com/fmilthaler/FinQuant/issues), include as many details as possible and follow the following guidelines:
- Use a clear and descriptive title for the issue to identify the problem.
- Provide a minimal example, a series of steps to reproduce the problem.
- Describe the behaviour you observed after following the steps.
- Explain which behaviour you expected to see instead and why.
- State versions of FinQuant/Python and Operating system you are using.
- Provide error messages if applicable.

## Coding Guidelines
So you wish to fix bugs yourself, or contribute by adding a new feature. Awesome! Here are a few guidelines I would like you to respect.

### Create a fork
First off, you should create a fork. Within your fork, create a new branch. Depending on what you want to do,
choose one of the following prefixes for your branch name:
- `bugfix/` followed by something like `<name of your fix>`: to be used for bug fixes
- `feature/` followed by something like `<name of new feature>`: to be used for adding a new feature

If you simply want to refactor the code base, or do other types of chores, use one of the following branch name prefixes:
- `refactor/` followed by something like `<what you are refactoring>`
- `chore/` followed by something like `<other type of contribution>`

**NOTE**: It is _mandatory_ to use one of the above prefixes for your branch name. FinQuant uses GitHub workflows
to automatically bump the version number when a PR is merged into `master` (or `develop`).
The new version number depends on the source branch name of the merged PR.

Example:
. If you are working on a bugfix to fix a print statement of the portfolio properties,
your branch name should be something like bugfix/print-statement-portfolio-properties.
For the automated versioning to work, the branch name is required to start with `bugfix/` or one of the other
above mentioned patterns.

### Custom data types
[FinQuant defines a number of custom data types](https://finquant.readthedocs.io/en/latest/developers.html#data-types)
in the module `finquant.data_types`.

These data types are useful as lots of functions/methods in FinQuant allow arguments to be of different data types.
For example:
- `data` is often accepted as either a `pandas.Series` or `pandas.DataFrame`, or
- `risk_free_rate` could be a Python `float` or a `numpy.float64` among others.

To accommodate and simplify this, custom data types are defined in the module `finquant.data_types`.
Please familiarize yourself with those and add more if your code requires them.

### Data type validation
[FinQuant provides a module/function for type validation](https://finquant.readthedocs.io/en/latest/developers.html#type-validation),
which is used throughout the code base for type validation purposes. Said function simplifies checking an argument
against its expected type and reduces the amount of copy-pasted `if` and `raise` statements.
You can check out the source code in `finquant.type_utilities`.

### Commit your changes
Make your changes to the code, and write sensible commit messages.

### Tests
In the root directory of your version of FinQuant, run `make test` and make sure all tests are passing.
If applicable, add new tests in the `./tests/` directory. Tests should be written with `pytest`.

Some few tests require you to have a [Quandl API key](https://docs.quandl.com/docs#section-authentication).
If you do not have one locally, you can ignore the tests that are failing due to a missing Quandl API key.
Once you open a PR, all tests are run by GitHub Actions with a pre-configured key.

### Documentation
If applicable, please add docstrings to new functions/classes/modules.
Follow example of existing docstrings. FinQuant uses `sphinx` to generate Documentation
for [ReadTheDocs](https://finquant.readthedocs.io) automatically from docstrings.

### Style
Fortunately for you, you can ignore code formatting and fully focus on your contribution.
FinQuant uses a GitHub workflow that is automatically triggered and runs [Black](https://github.com/psf/black) and
[isort](https://pycqa.github.io/isort/) to format the code base for you.

### Create a Pull Request
Create a new [Pull Request](https://github.com/fmilthaler/FinQuant/pulls).
Describe what your changes are in the Pull Request.
If your contribution fixes a bug, or adds a features listed under
[issues](https://github.com/fmilthaler/FinQuant/issues) as "#12", please add "fixes #12" or "closes #12".
