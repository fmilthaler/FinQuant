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
First off, you should create a fork. Within your fork, create a new branch. Depending on what you want to do, choose one of the following prefixes for your branch:
- bugfix/<name of your fix>: to be used for bug fixes
- feature/<name of new feature>: to be used for adding a new feature

### Commit your changes
Make your changes to the code, and write sensible commit messages.

### Tests
In the root directory of your version of FinQuant, run `make test` and make sure all tests are passing.
If applicable, add new tests in the `./tests/` directory. Tests should be written with `pytest`.

### Documentation
If applicable, please add docstrings to new functions/classes/modules. Follow example of existing docstrings. FinQuant uses `sphinx` to generate Documentation for [ReadTheDocs](https://finquant.readthedocs.io) automatically from docstrings.

### Style
To keep everything consistent, please use [Black](https://github.com/psf/black) with default settings.

### Create a Pull Request
Create a new [Pull Request](https://github.com/fmilthaler/FinQuant/pulls). Describe what your changes are in the Pull Request. If your contribution fixes a bug, or adds a features listed under [issues](https://github.com/fmilthaler/FinQuant/issues) as "#12", please add "fixes #12" or "closes #12". 

If you do not have a [Quandl API key](https://docs.quandl.com/docs#section-authentication) set on your Travis account, some of the tests are most likely going to fail. There are two ways forward:
1. you get a [Quandl API key](https://docs.quandl.com/docs#section-authentication), and set it as an environment variable under your Travis account settings.
2. Or you set the target of your Pull Request to either `feature/new-feature` or `bugfix/new-bugfix`, if your contribution is a new feature of bugfix respectively. That way your changes can be merged into a branch of FinQuant, and all the tests can be ran with my [Quandl API key](https://docs.quandl.com/docs#section-authentication).
