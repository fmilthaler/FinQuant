#!/bin/bash

# Code formatting with isort and black
echo "Code formatting with isort and black:"
isort $(git ls-files '*.py')
black $(git ls-files '*.py')
