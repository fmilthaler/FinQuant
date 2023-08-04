#!/bin/sh

echo "Running Pylint - finquant (ignoring TODOs)"
python -m pylint --disable=fixme --output-format=parseable *.py finquant | tee pylint.log
#echo "Running Pylint - tests (ignoring TODOs and access to protected attributes)"
#python -m pylint --disable=fixme,protected-access --output-format=parseable tests | tee -a pylint.log

echo ""
echo "Running Mypy"
python -m mypy *.py finquant | tee mypy.log

#echo ""
#echo "Running Black (check mode only)"
#python -m black --check *.py finquant tests

#echo ""
#echo "Running isort (check mode only)"
#python -m isort --check *.py finquant tests