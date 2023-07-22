#!/bin/sh

# Code formatting with isort and black
isort $(git ls-files '*.py')
black $(git ls-files '*.py')

git add $(git ls-files)

# Check Git diff-index
git diff-index --quiet HEAD --

if [ $? -eq 0 ]; then
	echo "No changes found, nothing to see/do here."
else
	echo "Changes found. Preparing commit."
	git config --local user.email "github-actions[bot]@users.noreply.github.com"
	git config --local user.name "github-actions[bot]"
	git commit -m "Automated formatting changes"
fi

git log | head
exit 0