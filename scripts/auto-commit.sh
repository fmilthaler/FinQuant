#!/bin/bash

echo "Running auto commit"

COMMITMSG=$1

if [ -z "$COMMITMSG" ]; then
	COMMITMSG="Automated formatting changes"
fi

# Stage changes
git add --update

# Check Git diff-index
git diff-index --quiet HEAD --

if [ $? -eq 0 ]; then
	echo "No changes found, nothing to see/do here."
	exit 1
else
	echo "Changes found. Preparing commit."
	git config --local user.email "github-actions[bot]@users.noreply.github.com"
	git config --local user.name "github-actions[bot]"
	git commit -m "${COMMITMSG}"
fi
