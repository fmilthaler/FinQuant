#!/bin/sh

echo "Updating README files:"
# Update version number in README files:
scripts/update_version_readme.sh
# Update README.tex.md
scripts/update_readme.tex.md.sh

# Code formatting with isort and black
echo "Code formatting with isort and black:"
isort $(git ls-files '*.py')
black $(git ls-files '*.py')

# Stage changes
#git add $(git ls-files)
git add --udpate

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