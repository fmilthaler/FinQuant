#!/bin/bash

update_version_readme() {
	local version_file="version"
	local readme_md="README.md"

	# Read the current version from the "version" file
	local current_version=$(grep -Eo 'version=([0-9]+\.){2}[0-9]+' "$version_file" | cut -d'=' -f2)

	update_file() {
		local file=$1
		sed -i "s/pypi-v[0-9]\+\.[0-9]\+\.[0-9]\+/pypi-v$current_version/" "$file"
		echo "Version updated to $current_version in $file"
	}

	# Update version in README.md
	update_file "$readme_md"
}

# Call the update_version function
update_version_readme