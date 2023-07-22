#!/bin/bash

update_version_readme() {
	local version_file="version"
	local readme_md="$1"

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

update_readme_tex() {
	local file_path="$1"

	# Copy README.md to README.tex.md
	cp README.md "$file_path"

	# Read the contents of README.tex.md
	local content=$(<"$file_path")

	# Replace patterns
	content=$(echo "$content" | sed -E "s/<img src=\"(.*?)\" align=middle width=194.52263655pt height=46.976899200000005pt\/>/\$\\\\displaystyle\\\\dfrac{\\\\text{price}_{t_i} - \\\\text{price}_{t_0} + \\\\text{dividend}}{\\\\text{price}_{t_0}}\$/")

	content=$(echo "$content" | sed -E "s/<img src=\"(.*?)\" align=middle width=126.07712039999997pt height=48.84266309999997pt\/>/\$\\\\displaystyle\\\\dfrac{\\\\text{price}_{t_i} - \\\\text{price}_{t_{i-1}}}{\\\\text{price}_{t_{i-1}}}\$/")

	content=$(echo "$content" | sed -E "s/<img src=\"(.*?)\" align=middle width=208.3327686pt height=57.53473439999999pt\/>/\$\\\\displaystyle\\\\log\\\\left(1 + \\\\dfrac{\\\\text{price}_{t_i} - \\\\text{price}_{t_{i-1}}}{\\\\text{price}_{t_{i-1}}}\\\\right)\$/")

	# Write the updated contents back to README.tex.md
	echo "$content" > "$file_path"
}

# Update both readme files:
update_version_readme "README.md"
update_readme_tex "README.tex.md"