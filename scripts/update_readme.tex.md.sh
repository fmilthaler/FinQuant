#!/bin/bash

replace_patterns() {
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

# Specify the file path for README.tex.md
tex_file_path="README.tex.md"

# Call the function to perform the replacements
replace_patterns "$tex_file_path"