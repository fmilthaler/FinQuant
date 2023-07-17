import shutil
import re

def replace_patterns(file_path):
    # Copy README.md to README.tex.md
    shutil.copyfile("README.md", file_path)

    # Read the contents of README.tex.md
    with open(file_path, "r") as file:
        content = file.read()

    # Replace patterns
    content = re.sub(
        r'The cumulative return: <img src="(.*?)" align=middle width=194.52263655pt height=46.976899200000005pt/>',
        r'The cumulative return: $\\displaystyle\\dfrac{\\text{price}_{t_i} - \\text{price}_{t_0} + \\text{dividend}}{\\text{price}_{t_0}}$',
        content,
    )

    content = re.sub(
        r'Percentage change of daily returns: <img src="(.*?)" align=middle width=126.07712039999997pt height=48.84266309999997pt/>',
        r'Percentage change of daily returns: $\\displaystyle\\dfrac{\\text{price}_{t_i} - \\text{price}_{t_{i-1}}}{\\text{price}_{t_{i-1}}}$',
        content,
    )

    content = re.sub(
        r'Log Return: <img src="(.*?)" align=middle width=208.3327686pt height=57.53473439999999pt/>',
        r'Log Return: $\\displaystyle\\log\\left(1 + \\dfrac{\\text{price}_{t_i} - \\text{price}_{t_{i-1}}}{\\text{price}_{t_{i-1}}}\\right)$',
        content,
    )

    # Write the updated contents back to README.tex.md
    with open(file_path, "w") as file:
        file.write(content)

# Specify the file path for README.tex.md
tex_file_path = "README.tex.md"

# Call the function to perform the replacements
replace_patterns(tex_file_path)