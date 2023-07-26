import argparse
import re
import sys

# Define the version increments based on the change type (patch, minor, major)
version_increments = {
    "patch": "0.0.1",
    "minor": "0.1.0",
    "major": "1.0.0",
}

# Define the branch name prefixes for each change type
branch_prefixes = {
    "patch": ["chore/", "refactor/", "bugfix/"],
    "minor": ["feature/"],
    "major": None,
}


class VersionFileReadError(Exception):
    pass


# Function to increment the version based on the branch name pattern
def increment_version(version, branch_name):
    for change_type, prefixes in branch_prefixes.items():
        prefixes = prefixes or []  # If None, set to an empty list
        for prefix in prefixes:
            if branch_name.startswith(prefix):
                increment = version_increments[change_type]
                return (
                    increment_version_by(version, increment) if increment else version
                )

    return version


# Function to increment the version by a given increment (e.g., "0.0.1" or "0.1.0" or "1.0.0")
def increment_version_by(version, increment):
    version_parts = version.split(".")
    increment_parts = increment.split(".")

    new_version_parts = []
    for i in range(len(version_parts)):
        if i < len(increment_parts):
            new_version_parts.append(
                str(int(version_parts[i]) + int(increment_parts[i]))
            )
        else:
            new_version_parts.append("0")

    # If increment is "0.1.0", reset the third digit to 0
    if increment == "0.1.0" and len(version_parts) > 2:
        new_version_parts[2] = "0"

    # If increment is "1.0.0", reset the second and third digit to 0
    if increment == "1.0.0" and len(version_parts) > 2:
        new_version_parts[1] = "0"
        new_version_parts[2] = "0"

    return ".".join(new_version_parts)


# Read the version from the file
def read_version_from_file(filename):
    with open(filename, "r") as file:
        version_content = file.read()
        version_match = re.search(r"version=(\d+\.\d+\.\d+)", version_content)

        if version_match:
            version = version_match.group(1)
        else:
            version = None

        return version


# Function to get the version from the base branch
def get_version_from_base(filename, base_branch_name ="master"):
    cmd = ["git", "show", f"{base_branch_name}:{filename}"]
    try:
        version_content = subprocess.check_output(cmd).decode("utf-8")
        version_match = re.search(r"version=(\d+\.\d+\.\d+)", version_content)

        if version_match:
            version = version_match.group(1)
            return version
        else:
            raise VersionFileReadError(f"Version not found in the {base_branch_name} branch.")
    except subprocess.CalledProcessError as e:
        raise VersionFileReadError(f"Failed to read the version from the {base_branch_name} branch.") from e


# Write the updated version back to the file
def write_version_to_file(filename, version):
    with open(filename, "r+") as file:
        file_content = file.read()
        updated_content = re.sub(
            r"version=\d+\.\d+\.\d+", f"version={version}", file_content
        )
        # Always set the release number to match the updated version number
        updated_content = re.sub(
            r"release=\d+\.\d+\.\d+", f"release={version}", updated_content
        )
        file.seek(0)
        file.write(updated_content)
        file.truncate()


# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Update version based on branch name.")
    parser.add_argument("base_branch", help="Base branch name")
    parser.add_argument("source_branch", help="Source branch name")
    return parser.parse_args()


# Main function
def main():
    args = parse_args()
    base_branch_name = args.base_branch
    source_branch_name = args.source_branch

    file_path = "version"

    if base_branch_name != "master":
        raise ValueError("Base branch name must be 'master'.")

    if source_branch_name is None:
        raise ValueError("Source branch name must not be empty/None.")

    # Get the version from the master branch
    current_version_base = get_version_from_base(file_path, base_branch_name)
    # Get the version from current branch
    current_version_source = read_version_from_file(file_path)

    if current_version_base is None or current_version_source is None:
        raise VersionFileReadError(f"Failed to read the version from {base_branch_name} or from branch.")

    # Increment the version based on the branch name pattern
    updated_version = increment_version(current_version_base, source_branch_name)

    print(f"Base branch: {base_branch_name}")
    print(f"Source branch: {source_branch_name}")
    print(f"Current version (master): {current_version_base}")
    print(f"Current version (branch): {current_version_source}")
    print(f"Updated version: {updated_version}")

    if updated_version == current_version_base:
        print("Version does not increase.")
        # Exit with error code 1
        sys.exit(1)
    elif updated_version == current_version_source:
        print("Version is already updated.")
        # Exit with error code 1
        sys.exit(1)
    else:
        write_version_to_file(file_path, updated_version)
        print("Version updated in the file 'version'.")


if __name__ == "__main__":
    main()
