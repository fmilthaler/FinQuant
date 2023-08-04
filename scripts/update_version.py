"""
Version Management Module

This module provides functions for managing version numbers based on branch names.

Functions:
----------
- `increment_version(version: str, branch_name: str) -> str`: Increment the version based on the branch name pattern.
- `increment_version_by(version: str, increment: str) -> str`: Increment the version by a given increment.
- `read_version_from_file(filename: str) -> Optional[str]`: Read the version from a file.
- `checkout_branch(branch_name: str) -> None`: Checkout a specific branch.
- `get_version_from_branch(filename: str, branch_name: str) -> Optional[str]`: Get version number from a specific
    branch.
- `compare_versions(version1: str, version2: str) -> int`: Compare two strings of version numbers.
- `write_version_to_file(filename: str, version: str) -> None`: Write the updated version back to the file.
- `parse_args() -> argparse.Namespace`: Parse command-line arguments.
- `main() -> None`: Main function that handles version updates based on branch names.

Exceptions:
-----------
- `VersionFileReadError`: Exception raised when there is an error reading a version file.
- `VersionUpdateError`: Exception raised when an error occurs during the update of a version.

"""

import argparse
import re
import subprocess
import sys
from typing import Optional, Tuple

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
    """
    Exception raised when there is an error reading a version file.
    """


class VersionUpdateError(Exception):
    """
    Exception raised when an error occurs during the update of a version.
    """


# Function to increment the version based on the branch name pattern
def increment_version(version: str, branch_name: str) -> str:
    """
    Increment the version number based on the branch name pattern.

    Parameters:
    -----------
    version (str): The current version number in "x.y.z" format.
    branch_name (str): The name of the branch being checked out.

    Returns:
    --------
    str: The updated version number after applying the increment.

    """

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
def increment_version_by(version: str, increment: str) -> str:
    """
    Increment the version by a given increment (e.g., "0.0.1" or "0.1.0" or "1.0.0").

    Parameters:
    -----------
    version (str): The current version number in "x.y.z" format.
    increment (str): The version increment to apply in "x.y.z" format.

    Returns:
    --------
    str: The updated version number after applying the increment.

    """

    version_parts = version.split(".")
    increment_parts = increment.split(".")

    new_version_parts = []
    for idx, part in enumerate(version_parts):
        if idx < len(increment_parts):
            new_version_parts.append(str(int(part) + int(increment_parts[idx])))
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
def read_version_from_file(filename: str) -> Optional[str]:
    """
    Read the version from a file.

    Parameters:
    -----------
    filename (str): The path to the file containing the version.

    Returns:
    --------
    Optional[str]: The version number read from the file, or None if not found.

    """

    with open(filename, "r") as file:
        version_content = file.read()
        version_match = re.search(r"version=(\d+\.\d+\.\d+)", version_content)
        if version_match:
            version = version_match.group(1)
        else:
            version = None
        return version


# Function to checkout a specific branch
def checkout_branch(branch_name: str) -> None:
    """
    Checkout a specific branch to access its content.

    Parameters:
    -----------
    branch_name (str): The name of the branch to be checked out.

    Returns:
    --------
    None

    """

    # Fetch the latest changes from the remote repository
    subprocess.run(["git", "fetch", "origin", branch_name], check=True)

    # Checkout the branch to access its content
    subprocess.run(["git", "checkout", branch_name], check=True)


# Function to get version number from a specific branch
def get_version_from_branch(filename: str, branch_name: str) -> Optional[str]:
    """
    Get the version number from a specific branch.

    Parameters:
    -----------
    filename (str): The path to the file containing the version.
    branch_name (str): The name of the branch from which to read the version.

    Returns:
    --------
    Optional[str]: The version number read from the file, or None if not found.

    """

    # Checkout branch
    checkout_branch(branch_name)

    # Get version from version file
    version = read_version_from_file(filename)

    # Read the version from the file
    return version


# Function to compare 2 strings of version numbers
def compare_versions(version1: str, version2: str) -> int:
    """
    Compare two strings of version numbers.

    Parameters:
    -----------
    version1 (str): The first version number to compare in "x.y.z" format.
    version2 (str): The second version number to compare in "x.y.z" format.

    Returns:
    --------
    int: -1 if version1 < version2, 1 if version1 > version2, 0 if they are equal.

    """

    def parse_version(version_str: str) -> Tuple[int, ...]:
        return tuple(map(int, version_str.split(".")))

    parsed_version1 = parse_version(version1)
    parsed_version2 = parse_version(version2)

    if parsed_version1 < parsed_version2:
        return -1
    elif parsed_version1 > parsed_version2:
        return 1
    else:
        return 0


# Write the updated version back to the file
def write_version_to_file(filename: str, version: str) -> None:
    """
    Write the updated version back to the file.

    Parameters:
    -----------
    filename (str): The path to the file to be updated.
    version (str): The updated version number in "x.y.z" format.

    Returns:
    --------
    None

    """

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
def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
    --------
    argparse.Namespace: An object containing the parsed arguments.

    """

    parser = argparse.ArgumentParser(description="Update version based on branch name.")
    parser.add_argument("base_branch", help="Base branch name")
    parser.add_argument("source_branch", help="Source branch name")
    return parser.parse_args()


# Main function
def main() -> None:
    """
    Main function that handles version updates based on branch names.

    Returns:
    --------
    None

    """

    args = parse_args()
    base_branch_name = args.base_branch
    source_branch_name = args.source_branch

    file_path = "version"

    if base_branch_name != "master":
        raise ValueError("Base branch name must be 'master'.")

    if source_branch_name is None:
        raise ValueError("Source branch name must not be empty/None.")

    # Get the version from the base branch
    current_version_base = get_version_from_branch(file_path, base_branch_name)
    # Get the version from source branch
    current_version_source = get_version_from_branch(file_path, source_branch_name)

    # Sanity check for version numbers of base and source branch
    if current_version_base is None or current_version_source is None:
        raise VersionFileReadError(
            f"Failed to read the version from {base_branch_name} or from branch."
        )

    # Increment the version based on the branch name pattern
    updated_version = increment_version(current_version_base, source_branch_name)

    print(f"Base branch: {base_branch_name}")
    print(f"Source branch: {source_branch_name}")
    print(f"Current version (base):   {current_version_base}")
    print(f"Current version (source): {current_version_source}")
    print(f"Updated version: {updated_version}")

    # Check if updated version is higher than version in base branch:
    version_comparison = compare_versions(updated_version, current_version_base)
    if version_comparison < 0:
        raise VersionUpdateError(
            "Error: Updated version is lower than version in base branch."
        )
    if version_comparison == 0:
        print("Version does not increase.")
        # Exit with error code 1
        sys.exit(1)
    if version_comparison > 0:
        if updated_version == current_version_source:
            print("Version is already updated.")
            # Exit with error code 1
            sys.exit(1)
        else:
            write_version_to_file(file_path, updated_version)
            print("Version updated in the file 'version'.")


if __name__ == "__main__":
    main()
