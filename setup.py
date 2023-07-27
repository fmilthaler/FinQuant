from typing import List

import setuptools


def read_file(file_path: str) -> str:
    with open(file_path, "r") as file:
        return file.read()


# get version/release from file
version = dict(line.rstrip().split("=") for line in read_file("version").splitlines())

# get long description from README
long_description = read_file("README.md")


# get dependencies
def read_requirements(file_path: str) -> List[str]:
    return [line.strip() for line in read_file(file_path).splitlines() if line.strip()]


install_requires = read_requirements("requirements.txt")

extras_require = {
    "cd": read_requirements("requirements_cd.txt"),
    "dev": read_requirements("requirements_dev.txt"),
    "docs": read_requirements("requirements_docs.txt"),
    "test": read_requirements("requirements_test.txt"),
}

setuptools.setup(
    name="FinQuant",
    version=version["version"],
    author="Frank Milthaler",
    author_email="f.milthaler@gmail.com",
    description="A program for financial portfolio management, analysis and optimisation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fmilthaler/FinQuant",
    download_url=f"https://github.com/fmilthaler/FinQuant/archive/v{version['release']}.tar.gz",
    license="MIT",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Other Audience",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords=[
        "finance",
        "portfolio",
        "investment",
        "numerical",
        "optimisation",
        "monte carlo",
        "efficient frontier",
        "quantitative",
        "quant",
    ],
    python_requires=">=3.10",
    install_requires=install_requires,
    extras_require=extras_require,
    project_urls={"Documentation": "https://finquant.readthedocs.io"},
)
