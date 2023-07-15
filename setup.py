import setuptools

# get version/release from file
with open("version", "r") as f:
    version = dict(x.rstrip().split("=") for x in f)

# get long description from README
with open("README.md", "r") as fh:
    long_description = fh.read()

# get dependencies
def read_requirements(file_path):
    with open(file_path, "r") as f:
        return [line.strip() for line in f if line.strip()]

install_requires = read_requirements("requirements.txt")

extras_require = {
    "test": read_requirements("requirements_test.txt"),
    "dev": read_requirements("requirements_dev.txt"),
    "docs": read_requirements("requirements_docs.txt"),
    "cd": read_requirements("requirements_cd.txt"),
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
    download_url="https://github.com/fmilthaler/FinQuant/archive/v{}.tar.gz".format(
        version["release"]
    ),
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
