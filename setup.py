import setuptools

# get version/release from file
with open("version", "r") as f:
    ver = dict(x.rstrip().split("=") for x in f)

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="FinQuant",
    version=ver["version"],
    author="Frank Milthaler",
    author_email="f.milthaler@gmail.com",
    description="A program for financial portfolio management, analysis and optimisation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fmilthaler/FinQuant",
    download_url="https://github.com/fmilthaler/FinQuant/archive/v{}.tar.gz".format(
        ver["release"]
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
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
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
    python_requires=">=3.5",
    install_requires=[
        "quandl",
        "yfinance",
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "pytest",
    ],
    project_urls={"Documentation": "https://finquant.readthedocs.io"},
)
