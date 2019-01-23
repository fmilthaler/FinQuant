import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='QuantPy-fmilthaler',
    version='0.0.1',
    author='Frank Milthaler',
    description='A program for financial portfolio management, analysis and optimisation',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/fmilthaler/QuantPy',
    packages=setuptools.find_packages(),
    classifiers=[
        'Development Status :: 1 - Alpha',
        'Intended Audience :: Finance',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    keywords=['finance', 'portfolio', 'investment',
    'numerical', 'optimisation', 'monte carlo',
    'efficient frontier', 'quantitative', 'quant'
    ],
    install_requires=[
        'quandl', 'numpy', 'pandas', 'scipy',
        'matplotlib'
    ],
    tests_require=[
        'pytest'
    ],
)
