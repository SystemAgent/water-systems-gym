from setuptools import setup, find_packages


setup(
    name="water-systems-gym",
    version="0.1",
    packages=find_packages(),
    description="The data science part of zzzax",
    url='https://github.com/SystemAgent/water-systems-gym',
    extras_require={
        'dev': ['ipython', 'ipdb'],
        'test': ['pytest', 'mock'],
    },
    project_urls={
        'Source': 'https://github.com/SystemAgent/water-systems-gym',
        'Bug Reports': 'https://github.com/SystemAgent/water-systems-gym/issues',
    },
)
