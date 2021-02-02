from setuptools import setup


setup(
    name='milp',
    version='1.0',
    packages=['milp'],
    package_dir={'milp': 'milp'},
    install_requires=[
        'matplotlib',
        'numpy',
        'scipy',
    ],
)

