from setuptools import setup


setup(
    name='milp',
    version='0.0.2',
    packages=['milp'],
    package_dir={'milp': 'milp'},
    install_requires=[],
    install_requires=[
        'matplotlib',
        'numpy',
        'scipy',
    ],
)

