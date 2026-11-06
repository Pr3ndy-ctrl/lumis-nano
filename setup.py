from setuptools import setup

setup(
    name='lumis-nano',
    version='0.1.1',
    description='Axiom Photonics / Open Source Division - Photonic Matrix Compiler',
    author='Dwain Prendergast',
    author_email='founder@axiomprocessor.com',
    url='https://github.com/Pr3ndy-ctrl/lumis-nano',
    py_modules=['lumis_nano'],
    install_requires=[
        'numpy',
    ],
    entry_points={
        'console_scripts': [
            'lumis-nano=lumis_nano:main',
        ],
    },
    python_requires='>=3.6',
)
