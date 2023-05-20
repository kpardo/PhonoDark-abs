# -*- coding: utf-8 -*-

"""
setup.py - Kris Pardo (kmpardo@usc.edu)
"""
__version__ = "0.0.0"

import sys
from setuptools import setup
from setuptools.command.test import test as TestCommand


class PyTest(TestCommand):
    user_options = [("pytest-args=", "a", "Arguments to pass to pytest")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = []

    def run_tests(self):
        import shlex
        import pytest

        if not self.pytest_args:
            targs = []
        else:
            targs = shlex.split(self.pytest_args)

        errno = pytest.main(targs)
        sys.exit(errno)


def readme():
    with open("README.md") as f:
        return f.read()


INSTALL_REQUIRES = [
    "numpy>=1.4.0",
    "scipy",
    "matplotlib",
]

EXTRAS_REQUIRE = {"all": ["pandas", "phonopy"]}

###############
## RUN SETUP ##
###############

# run setup.
setup(
    name="pda",
    version=__version__,
    description=("code that calculates the absorption of various dark matter models onto phonons"),
    long_description=readme(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 1 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
    keywords=["dark matter", "phonon"],
    url="https://github.com/kpardo/phonodark_abs",
    author=["Kris Pardo", "Tanner Trickle"],
    license="MIT",
    packages=[
        "pda",
    ],
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    tests_require=[
        "pytest==3.8.2",
    ],
    cmdclass={"test": PyTest},
    include_package_data=True,
    zip_safe=False,
)
