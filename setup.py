from setuptools import setup

install_requires = [
    "dill",  # pickle package is not able to pickle the FSAs
    "frozendict",
    "numpy",
    "pytest",
    "scipy",
]


setup(
    name="fsrnn",
    install_requires=install_requires,
    version="1.0",
    scripts=[],
    packages=["fsrnn"],
)
