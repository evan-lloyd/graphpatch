from setuptools import setup, find_packages

setup(
    name="tox_init_env",
    version="0.1",
    packages=find_packages("./tox_init_env"),
    entry_points={
        "tox": [
            "tox_init_env=tox_init_env",
        ],
    },
)
