from setuptools import setup, find_packages

def fetch_requirements():
    with open("requirements.txt", "r", encoding="utf-8", errors="ignore") as f:
        reqs = f.read().strip().split("\n")
    return reqs

setup(
    name='swarm-contrastive decomposition',
    version='0.1.0',
    author='Agnese Grison',
    author_email='agnese.grison16@imperial.ac.uk',
    description='swarm contrastive decomposition',

    url='https://github.com/AgneGris/swarm-contrastive decomposition',

    packages=find_packages(),
    install_requires=fetch_requirements(),
)