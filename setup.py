from setuptools import setup, find_packages
from typing import List

HYPON_E_DOT = '-e .'

def get_requirements(filepath: str) -> List[str]:
    requirement = []

    with open(filepath) as f:
        requirement = f.readlines()
        requirement = [i.replace('\n', '') for i in requirement]


        if HYPON_E_DOT in requirement:
            requirement.remove(HYPON_E_DOT)


setup(name='MLPipelines',
      version='0.0.1',
      description='ML Pipelines projects',
      author='Aryan nandi',
      author_email='aryannandi63@gmail.com',
      packages=find_packages(),
      install_requires=get_requirements("requirements.txt")
     )

