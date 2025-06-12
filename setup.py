from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path: str) -> List[str]:
    """
    This function will return the list of requirements
    """
    try:
        with open(file_path) as file_obj:
            requirements = file_obj.readlines()
            requirements = [req.strip() for req in requirements if req.strip() and not req.startswith('#') and not req.startswith('-e .')]
        return requirements
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return []   

setup(
    name='networksecurity',
    packages=find_packages(),
    version='0.0.1',
    description='This is e2e ML ops project',
    author='Souvik Paul',
    license='',
    install_requires=get_requirements('requirements.txt')
)

