from setuptools import setup, find_packages


def find_dependencies(requirements_file='requirements.txt'):
    with open(requirements_file, 'r') as file:
        return [line.strip() for line in file]

setup(
    name='phd_visualizations',
    version='0.1.0',
    url='https://github.com/juan11iguel/phd_visualizations',
    author='Juan Miguel Serrano',
    author_email='jmserrano@psa.es',
    description='Several visualizations implemented during my PhD studies, mainly using plotly (with plotly resampler) and matplotlib',
    packages=find_packages(),  # automatically find all packages and subpackages
    # install_requires=find_dependencies(),
)