from setuptools import setup, find_packages


setup(
    name='gentrl',
    version='0.1',
    python_requires='>=3.5.0',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.15',
        'pandas>=0.23',
        'scipy>=1.1.0',
        'torch==1.0.1',
        'molsets==0.1.3'
    ],
    description='Generative Tensorial Reinforcement Learning (GENTRL)',
)
