from setuptools import setup, find_packages

setup(
    name='safe',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'numpy==1.26.4',
        'pyfiglet==1.0.2',
        'r2pipe==1.9.4',
        'scikit-learn==1.6.0',
        'scipy==1.13.1',
        'tensorflow==2.18.0',
        'tqdm==4.67.1',
        'matplotlib==3.10.0'
    ],
)