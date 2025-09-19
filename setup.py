import json
import os
from setuptools import setup, find_packages


def get_other_requirements():
    other_requirements = {}
    for file in os.listdir('./other_requirements'):
        with open(f'./other_requirements/{file}', encoding='utf-8') as rq:
            requirements = json.load(rq)
            other_requirements.update(requirements)
            return other_requirements

setup(
    version="0.1.0",
    packages=find_packages(exclude=['tests']),
    install_requires=[
        'numpy<2.0.0',
        'torch',
        'transformers==4.43.3',
        'tiktoken==0.7.0',
        'torchtune'
        'triton==3.1.0'
        'tiktoken==0.6.0',
        'datasets==2.20.0',
        'hydra-core',
        'wandb',
        'flash-attn',
    ],
)
