from setuptools import setup, find_packages
from pathlib import Path

with open('requirements.txt') as f:
    required = f.read().splitlines()

long_description = Path('README.md').read_text(encoding='utf-8')

setup(
    name='MuscleMap',
    version='2.0',
    author='Kenneth Weber, Eddo Wesselink, Benjamin DeLeener, Brian Kim, Richard Yin, Steffen Bollmann',
    description='A toolbox for muscle imaging.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/MuscleMap/MuscleMap.git',
    packages=find_packages(),
    install_requires=required,
    entry_points={
        'console_scripts': [
            'mm_segment=scripts.mm_segment:main',
            'mm_extract_metrics=scripts.mm_extract_metrics:main',
            'mm_gui=scripts.mm_gui:main',
            'mm_qc_gui=scripts.mm_qc_gui:main',
            'mm_register_to_template=scripts.mm_register_to_template:main',
            'mm_setup=scripts.mm_setup:main',
        ]
    },
    python_requires='>=3.11',
)
