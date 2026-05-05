from setuptools import setup, find_packages

# Read requirements.txt and use it for the install_requires field
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='scripts',
    version='2.0',
    authors='Kenneth Weber, Eddo Wesselink, Benjamin DeLeener, Brian Kim, Richard Yin, Steffen Bollmann',
    description='A toolbox for muscle imaging.',
    url='https://github.com/MuscleMap/MuscleMap.git',
    packages=find_packages(),
    install_requires=required,
    entry_points={
        'console_scripts': [
            'mm_segment=scripts.mm_segment:main',
            'mm_extract_metrics=scripts.mm_extract_metrics:main',
            'mm_gui=scripts.mm_gui:main',
            'mm_register_to_template=scripts.mm_register_to_template:main' 
        ]
    },
    python_requires='>=3.11',
)
