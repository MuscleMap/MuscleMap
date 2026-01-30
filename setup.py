from setuptools import setup, find_packages

# Read requirements.txt and use it for the install_requires field
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='scripts',
    version='1.2',
    authors='Kenneth Weber, Eddo Wesselink, Benjamin DeLeener,Brian Kim, Richard Yin',  
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
    package_data={
        'scripts': ['models/**/*.json', 'models/**/*.pth', 'templates/**/*nii.gz'],
    },
    include_package_data=True,
    python_requires='>=3.6',
)
