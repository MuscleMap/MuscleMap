from setuptools import setup, find_packages

# Read requirements.txt and use it for the install_requires field
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='scripts',
    version='0.1.0',
    author='Richard Yin,  Kenneth Weber, Eddo Wesselink',  # Add another author's name
    description='An imaging toolbox for muscle.',
    url='https://github.com/MuscleMap/MuscleMap.git',
    packages=find_packages(),
    install_requires=required,
    entry_points={
        'console_scripts': [
            'mm_segment=scripts.mm_segment:main',
            'mm_extract_metrics=scripts.mm_extract_metrics:main',
            'mm_gui=scripts.mm_gui:main' 
        ]
    },
    package_data={
        'scripts': ['models/**/*.json', 'models/**/*.pth'],
    },
    include_package_data=True,
    python_requires='>=3.6',
)
