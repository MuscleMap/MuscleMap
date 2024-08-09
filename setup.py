from setuptools import setup, find_packages

setup(
    name='scripts',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A package for muscle and fat segmentation using Python, with a GUI component.',
    url='https://github.com/MuscleMap/MuscleMap.git',
    packages=find_packages(),
    install_requires=[
        'monai==1.3.2',
        'torch==2.4.0',
        'numpy==1.24.4',
        'nibabel==5.2.1',
        'scikit-learn==1.3.2',
        'pandas==2.0.3',
        'scikit-image'

    ],
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
