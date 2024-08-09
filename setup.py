from setuptools import setup, find_packages

setup(
    name='mm_package',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A package for muscle and fat segmentation using Python, with a GUI component.',
    url='https://github.com/MuscleMap/MuscleMap.git',
    packages=find_packages(),
    install_requires=[
        'monai==0.9.0',
        'torch==1.13.0',
        'numpy==1.23.5',
        'nibabel==5.1.0',
        'scikit-learn==1.3.0',
        'pandas==1.5.3',
        'scikit-image==0.24.0'

    ],
    entry_points={
        'console_scripts': [
            'mm_segment=mm_package.mm_segment:main',
            'mm_extract_metrics=mm_package.mm_extract_metrics:main',
            'mm_gui=mm_package.mm_gui' 
        ]
    },
    package_data={
        'mm_package': ['models/**/*.json', 'models/**/*.pth'],
    },
    include_package_data=True,
    python_requires='>=3.6',
)
