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
        #add versions
        'numpy',
        'pandas',  # Assuming you use pandas
        'nibabel',
        'scikit-learn'


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
