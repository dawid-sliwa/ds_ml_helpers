from setuptools import setup, find_packages

setup(
    name='ds_ml_data_kit',
    version='0.1',
    packages=find_packages(),
    description='Functions and helpers to accelerate ml and data analysis tasks currently functions only for pytorch',
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'matplotlib',
        'torch',
        'torchvision',
    ],
    author='Dawid Śliwa',
    license='MIT',
    python_requires='>=3.6',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)