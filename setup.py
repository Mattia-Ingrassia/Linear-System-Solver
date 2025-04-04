from setuptools import setup, find_packages

setup(
    name='linear system solver',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy'
    ],
    author='Mattia Ingrassia, Riccardo Ghilotti',  
    author_email='m.ingrassia3@campus.unimib.it, r.ghilotti@campus.unimib.it',
    description='A Python package for solving linear systems',
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)