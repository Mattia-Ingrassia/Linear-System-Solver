from setuptools import setup, find_packages

# TODO

setup(
    name='linear system solver',  # Replace with your package’s name
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        # List your dependencies here
    ],
    author='Mattia Ingrassia, Riccardo Ghilotti',  
    author_email='Your e-mail',
    description='A library for checking multiples of 2 and 5.',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # License type
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',

)