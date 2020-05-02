from setuptools import setup, find_packages

setup(
    name='tl-env',
    version='0.0.1',
    author='Mohammad Hussein Tavakoli Bina',
    author_email='mhtb32@gmail.com',
    description='A package based on highway-env for temporal logic based reward shaping',
    url='https://github.com/mhtb32/tl-env',
    packages=find_packages(exclude=['scripts', 'tests']),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research'
    ],
    python_requires='>=3.6',
    install_requires=['gym', 'highway-env'],
    tests_require=['pytest']
)
