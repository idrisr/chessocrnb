from setuptools import setup, find_packages

setup(
    name='sitegen',
    version='0.0.1',
    description='Blog site generator',
    author='idrisr',
    author_email='id@hippoplant.com',
    keywords=['sitegen', 'jekyll'],
    #  entry_points={'console_scripts': ['kaggle = kaggle.cli:main']},
    install_requires=[
        'nbconvert',
        'tzlocal',
        'traitlets',
    ],
    packages=find_packages(),
    license='Apache 2.0')
