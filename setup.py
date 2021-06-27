from setuptools import setup, find_packages

setup(
    name='chessocr',
    version='0.0.1',
    description='chess board finder',
    author='idrisr',
    author_email='id@hippoplant.com',
    keywords=['chess', 'ocr'],
    #  entry_points={'console_scripts': ['kaggle = kaggle.cli:main']},
    install_requires=[
        'fastai',
    ],
    packages=find_packages(),
    license='Apache 2.0')
