from setuptools import setup, find_packages
import os

def strip_comments(l):
    return l.split('#', 1)[0].strip()

def reqs(*f):
    return list(filter(None, [strip_comments(l) for l in open(os.path.join(os.getcwd(), *f)).readlines()]))

setup(
    name='velvet',
    version='0.0.2',
    description='カテゴリ変数からの特徴抽出',
    long_description='カテゴリ変数からの特徴抽出方法をまとめたパッケージです',
    author='Masashi Kishimoto',
    license='MIT',
    classifiers=[
        'Development Status :: 1 - Planning'
    ],
    install_requires=reqs('requirements.txt'),
    packages=find_packages(exclude=('tests', 'docs'))
)