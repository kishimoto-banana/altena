from setuptools import setup, find_packages
import os


def strip_comments(l):
    return l.split('#', 1)[0].strip()


def reqs(*f):
    return list(
        filter(None, [
            strip_comments(l)
            for l in open(os.path.join(os.getcwd(), *f)).readlines()
        ]))


def read_file(filename):
    basepath = os.path.dirname(os.path.dirname((__file__)))
    filepath = os.path.join(basepath, filename)
    if os.path.exists(filepath):
        return open(filepath).read()
    else:
        return ''


setup(
    name='altena',
    version='0.0.1',
    description='Feature extraction for categorical variables',
    long_description=read_file('README.md'),
    author='Masashi Kishimoto',
    author_email='drehbleistift@gmail.com',
    url='https://github.com/kishimoto-banana/altena',
    classifiers=[
        'Development Status :: 4 - Beta', 'Programming Language :: Python',
        'Programming Language :: Python :: 3.6'
    ],
    install_requires=reqs('requirements.txt'),
    packages=find_packages(exclude=('tests', 'docs')))
