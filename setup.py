"""
Setup conf for FLEX
"""
import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='FLEX',
    version='1.0',
    description='Flex protocol',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(exclude=['test', 'test.*']),
    python_requires='>=3.6',
    include_package_data=True,
)
