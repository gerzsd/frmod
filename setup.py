import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='frmod',
    version='0.1.3',
    author='Dávid Gerzsenyi',
    author_email='gerzsd@student.elte.hu',
    description='Landslide susceptibility analysis using frequency or likelihood ratios',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gerzsd/frmod",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires='>=3.6',
    install_requires=['numpy', 'matplotlib', 'pandas', 'gdal', 'osr'],
    license='MIT',
)