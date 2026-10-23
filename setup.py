import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='frmod',
    version='0.1.5',
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
    # numpy 1.22 is the first version with the method argument of np.quantile.
    python_requires='>=3.8',
    install_requires=['numpy>=1.22', 'matplotlib', 'pandas'],
    # GDAL is optional: SAGA grids can be read and written without it.
    # Install it with: pip install frmod[gdal]
    extras_require={'gdal': ['gdal']},
    license='MIT',
)