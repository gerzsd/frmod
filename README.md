Click the Binder button below for the interactive guide on frmod's features.  

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/gerzsd/frmod/main?filepath=frmod_demo.ipynb)  
  
# frmod - Frequency ratio modeller
Landslide susceptibility analysis of raster grids using a frequency ratio model style approach.

The script uses a probabilistic method for landslide susceptibility assessment. It is assumed, that landslide-affected areas in the future will have similar terrain and environmental conditions to the already landslide-affected areas. The inputs of the analysis are the landslide sample areas and the continuous or categorical data layers of the analyzed variables. The method works with raster grids. The analysis has two variations, the frequency ratio and the likelihood ratio.  
The steps of the analysis:  

1. Partition the study area into landslide and non-landslide subareas  
2. Compute the frequency distribution of the analyzed variables for the landslide, the non-landslide, and the total area 
3. Compute the ratios (weights)
- Frequency ratio: Take the ratio of the landslide and total area frequency distributions - *the frequency ratio* - for each analyzed variable
- Likelihood ratio: Take the ratio of the landslide and non-landslide frequency distributions - *the likelihood ratio* - for each analyzed variable
4. Create the **weighted grids**: assign the ratios to the corresponding values of the analyzed variable grids
5. Get the landslide **susceptibility grid**: average the **weighted grids** or take the product of the **weighted grids**

The frmod script uses k-fold cross validation with random splits to evaluate the results.
1. The landslide area is split into equal sized parts, called splits.
2. One part is attached to the non-landslide area, these are the validation pixels
3. The result of the analysis is evaluated by checking the number of validation pixels in the different susceptibility categories
4. This process is then repeated with each split
5. The final susceptibility estimates are the average of the results of the runs with the different splits

The package contains two modules:
- **analysis:** Conducts the frequency ratio analysis using continuous and thematic rasters and a landslide mask raster. Uses k-fold cross validation to evaluate the results.  

- **utils:** Utilities for handling the input and output of georeferenced raster grids. Based mainly on code snippets from the Python GDAL/OGR Cookbook 1.0 (https://pcjericks.github.io/py-gdalogr-cookbook/)

## Installation and GDAL

GDAL is optional. `pip install .` installs frmod with numpy, pandas and matplotlib only. `pip install ".[gdal]"` also installs the GDAL Python bindings (the GDAL library must be available on the system). The conda `environment.yml` includes GDAL.

- **Without GDAL**, SAGA grids (`.sdat` + `.sgrd`) are read with numpy: `utils.raster2array_saga`, `utils.get_nodata_value_saga`. The `analysis` module uses them automatically. Results can be saved as SAGA grids with `utils.array2raster_saga`.
- **With GDAL**, GDAL reads the rasters and any GDAL-compatible format (eg. GeoTIFF) can be used. `utils.array2raster` writes them.

@ Dávid Gerzsenyi, 2024


