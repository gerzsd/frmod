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

- **By default**, `VRaster` and `LandslideMask` read SAGA grids (`.sdat` + `.sgrd`) with numpy, GDAL is not needed (`utils.raster2array_saga`, `utils.get_nodata_value_saga`). Results can be saved as SAGA grids with `utils.array2raster_saga`.
- **With `use_gdal=True`**, eg. `VRaster(name, path, use_gdal=True)` and `LandslideMask(name, path, use_gdal=True)`, GDAL reads the rasters (`utils.raster2array`, `utils.get_nodata_value`) and any GDAL-compatible format (eg. GeoTIFF) can be used. GDAL must be installed. `utils.array2raster` writes rasters with GDAL.

frmod needs Python 3.8 or newer and numpy 1.22 or newer.

### Upgrading from 0.1.5 or earlier

Reading rasters without GDAL is now the default, so **inputs in other formats than SAGA grids (eg. GeoTIFF) need `use_gdal=True`**. Without it, `VRaster` and `LandslideMask` raise `ValueError: Expected a SAGA .sgrd or .sdat file.` See the [changelog](CHANGELOG.md) for the other changes, some of which can change the results on your data.

## Input data requirements

- All rasters (the landslide mask and every variable) must have the **same grid**: the same number of rows and columns, the same extent and the same coordinate reference system. frmod does not check this.
- Only the first band of each raster is used.
- In the landslide mask, `ls_marker` (default 1) marks the landslide pixels and `nls_marker` (default 0) marks the non-landslide pixels. Every other value, eg. nodata, is treated as outside of the study area.

## Running the tests

```
python -m unittest discover tests
```

The tests use small synthetic grids. The few tests that compare the SAGA reader and writer with GDAL are skipped if GDAL is not installed.

## Interpreting the cross validation

The validation pixels are chosen randomly, pixel by pixel. Neighbouring pixels of the same landslide can therefore end up in both the training and the validation set, which tends to make the success rates and AUC values optimistic. Treat them as a measure of how well the model reproduces the mapped landslide areas, not as an independent test. Note also that frmod reports the area under the success rate curve on a 0-100 scale, where a **lower value is better** and a random model scores 50. The conventional AUC is `1 - value / 100`.

@ Dávid Gerzsenyi, 2026


