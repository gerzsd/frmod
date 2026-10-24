# Changelog

All notable changes to frmod are listed here. The newest version is on top.

## [0.2.0] - 2026-10-24


### Changed (action may be needed)

- **SAGA grids are read without GDAL by default.** `VRaster` and `LandslideMask` have a new `use_gdal` option (default `False`). Rasters in other formats, eg. GeoTIFF, need `use_gdal=True`, otherwise a `ValueError` is raised.
- **GDAL is an optional dependency.** `pip install .` no longer installs it, use `pip install ".[gdal]"` or the conda `environment.yml`. The invalid `osr` requirement was removed from `setup.py`.
- **Python 3.8 or newer and numpy 1.22 or newer are required** (`python_requires` was 3.6).

### Fixed (results can change)

- Custom `ls_marker` and `nls_marker` values of the `LandslideMask` were ignored in the analysis (the defaults 1 and 0 were always used). Non-default markers now work.
- With `product=True` and an even number of variables, nodata cells (-99999 squared) were treated as valid and became the highest susceptibility class. Cells that are nodata in any layer are now nodata in the result.
- The maximum value of a continuous layer was never reclassified and became nodata. The last bin now includes its upper edge, like `np.histogram`.
- `reclass_raster` ignored the nodata value of the layer. It has new `nodata` and `categorical` arguments: nodata cells are never reclassified, and categories that were not part of the analysis stay nodata instead of taking the ratio of the previous class.
- Categorical layers failed or got wrong bin edges when the layer was a masked array, in both `get_freq_ratios_classic` and `get_freq_ratios` (the latter crashed with `classic_mode=False`).
- `get_result()` appended its results every time it was called, so calling it twice gave 10 "folds" in `get_src()`, `get_auc()` and `plot_success_rates()`. It can now be called repeatedly.
- `np.quantile(..., interpolation='nearest')` does not work on NumPy 2.0 or newer. It uses `method='nearest'` now.

### Added

- `utils.raster2array_saga`, `utils.get_nodata_value_saga` and `utils.array2raster_saga`: read and write SAGA grids (`.sdat` + `.sgrd`) without GDAL.
- `utils.read_raster(path, use_gdal=False)`: reads a raster and its nodata value with or without GDAL.
- Unit tests in `tests/test_frmod.py`, run with `python -m unittest discover tests`.
- README sections about the installation, the input data requirements, upgrading, the tests and the interpretation of the cross validation.

### Known issues

- The input rasters are not checked for the same shape, extent and coordinate reference system.
- Empty non-landslide histogram bins give `inf` or `nan` ratios without a warning.
- The cross validation splits the landslide pixels randomly, so the validation is not spatially independent (see the README).

## [0.1.5] - 2024-03-24

- Alternative aggregation method: the weighted grids can be averaged or multiplied (`get_result(product=True)`).
- `FRAnalysis.auc_folds` is reset at the start of `get_auc()` (from 0.1.4).
- Documentation, license and tutorial updates.
