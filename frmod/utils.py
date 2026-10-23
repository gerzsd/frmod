"""
Input and output utilities for handling raster files.

Based mainly on code snippets from the Python GDAL/OGR Cookbook 1.0
https://pcjericks.github.io/py-gdalogr-cookbook/

@author: Dávid Gerzsenyi
"""

import os
import shutil

import numpy as np

# The data types of SAGA grids: DATAFORMAT value in the .sgrd header -> numpy
# type. Used by raster2array_saga (reading) and array2raster_saga (writing).
_SAGA_DATA_TYPES = {
    'BYTE': np.int8,
    'BYTE_UNSIGNED': np.uint8,
    'SHORTINT': np.int16,
    'SHORTINT_UNSIGNED': np.uint16,
    'INTEGER': np.int32,
    'INTEGER_UNSIGNED': np.uint32,
    'FLOAT': np.float32,
    'DOUBLE': np.float64,
}


def _import_gdal():
    """Return the osgeo.gdal module, or None if GDAL is not installed."""
    try:
        import osgeo.gdal as gdal
    except ImportError:
        return None
    return gdal


def _is_saga_file(rasterfn):
    """Return True if the path has a SAGA grid extension (.sdat or .sgrd)."""
    extension = os.path.splitext(os.fspath(rasterfn))[1].lower()
    return extension in ('.sdat', '.sgrd')


def _read_saga_header(rasterfn):
    """
    Read the .sgrd header of a SAGA grid.

    Parameters
    ----------
    rasterfn : str
        Path to a SAGA .sgrd header or .sdat data file.

    Returns
    -------
    header : dict
        The header lines as {KEY: value}. The keys are upper case.

    """
    headerfn = os.path.splitext(os.fspath(rasterfn))[0] + '.sgrd'
    header = {}
    with open(headerfn, encoding='utf-8-sig', errors='replace') as header_file:
        for line in header_file:
            key, separator, value = line.partition('=')
            if separator:
                header[key.strip().upper()] = value.strip()
    return header


def raster2array(rasterfn):
    """
    Raster to array.

    GDAL reads the raster. If GDAL is not installed, SAGA grids (.sdat) are
    read with raster2array_saga instead.

    Parameters
    ----------
    rasterfn : str

        Path to the raster to be converted to an array.

    Returns
    -------
    band_as_array : Array
        The first band of the input raster as a 2D array.
        The nodata values of the input raster are masked.

    """
    gdal = _import_gdal()
    if gdal is None:
        if _is_saga_file(rasterfn):
            return raster2array_saga(rasterfn)
        raise ImportError("GDAL is not installed. Without GDAL, only SAGA "
                          "grids (.sdat) can be read.")
    raster = gdal.Open(rasterfn)
    band = raster.GetRasterBand(1)
    band_as_array = band.ReadAsArray()
    return band_as_array


def raster2array_saga(rasterfn):
    """
    Read a SAGA grid from its .sgrd header and .sdat data file without GDAL.

    Either the .sgrd header or its matching .sdat data file may be supplied.
    The returned array contains the first (and only) grid band; no-data values
    are left unchanged.

    Parameters
    ----------
    rasterfn : str
        Path to a SAGA .sgrd header or .sdat data file.

    Returns
    -------
    band_as_array : numpy.ndarray
        The grid as a 2D array, ordered from top to bottom.

    """
    rasterfn = os.fspath(rasterfn)
    extension = os.path.splitext(rasterfn)[1].lower()

    if extension == '.sgrd':
        headerfn = rasterfn
        datafn = None
    elif extension == '.sdat':
        headerfn = os.path.splitext(rasterfn)[0] + '.sgrd'
        datafn = rasterfn
    else:
        raise ValueError("Expected a SAGA .sgrd or .sdat file.")

    header = {}
    with open(headerfn, encoding='utf-8-sig') as header_file:
        for line in header_file:
            if '=' in line:
                key, value = line.split('=', 1)
                header[key.strip().upper()] = value.strip()

    required = ('DATAFORMAT', 'CELLCOUNT_X', 'CELLCOUNT_Y')
    missing = [key for key in required if key not in header]
    if missing:
        raise ValueError(
            "Missing required SAGA grid metadata: {}".format(
                ', '.join(missing)))

    data_format = header['DATAFORMAT'].upper()
    data_types = _SAGA_DATA_TYPES
    if data_format not in data_types:
        raise ValueError(
            "Unsupported SAGA grid data format: {}".format(data_format))

    cols = int(header['CELLCOUNT_X'])
    rows = int(header['CELLCOUNT_Y'])
    if cols <= 0 or rows <= 0:
        raise ValueError("SAGA grid dimensions must be positive.")

    offset = int(header.get('DATAFILE_OFFSET', '0'))
    if offset < 0:
        raise ValueError("SAGA grid data offset must not be negative.")

    if datafn is None:
        datafile = header.get('DATAFILE')
        if datafile:
            datafile = datafile.strip('"\'')
            datafn = os.path.join(os.path.dirname(headerfn), datafile)
        else:
            datafn = os.path.splitext(headerfn)[0] + '.sdat'

    dtype = np.dtype(data_types[data_format])
    if dtype.itemsize > 1:
        byteorder_big = header.get('BYTEORDER_BIG', 'FALSE').upper() == 'TRUE'
        dtype = dtype.newbyteorder('>' if byteorder_big else '<')

    value_count = rows * cols
    band_as_array = np.fromfile(
        datafn, dtype=dtype, count=value_count, offset=offset)
    if band_as_array.size != value_count:
        raise ValueError(
            "SAGA data file is too small for the dimensions in its header.")

    band_as_array = band_as_array.astype(dtype.newbyteorder('='), copy=False)
    band_as_array = band_as_array.reshape((rows, cols))

    if header.get('TOPTOBOTTOM', 'FALSE').upper() != 'TRUE':
        band_as_array = np.flipud(band_as_array).copy()

    return band_as_array


def array2raster(rasterfn, new_raster_fn, array, driver='SAGA'):
    """
    Write an array to a GDAL-compatible raster file.

    Parameters
    ----------
    rasterfn : str
        Path to the raster used as a sample. The function copies
        the characterictics of this raster to the new raster.
    new_raster_fn : str
        Path and name of the new raster.
    array : Array
        Array to write to raster file.
    driver : str, optional
        GDAL raster driver to use for writing the file.
        See options at: https://gdal.org/drivers/raster/index.html
        The default is 'SAGA'.

    Returns
    -------
    None.

    """
    gdal = _import_gdal()
    if gdal is None:
        raise ImportError("GDAL is needed to write raster files. "
                          "SAGA grids can be written with array2raster_saga.")
    import osgeo.osr as osr

    raster = gdal.Open(rasterfn)
    geotransform = raster.GetGeoTransform()
    origin_x = geotransform[0]
    origin_y = geotransform[3]
    pixel_width = geotransform[1]
    pixel_height = geotransform[5]
    cols = raster.RasterXSize
    rows = raster.RasterYSize

    driver = gdal.GetDriverByName(driver)
    out_raster = driver.Create(new_raster_fn, cols, rows, 1,
                               gdal.GDT_Float32)
    out_raster.SetGeoTransform((origin_x, pixel_width, 0, origin_y, 0,
                                pixel_height))
    outband = out_raster.GetRasterBand(1)
    outband.WriteArray(array)
    out_raster_SRS = osr.SpatialReference()
    out_raster_SRS.ImportFromWkt(raster.GetProjectionRef())
    out_raster.SetProjection(out_raster_SRS.ExportToWkt())
    outband.FlushCache()


def array2raster_saga(rasterfn, new_raster_fn, array, nodata=-99999.):
    """
    Write an array to a SAGA grid without GDAL.

    Like array2raster, the new grid gets its size, position, cell size and
    projection from a sample grid. The data type of the new grid is the data
    type of the array, eg. float32 -> FLOAT, uint8 -> BYTE_UNSIGNED. Note that
    a float64 array is written as DOUBLE, use array.astype('float32') to
    get a smaller file.

    The rows are stored from the bottom up and the values are little-endian,
    like in the sample SAGA grids. Two files are written: .sgrd (header) and
    .sdat (data). The .prj (projection) file of the sample grid is copied if
    it exists.

    Parameters
    ----------
    rasterfn : str
        Path to the SAGA grid (.sgrd or .sdat) used as a sample.
    new_raster_fn : str
        Path of the new grid, ending with .sgrd or .sdat.
    array : Array
        2D array to write, ordered from top to bottom, like the output of
        raster2array_saga. Its shape must match the sample grid. Supported
        types: int8, uint8, int16, uint16, int32, uint32, float32, float64
        and bool (written as BYTE_UNSIGNED).
    nodata : int / float, optional
        The nodata value written to the header. The default is -99999.,
        the nodata value of the frmod results. The values of the array
        are not changed.

    Returns
    -------
    None.

    """
    if not (_is_saga_file(rasterfn) and _is_saga_file(new_raster_fn)):
        raise ValueError("Expected SAGA .sgrd or .sdat files.")

    # Grid system of the sample grid: size, position, cell size.
    sample = _read_saga_header(rasterfn)
    required = ('CELLCOUNT_X', 'CELLCOUNT_Y', 'POSITION_XMIN',
                'POSITION_YMIN', 'CELLSIZE')
    missing = [key for key in required if key not in sample]
    if missing:
        raise ValueError(
            "Missing required SAGA grid metadata: {}".format(
                ', '.join(missing)))

    array = np.asarray(array)
    if array.dtype == bool:
        array = array.astype(np.uint8)
    sample_shape = (int(sample['CELLCOUNT_Y']), int(sample['CELLCOUNT_X']))
    if array.shape != sample_shape:
        raise ValueError("The shape of the array {} does not match the "
                         "sample grid {}.".format(array.shape, sample_shape))

    # Data type: DATAFORMAT value belonging to the numpy type of the array.
    data_formats = {np.dtype(np_type): name
                    for name, np_type in _SAGA_DATA_TYPES.items()}
    dtype = array.dtype.newbyteorder('=')
    if dtype not in data_formats:
        raise ValueError(
            "Unsupported array type: {}. Supported types: {}.".format(
                array.dtype, ', '.join(str(t) for t in data_formats)))

    new_base = os.path.splitext(os.fspath(new_raster_fn))[0]

    # Data file: little-endian values, rows from the bottom up.
    data = np.ascontiguousarray(np.flipud(array),
                                dtype=dtype.newbyteorder('<'))
    data.tofile(new_base + '.sdat')

    # Header file.
    header = [
        ('NAME', os.path.basename(new_base)),
        ('DESCRIPTION', ''),
        ('UNIT', ''),
        ('DATAFORMAT', data_formats[dtype]),
        ('DATAFILE_OFFSET', '0'),
        ('BYTEORDER_BIG', 'FALSE'),
        ('TOPTOBOTTOM', 'FALSE'),
        ('POSITION_XMIN', sample['POSITION_XMIN']),
        ('POSITION_YMIN', sample['POSITION_YMIN']),
        ('CELLCOUNT_X', sample['CELLCOUNT_X']),
        ('CELLCOUNT_Y', sample['CELLCOUNT_Y']),
        ('CELLSIZE', sample['CELLSIZE']),
        ('Z_FACTOR', '1.000000'),
        ('Z_OFFSET', '0.000000'),
        ('NODATA_VALUE', '{0!r};{0!r}'.format(float(nodata))),
    ]
    with open(new_base + '.sgrd', 'w', encoding='utf-8') as header_file:
        for key, value in header:
            header_file.write('{}\t= {}\n'.format(key, value))

    # Projection file.
    sample_prj = os.path.splitext(os.fspath(rasterfn))[0] + '.prj'
    new_prj = new_base + '.prj'
    if (os.path.isfile(sample_prj)
            and os.path.abspath(sample_prj) != os.path.abspath(new_prj)):
        shutil.copyfile(sample_prj, new_prj)


def get_nodata_mask(rasterfn):
    """
    Get nodata mask.

    Parameters
    ----------
    rasterfn : str
        Path to the GDAL-compatible raster file.

    Returns
    -------
    nodata_mask : ndarray
        Nodata mask array.

    """
    arr = raster2array(rasterfn)
    nodata_value = get_nodata_value(rasterfn)
    nodata_mask = arr == nodata_value
    return nodata_mask


def get_nodata_value(rasterfn):
    """
    Fetch the nodata value of the input file.

    GDAL reads the nodata value. If GDAL is not installed, the nodata value
    of SAGA grids (.sdat) is read with get_nodata_value_saga instead.

    Parameters
    ----------
    rasterfn : str
        Path to the input file.

    Returns
    -------
    Int / float
        The nodata value of the input file.

    """
    gdal = _import_gdal()
    if gdal is None:
        if _is_saga_file(rasterfn):
            return get_nodata_value_saga(rasterfn)
        raise ImportError("GDAL is not installed. Without GDAL, only SAGA "
                          "grids (.sdat) can be read.")
    raster = gdal.Open(rasterfn)
    band = raster.GetRasterBand(1)
    return band.GetNoDataValue()


def get_nodata_value_saga(rasterfn):
    """
    Fetch the nodata value of a SAGA grid from its .sgrd header without GDAL.

    The header stores a nodata range, eg. "-99999.0;-99999.0". The first
    value is returned, like GDAL does.

    Parameters
    ----------
    rasterfn : str
        Path to a SAGA .sgrd header or .sdat data file.

    Returns
    -------
    float or None
        The nodata value, None if the header does not have one.

    """
    if not _is_saga_file(rasterfn):
        raise ValueError("Expected a SAGA .sgrd or .sdat file.")
    header = _read_saga_header(rasterfn)
    if 'NODATA_VALUE' not in header:
        return None
    return float(header['NODATA_VALUE'].split(';')[0])


def get_output_path(fname, tag='_reclass'):
    """
    Add a tag to a filename  or path.

    Parameters
    ----------
    fname : str
        Filename.
    tag : str, optional
        Tag to add to the end of the file name, goes before the extension.
        The default is '_reclass'.

    Returns
    -------
    newName : str
        The new file name with the tag.

    """
    split_path = os.path.splitext(fname)
    new_name = split_path[0] + tag + split_path[1]
    return new_name
