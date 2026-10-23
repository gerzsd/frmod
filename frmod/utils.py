"""
Input and output utilities for handling raster files.

Based mainly on code snippets from the Python GDAL/OGR Cookbook 1.0
https://pcjericks.github.io/py-gdalogr-cookbook/

@author: Dávid Gerzsenyi
"""

import os

import numpy as np
import osgeo.gdal as gdal
import osgeo.osr as osr


def raster2array(rasterfn):
    """
    Raster to array.

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
    data_types = {
        'BYTE': np.int8,
        'BYTE_UNSIGNED': np.uint8,
        'SHORTINT': np.int16,
        'SHORTINT_UNSIGNED': np.uint16,
        'INTEGER': np.int32,
        'INTEGER_UNSIGNED': np.uint32,
        'FLOAT': np.float32,
        'DOUBLE': np.float64,
    }
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

    Parameters
    ----------
    rasterfn : str
        Path to the input file.

    Returns
    -------
    Int / float
        The nodata value of the input file.

    """
    raster = gdal.Open(rasterfn)
    band = raster.GetRasterBand(1)
    return band.GetNoDataValue()


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
