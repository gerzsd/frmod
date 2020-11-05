"""
Input and output utilities for handling raster files.

@author: Dávid Gerzsenyi
"""

import os
import gdal
import osr


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
