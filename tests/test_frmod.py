"""Regression tests for frmod.analysis.

Run them with:  python -m unittest discover tests   (or with pytest)

The tests use small synthetic grids instead of raster files: the functions
that read files in frmod.utils are replaced in setUp.
"""

import os
import pathlib
import tempfile
import unittest
from unittest import mock

import numpy as np

import frmod.utils as utils
from frmod.analysis import FRAnalysis, LandslideMask, VRaster, reclass_raster

NODATA = -99999.


class TestReclassRaster(unittest.TestCase):

    def test_last_bin_includes_upper_edge(self):
        vr = np.array([0., 1., 2.])
        reclassed = reclass_raster(vr, [10., 20.], np.array([0., 1., 2.]))
        # The largest value (2) belongs to the last bin, like in np.histogram.
        self.assertEqual(list(reclassed), [10., 20., 20.])

    def test_category_not_in_bins_stays_nodata(self):
        vr = np.array([1, 2, 3, 4])
        # Categories 1, 2 and 4 were analysed, 3 was not.
        bin_edges = np.array([1, 2, 4, 5])
        reclassed = reclass_raster(vr, [0.5, 1.5, 2.5], bin_edges,
                                   categorical=True)
        self.assertEqual(list(reclassed), [0.5, 1.5, NODATA, 2.5])

    def test_nodata_inside_the_value_range_is_kept(self):
        vr = np.array([0., 5., 10.])
        reclassed = reclass_raster(vr, [1., 2.], np.array([0., 5., 10.]),
                                   nodata=0.)
        self.assertEqual(list(reclassed), [NODATA, 2., 2.])


class TestFRAnalysis(unittest.TestCase):

    def setUp(self):
        # Dict {path: (array, nodata value)} that stands in for raster files.
        self.files = {}
        patches = [
            mock.patch.object(utils, 'raster2array',
                              lambda path: self.files[path][0].copy()),
            mock.patch.object(utils, 'get_nodata_value',
                              lambda path: self.files[path][1]),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def make_analysis(self, ls_marker=1, nls_marker=0, classic_mode=True):
        """Create an FRAnalysis of two layers and a mask on a 30 x 30 grid.

        The first 4 columns are outside of the study area (nodata everywhere).
        Returns the analysis and the boolean array of the outside cells.
        """
        rng = np.random.default_rng(0)
        elevation = rng.uniform(0, 100, (30, 30))
        slope = rng.uniform(0, 40, (30, 30))
        mask = np.full((30, 30), nls_marker, dtype=float)
        mask[10:15, 10:15] = ls_marker
        outside = np.zeros((30, 30), dtype=bool)
        outside[:, :4] = True
        for grid in (elevation, slope, mask):
            grid[outside] = NODATA
        self.files['elevation'] = (elevation, NODATA)
        self.files['slope'] = (slope, NODATA)
        self.files['mask'] = (mask, NODATA)

        np.random.seed(1)  # The folds are random.
        lsm = LandslideMask('mask', 'mask', ls_marker=ls_marker,
                            nls_marker=nls_marker, fold_count=5)
        var_list = [VRaster('elevation', 'elevation', bins=5),
                    VRaster('slope', 'slope', bins=5)]
        return FRAnalysis(lsm, var_list, classic_mode=classic_mode), outside

    def test_nodata_stays_nodata_in_the_result(self):
        # With two layers (-99999) * (-99999) used to become a valid 1e10.
        for product in (False, True):
            with self.subTest(product=product):
                fra, outside = self.make_analysis()
                fra.get_result(product=product)
                self.assertTrue(np.all(fra.fresult[outside] == NODATA))
                self.assertTrue(np.all(fra.fresult[~outside] >= 0))
                self.assertLess(fra.ranks[-1], 1e6)

    def test_custom_markers_give_the_same_result_as_default_markers(self):
        for classic_mode in (True, False):
            with self.subTest(classic_mode=classic_mode):
                default, _ = self.make_analysis(classic_mode=classic_mode)
                default.get_result()
                custom, _ = self.make_analysis(ls_marker=5, nls_marker=7,
                                               classic_mode=classic_mode)
                custom.get_result()
                np.testing.assert_array_equal(default.fresult, custom.fresult)

    def test_get_result_can_be_called_again(self):
        fra, _ = self.make_analysis()
        fra.get_result()
        first = fra.fresult.copy()
        fra.get_result()
        np.testing.assert_array_equal(fra.fresult, first)
        folds = fra.ls_mask.fold_count
        self.assertEqual(len(fra.v_dist), folds)
        self.assertEqual(len(fra.fold_percentiles), folds)
        self.assertEqual(len(fra.get_src()), folds)

    def test_highest_value_cell_is_not_lost(self):
        fra, outside = self.make_analysis()
        fra.get_result()
        fra.get_percentile_grid()
        # Every cell of the study area has a reclassified value and a percentile.
        self.assertTrue(np.all(fra.percentile_grid[~outside] >= 1))
        self.assertTrue(np.all(fra.rc_folds[0][0][~outside] >= 0))


class TestRaster2ArraySaga(unittest.TestCase):

    DATA_TYPES = {
        'BYTE': np.int8,
        'BYTE_UNSIGNED': np.uint8,
        'SHORTINT': np.int16,
        'SHORTINT_UNSIGNED': np.uint16,
        'INTEGER': np.int32,
        'INTEGER_UNSIGNED': np.uint32,
        'FLOAT': np.float32,
        'DOUBLE': np.float64,
    }
    SAMPLE_FILES = ['SRTM31_EG_GF_m', 'SRTM31_EG_GF_Slope_m',
                    'fdt100_14k', 'scarps']

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = folder.name

    def sample_path(self, name):
        here = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(here, '..', 'data', name + '.sdat')

    def write_grid(self, grid, data_format, name='grid', big_endian=False,
                   top_to_bottom=False, offset=0):
        """Write grid (first row = top row) as a SAGA grid, return its path."""
        dtype = np.dtype(self.DATA_TYPES[data_format])
        if dtype.itemsize > 1:
            dtype = dtype.newbyteorder('>' if big_endian else '<')
        # SAGA grids are stored from the bottom row up, unless TOPTOBOTTOM.
        stored = grid if top_to_bottom else grid[::-1]
        path = os.path.join(self.folder, name)
        with open(path + '.sdat', 'wb') as data_file:
            data_file.write(b'\0' * offset)
            data_file.write(stored.astype(dtype).tobytes())
        header = ['DATAFORMAT\t= ' + data_format,
                  'DATAFILE_OFFSET\t= ' + str(offset),
                  'BYTEORDER_BIG\t= ' + str(big_endian).upper(),
                  'TOPTOBOTTOM\t= ' + str(top_to_bottom).upper(),
                  'CELLCOUNT_X\t= ' + str(grid.shape[1]),
                  'CELLCOUNT_Y\t= ' + str(grid.shape[0])]
        with open(path + '.sgrd', 'w') as header_file:
            header_file.write('\n'.join(header) + '\n')
        return path

    def test_all_formats_byte_orders_and_row_orders(self):
        for data_format, np_type in self.DATA_TYPES.items():
            signed = not np.issubdtype(np_type, np.unsignedinteger)
            grid = (np.arange(12).reshape(3, 4) - (5 if signed else 0))
            grid = grid.astype(np_type)
            for big_endian in (False, True):
                for top_to_bottom in (False, True):
                    for offset in (0, 7):
                        with self.subTest(data_format=data_format,
                                          big_endian=big_endian,
                                          top_to_bottom=top_to_bottom,
                                          offset=offset):
                            path = self.write_grid(
                                grid, data_format, big_endian=big_endian,
                                top_to_bottom=top_to_bottom, offset=offset)
                            result = utils.raster2array_saga(path + '.sdat')
                            np.testing.assert_array_equal(result, grid)
                            self.assertEqual(result.dtype, grid.dtype)

    def test_header_and_data_file_paths_both_work(self):
        grid = np.arange(12, dtype=np.float32).reshape(3, 4)
        path = self.write_grid(grid, 'FLOAT')
        for given_path in (path + '.sgrd', path + '.sdat',
                           pathlib.Path(path + '.sdat')):
            np.testing.assert_array_equal(
                utils.raster2array_saga(given_path), grid)

    def test_nodata_values_and_header_scaling_are_left_unchanged(self):
        grid = np.array([[-99999., 1.], [2., 3.]], dtype=np.float32)
        path = self.write_grid(grid, 'FLOAT')
        with open(path + '.sgrd', 'a') as header_file:
            header_file.write('Z_FACTOR\t= 2.0\nZ_OFFSET\t= 10.0\n')
        np.testing.assert_array_equal(
            utils.raster2array_saga(path + '.sdat'), grid)

    def test_wrong_input_is_reported(self):
        grid = np.arange(12, dtype=np.float32).reshape(3, 4)
        path = self.write_grid(grid, 'FLOAT')
        with self.assertRaises(ValueError):  # Not a SAGA file.
            utils.raster2array_saga(path + '.tif')
        with open(path + '.sdat', 'r+b') as data_file:
            data_file.truncate(20)
        with self.assertRaises(ValueError):  # Smaller than the header says.
            utils.raster2array_saga(path + '.sdat')
        bad_header = self.write_grid(grid, 'FLOAT', name='bad_header')
        with open(bad_header + '.sgrd', 'w') as header_file:
            header_file.write('DATAFORMAT\t= COLOR\nCELLCOUNT_X\t= 4\n'
                              'CELLCOUNT_Y\t= 3\n')
        with self.assertRaises(ValueError):  # Unsupported data format.
            utils.raster2array_saga(bad_header + '.sdat')

    def test_sample_data(self):
        elevation = utils.raster2array_saga(self.sample_path('SRTM31_EG_GF_m'))
        self.assertEqual(elevation.shape, (475, 1064))
        self.assertEqual(elevation.dtype, np.float32)
        self.assertAlmostEqual(elevation[elevation != -99999].min(),
                               100.67122, places=4)
        geology = utils.raster2array_saga(self.sample_path('fdt100_14k'))
        self.assertEqual(geology.dtype, np.uint8)
        self.assertEqual(geology.max(), 14)

    def test_matches_the_gdal_reader(self):
        # Compares the two readers on the sample data, including the row
        # order. This is the check that needs a real GDAL installation.
        for name in self.SAMPLE_FILES:
            with self.subTest(name=name):
                path = self.sample_path(name)
                with_gdal = utils.raster2array(path)
                without_gdal = utils.raster2array_saga(path)
                np.testing.assert_array_equal(without_gdal, with_gdal)
                self.assertEqual(without_gdal.dtype, with_gdal.dtype)


if __name__ == '__main__':
    unittest.main()
