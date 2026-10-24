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


def sample_path(name):
    """Path of a sample grid (.sdat) in the data folder."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, '..', 'data', name + '.sdat')


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
        patch = mock.patch.object(
            utils, 'read_raster',
            lambda path, use_gdal=False: (self.files[path][0].copy(),
                                          self.files[path][1]))
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

    def test_categorical_layer_with_nodata_inside_the_study_area(self):
        # Geology classes 1-4. Class 0 is nodata, also in a few cells inside
        # the study area: these must not become a class or break the bins.
        rng = np.random.default_rng(0)
        geology = rng.integers(1, 5, (30, 30)).astype(float)
        geology[20:22, 10:12] = 0
        geology[:, :4] = 0
        mask = np.zeros((30, 30))
        mask[10:15, 10:15] = 1
        mask[:, :4] = NODATA
        self.files['geology'] = (geology, 0.)
        self.files['mask'] = (mask, NODATA)
        for classic_mode in (True, False):
            with self.subTest(classic_mode=classic_mode):
                np.random.seed(1)
                lsm = LandslideMask('mask', 'mask', fold_count=5)
                layer = VRaster('geology', 'geology', categorical=True)
                fra = FRAnalysis(lsm, [layer], classic_mode=classic_mode)
                fra.get_result()
                self.assertEqual(list(fra.stats['geology']['geology_min']),
                                 [1., 2., 3., 4.])
                # The nodata cells have no susceptibility.
                self.assertTrue(np.all(fra.fresult[20:22, 10:12] == NODATA))


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
        elevation = utils.raster2array_saga(sample_path('SRTM31_EG_GF_m'))
        self.assertEqual(elevation.shape, (475, 1064))
        self.assertEqual(elevation.dtype, np.float32)
        self.assertAlmostEqual(elevation[elevation != -99999].min(),
                               100.67122, places=4)
        geology = utils.raster2array_saga(sample_path('fdt100_14k'))
        self.assertEqual(geology.dtype, np.uint8)
        self.assertEqual(geology.max(), 14)

    def test_matches_the_gdal_reader(self):
        # Compares the two readers on the sample data, including the row
        # order. This is the check that needs a real GDAL installation.
        if utils._import_gdal() is None:
            self.skipTest("GDAL is not installed.")
        for name in self.SAMPLE_FILES:
            with self.subTest(name=name):
                path = sample_path(name)
                with_gdal = utils.raster2array(path)
                without_gdal = utils.raster2array_saga(path)
                np.testing.assert_array_equal(without_gdal, with_gdal)
                self.assertEqual(without_gdal.dtype, with_gdal.dtype)


class TestArray2RasterSaga(unittest.TestCase):

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = folder.name
        self.sample = sample_path('scarps')  # 475 rows x 1064 columns.
        self.shape = (475, 1064)

    def out_path(self, name='out'):
        return os.path.join(self.folder, name + '.sdat')

    def test_exporting_a_sample_grid_gives_the_same_files(self):
        # The sample grids were written by SAGA: the data and projection
        # files must be the same, byte by byte.
        for name, nodata in (('scarps', -99999.), ('SRTM31_EG_GF_m', -99999.),
                             ('fdt100_14k', 0.)):
            with self.subTest(name=name):
                sample = sample_path(name)
                new = self.out_path(name)
                utils.array2raster_saga(
                    sample, new, utils.raster2array_saga(sample),
                    nodata=nodata)
                for extension in ('.sdat', '.prj'):
                    with open(sample[:-5] + extension, 'rb') as expected:
                        with open(new[:-5] + extension, 'rb') as written:
                            self.assertEqual(written.read(), expected.read())
                self.assertEqual(utils.get_nodata_value_saga(new), nodata)

    def test_data_type_comes_from_the_array(self):
        for data_format, np_type in utils._SAGA_DATA_TYPES.items():
            with self.subTest(data_format=data_format):
                signed = not np.issubdtype(np_type, np.unsignedinteger)
                values = np.arange(self.shape[0] * self.shape[1]) % 100
                array = (values - (50 if signed else 0)).astype(np_type)
                array = array.reshape(self.shape)
                new = self.out_path()
                utils.array2raster_saga(self.sample, new, array)
                self.assertEqual(
                    utils._read_saga_header(new)['DATAFORMAT'], data_format)
                result = utils.raster2array_saga(new)
                np.testing.assert_array_equal(result, array)
                self.assertEqual(result.dtype, array.dtype)

    def test_header_values(self):
        new = self.out_path('new_grid')
        array = np.zeros(self.shape, dtype=np.float32)
        utils.array2raster_saga(self.sample, new, array, nodata=-1.5)
        sample_header = utils._read_saga_header(self.sample)
        header = utils._read_saga_header(new)
        for key in ('POSITION_XMIN', 'POSITION_YMIN', 'CELLSIZE',
                    'CELLCOUNT_X', 'CELLCOUNT_Y'):
            self.assertEqual(header[key], sample_header[key])
        self.assertEqual(header['NAME'], 'new_grid')
        self.assertEqual(header['TOPTOBOTTOM'], 'FALSE')
        self.assertEqual(utils.get_nodata_value_saga(new), -1.5)
        # The header path of the new grid works as a sample grid too.
        utils.array2raster_saga(new[:-5] + '.sgrd', self.out_path('second'),
                                array)

    def test_bool_array_is_written_as_unsigned_byte(self):
        new = self.out_path()
        array = np.zeros(self.shape, dtype=bool)
        array[0, 0] = True
        utils.array2raster_saga(self.sample, new, array)
        self.assertEqual(
            utils._read_saga_header(new)['DATAFORMAT'], 'BYTE_UNSIGNED')
        np.testing.assert_array_equal(utils.raster2array_saga(new), array)

    def test_wrong_input_is_reported(self):
        good = np.zeros(self.shape, dtype=np.float32)
        new = self.out_path()
        with self.assertRaises(ValueError):  # Wrong shape.
            utils.array2raster_saga(self.sample, new, good[:10])
        with self.assertRaises(ValueError):  # No SAGA type for int64.
            utils.array2raster_saga(self.sample, new, good.astype(np.int64))
        with self.assertRaises(ValueError):  # Not a SAGA output path.
            utils.array2raster_saga(self.sample, new[:-5] + '.tif', good)
        with self.assertRaises(ValueError):  # Not a SAGA sample.
            utils.array2raster_saga('sample.tif', new, good)
        with open(os.path.join(self.folder, 'bad.sgrd'), 'w') as header_file:
            header_file.write('DATAFORMAT\t= FLOAT\n')
        with self.assertRaises(ValueError):  # Header without grid system.
            utils.array2raster_saga(
                os.path.join(self.folder, 'bad.sgrd'), new, good)

    def test_gdal_reads_the_new_grid(self):
        # Needs a real GDAL installation: GDAL must read the same values and
        # the same georeference from the new grid as from the sample grid.
        gdal = utils._import_gdal()
        if gdal is None:
            self.skipTest("GDAL is not installed.")
        new = self.out_path()
        array = np.arange(self.shape[0] * self.shape[1], dtype=np.float32)
        array = array.reshape(self.shape)
        utils.array2raster_saga(self.sample, new, array)
        np.testing.assert_array_equal(utils.raster2array(new), array)
        self.assertEqual(gdal.Open(new).GetGeoTransform(),
                         gdal.Open(self.sample).GetGeoTransform())
        self.assertEqual(utils.get_nodata_value(new), -99999.)


class TestUseGdalOption(unittest.TestCase):
    """VRaster and LandslideMask read SAGA grids by default, GDAL on request.

    GDAL is not needed here: its readers are replaced by the SAGA readers,
    wrapped in mocks that remember the calls.
    """

    def setUp(self):
        self.path = sample_path('scarps')
        self.gdal_reader = mock.patch.object(
            utils, 'raster2array', wraps=utils.raster2array_saga).start()
        self.gdal_nodata = mock.patch.object(
            utils, 'get_nodata_value',
            wraps=utils.get_nodata_value_saga).start()
        self.saga_reader = mock.patch.object(
            utils, 'raster2array_saga', wraps=utils.raster2array_saga).start()
        self.saga_nodata = mock.patch.object(
            utils, 'get_nodata_value_saga',
            wraps=utils.get_nodata_value_saga).start()
        self.addCleanup(mock.patch.stopall)

    def create(self, use_gdal):
        """Create a VRaster and a LandslideMask of the same file."""
        kwargs = {} if use_gdal is None else {'use_gdal': use_gdal}
        return (VRaster('layer', self.path, **kwargs),
                LandslideMask('mask', self.path, **kwargs))

    def test_saga_grids_are_read_without_gdal_by_default(self):
        for use_gdal in (None, False):
            with self.subTest(use_gdal=use_gdal):
                for mocked in (self.gdal_reader, self.gdal_nodata,
                               self.saga_reader, self.saga_nodata):
                    mocked.reset_mock()
                layer, mask = self.create(use_gdal)
                self.assertFalse(layer.use_gdal)
                self.assertFalse(mask.use_gdal)
                self.assertEqual(self.saga_reader.call_count, 2)
                self.assertEqual(self.saga_nodata.call_count, 2)
                self.gdal_reader.assert_not_called()
                self.gdal_nodata.assert_not_called()
                self.assertEqual(layer.grid.shape, (475, 1064))
                self.assertEqual(layer.nodata, -99999.)

    def test_gdal_readers_are_used_on_request(self):
        layer, mask = self.create(True)
        self.assertTrue(layer.use_gdal)
        self.assertTrue(mask.use_gdal)
        self.assertEqual(self.gdal_reader.call_count, 2)
        self.assertEqual(self.gdal_nodata.call_count, 2)
        self.saga_reader.assert_not_called()
        self.saga_nodata.assert_not_called()
        for item in (layer, mask):
            np.testing.assert_array_equal(
                item.grid, utils.raster2array_saga(self.path))
            self.assertEqual(item.nodata, -99999.)

    def test_use_gdal_without_gdal_is_reported(self):
        with mock.patch.object(utils, '_import_gdal', return_value=None):
            # The GDAL readers are the real ones again.
            mock.patch.stopall()
            with self.assertRaises(ImportError):
                VRaster('layer', self.path, use_gdal=True)
            with self.assertRaises(ImportError):
                LandslideMask('mask', self.path, use_gdal=True)
            with self.assertRaises(ImportError):
                utils.read_raster(self.path, use_gdal=True)

    def test_gdal_gives_the_same_grid(self):
        # Needs a real GDAL installation.
        mock.patch.stopall()
        if utils._import_gdal() is None:
            self.skipTest("GDAL is not installed.")
        saga_layer, saga_mask = self.create(False)
        gdal_layer, gdal_mask = self.create(True)
        for saga, gdal in ((saga_layer, gdal_layer), (saga_mask, gdal_mask)):
            np.testing.assert_array_equal(gdal.grid, saga.grid)
            self.assertEqual(gdal.nodata, saga.nodata)


class TestWithoutGdal(unittest.TestCase):
    """SAGA grids can be used when GDAL is not installed."""

    def setUp(self):
        patch = mock.patch.object(utils, '_import_gdal', return_value=None)
        patch.start()
        self.addCleanup(patch.stop)

    def test_saga_grids_are_read_by_default(self):
        path = sample_path('scarps')
        grid, nodata = utils.read_raster(path)
        np.testing.assert_array_equal(grid, utils.raster2array_saga(path))
        self.assertEqual(grid.shape, (475, 1064))
        self.assertEqual(nodata, -99999.)

    def test_nodata_value_is_read_from_the_header(self):
        for name, nodata in (('scarps', -99999.), ('SRTM31_EG_GF_m', -99999.),
                             ('fdt100_14k', 0.)):
            with self.subTest(name=name):
                path = sample_path(name)
                self.assertEqual(utils.get_nodata_value_saga(path), nodata)
                # The header path works too.
                self.assertEqual(
                    utils.get_nodata_value_saga(path[:-5] + '.sgrd'), nodata)

    def test_header_without_nodata_value_gives_none(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'grid.sgrd')
            with open(path, 'w') as header_file:
                header_file.write('DATAFORMAT\t= FLOAT\n')
            self.assertIsNone(utils.get_nodata_value_saga(path))

    def test_the_gdal_functions_need_gdal(self):
        # Also for SAGA grids: the SAGA functions are used for them.
        path = sample_path('scarps')
        with self.assertRaises(ImportError):
            utils.raster2array(path)
        with self.assertRaises(ImportError):
            utils.get_nodata_value(path)
        with self.assertRaises(ImportError):
            utils.array2raster(path, 'out.tif', np.zeros((2, 2)))

    def test_results_can_be_exported_and_read_back(self):
        np.random.seed(2021)
        slope = VRaster('slope', sample_path('SRTM31_EG_GF_Slope_m'), bins=20)
        scarps = LandslideMask('scarps', sample_path('scarps'),
                               ls_marker=1, fold_count=5)
        fra = FRAnalysis(scarps, [slope])
        fra.get_result()
        fra.get_percentile_grid()
        with tempfile.TemporaryDirectory() as folder:
            new = os.path.join(folder, 'percentiles.sdat')
            utils.array2raster_saga(sample_path('scarps'), new,
                                    fra.percentile_grid)
            np.testing.assert_array_equal(utils.raster2array_saga(new),
                                          fra.percentile_grid)
            self.assertEqual(utils.get_nodata_value_saga(new), -99999.)

    def test_analysis_runs_on_the_sample_data(self):
        # The demo analysis of the notebook: mean AUC 19.0679 with this seed.
        np.random.seed(2021)
        elevation = VRaster('elevation', sample_path('SRTM31_EG_GF_m'),
                            bins=50)
        slope = VRaster('slope', sample_path('SRTM31_EG_GF_Slope_m'), bins=20)
        geology = VRaster('geology_14', sample_path('fdt100_14k'),
                          categorical=True)
        scarps = LandslideMask('scarps', sample_path('scarps'),
                               ls_marker=1, fold_count=5)
        fra = FRAnalysis(scarps, [slope, geology, elevation])
        fra.get_result()
        fra.get_src()
        with mock.patch('builtins.print'):  # get_auc prints the scores.
            fra.get_auc()
        self.assertAlmostEqual(fra.auc_mean, 19.0679, places=3)


if __name__ == '__main__':
    unittest.main()
