"""Regression tests for frmod.analysis.

Run them with:  python -m unittest discover tests   (or with pytest)

The tests use small synthetic grids instead of raster files: the functions
that read files in frmod.utils are replaced in setUp.
"""

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


if __name__ == '__main__':
    unittest.main()
