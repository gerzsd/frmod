import os
import pytest

from frmod.utils import get_output_path
from frmod.utils import get_nodata_value


def test_get_nodata_value():
    fname = os.path.join(os.path.dirname(__file__), '..', 'data', 'scarps.sdat')
    assert get_nodata_value(fname) == -99999.000000


def test_get_output_path():
    fname = 'slope.sdat'
    assert get_output_path(fname, '_reclass') == 'slope_reclass.sdat'
