"""Tutorial script for frmod."""

import numpy as np
import matplotlib.pyplot as plt

from frmod.analysis import VRaster, LandslideMask, FRAnalysis


if __name__ == "__main__":
    slope = VRaster(name='slope',
                    path='./data/SRTM31_EG_GF_Slope_m.sdat',
                    bins=30,
                    categorical=False)
    geology = VRaster(name='geology_14',
                      path='./data/fdt100_14k.sdat',
                      categorical=True)
    scarps = LandslideMask(name='scarps',
                           path='./data/scarps.sdat',
                           ls_marker=1,
                           fold_count=2)
    fra = FRAnalysis(ls_mask=scarps,
                     var_list=[slope,
                               geology]
                     )
    result_percentile_bins = fra.get_result()

    # Displaying the results
    plt.figure()
    plt.imshow(fra.fresult, vmin=fra.ranks[0], cmap='viridis')
    plt.colorbar()

    # Plotting the success rate curve
    success_rates = fra.get_src()
    fra.plot_success_rates()
