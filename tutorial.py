"""Tutorial script for frmod."""

import numpy as np
import matplotlib.pyplot as plt

from frmod.analysis import VRaster, LandslideMask, FRAnalysis


if __name__ == "__main__":
    elevation = VRaster(name='elevation',
                        path='./data/SRTM31_EG_GF_m.sdat',
                        bins=15,
                        categorical=False)
    slope = VRaster(name='slope',
                    path='./data/SRTM31_EG_GF_Slope_m.sdat',
                    bins=15,
                    categorical=False)
    geology = VRaster(name='geology_14',
                      path='./data/fdt100_14k.sdat',
                      categorical=True)
    scarps = LandslideMask(name='scarps',
                           path='./data/scarps.sdat',
                           ls_marker=1,
                           fold_count=5)
    fra = FRAnalysis(ls_mask=scarps,
                     var_list=[slope,
                               geology,
                               elevation]
                     )
    result_percentile_bins = fra.get_result()

    # Displaying the results
    plt.figure()
    plt.imshow(fra.fresult, vmin=fra.ranks[0], cmap='viridis')
    plt.colorbar()

    # Computing the success rates
    success_rates = fra.get_src()

    # Plotting the success rate curve
    fra.plot_success_rates()

    auc_folds = fra.get_auc()
    fra.get_percentile_grid(show=True)

    # for i in range(0, fra.ls_mask.fold_count):
    #     fra.plot_var_fold_fr("slope", i)
