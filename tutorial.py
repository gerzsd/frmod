"""Tutorial script for frmod.

You can use this script to perform a frequency ratio analysis on a set
of raster grids. The results are used to estimate susceptibility to
landslides based on the analyzed conditions. The script does the analysis
with cross validation and gives metrics about the quality of the predictions.
"""

import numpy as np
import matplotlib.pyplot as plt

from frmod.analysis import VRaster, LandslideMask, FRAnalysis, show_grid

if __name__ == "__main__":
    np.random.seed(2021)
    elevation = VRaster(name='elevation',
                        path='./data/SRTM31_EG_GF_m.sdat',
                        bins=50,
                        categorical=False)
    slope = VRaster(name='slope',
                    path='./data/SRTM31_EG_GF_Slope_m.sdat',
                    bins=20,
                    categorical=False)
    geology = VRaster(name='geology_14',
                      path='./data/fdt100_14k.sdat',
                      categorical=True)
    scarps = LandslideMask(name='scarps',
                           path='./data/scarps.sdat',
                           ls_marker=1,
                           fold_count=5)
    fra = FRAnalysis(ls_mask=scarps,
                     var_list=[
                         slope,
                         geology,
                         elevation
                        ],
                     classic_mode=True
                     )

    fra.get_result()

    # Display the results
    fra.show_results(cmap='coolwarm')

    # Compute the success rates
    success_rates = fra.get_src()

    # Plot the success rate curve
    fra.plot_success_rates()

    auc_folds = fra.get_auc()
    fra.get_percentile_grid(show=True, cmap='coolwarm')

    # Plot the frequency ratio statistics for the 1st slope fold
    slope_1_fig = fra.plot_var_fold_fr("slope", 0)
    # Plot the result of the 5th fold
    show_grid(fra.fold_percentiles[0], -99999)
