"""Tutorial script for frmod.

You can use this script to perform a frequency ratio analysis on a set
of raster grids. The results are used to estimate susceptibility to
landslides based on the analyzed conditions. The script does the analysis
with cross validation and gives metrics about the quality of the predictions.
"""

import numpy as np
import matplotlib.pyplot as plt

from frmod.analysis import VRaster, LandslideMask, FRAnalysis


if __name__ == "__main__":
    np.random.seed(42)
    ELEVATION = VRaster(name='elevation',
                        path='./data/SRTM31_EG_GF_m.sdat',
                        bins=15,
                        categorical=False)
    SLOPE = VRaster(name='slope',
                    path='./data/SRTM31_EG_GF_Slope_m.sdat',
                    bins=15,
                    categorical=False)
    GEOLOGY = VRaster(name='geology_14',
                      path='./data/fdt100_14k.sdat',
                      categorical=True)
    SCARPS = LandslideMask(name='scarps',
                           path='./data/scarps.sdat',
                           ls_marker=1,
                           fold_count=5)
    FRA = FRAnalysis(ls_mask=SCARPS,
                     var_list=[SLOPE,
                               GEOLOGY,
                               ELEVATION]
                     )

    FRA.get_result()

    # Display the results
    FRA.show_results(cmap='coolwarm')

    # Compute the success rates
    success_rates = FRA.get_src()

    # Plot the success rate curve
    FRA.plot_success_rates()

    auc_folds = FRA.get_auc()
    FRA.get_percentile_grid(show=True, cmap='coolwarm')

    # Plot the frequency ratio statistics for the 1st slope fold
    slope_1_fig = FRA.plot_var_fold_fr("slope", 0)
