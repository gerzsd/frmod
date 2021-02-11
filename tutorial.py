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

    # Preparing the dataframes for displaying the statistics
    slope_df1 = fra.fr_stats_full["slope"][0]
    slope_df2 = fra.fr_stats_full["slope"][1]
    # Plotting the statistics for the slope variable
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.set_xlabel("Distribution of slope values in the LS and NLS areas")
    ax2.set_xlabel("Frequency ratio [LS / NLS]")
    line_LS, = ax1.plot(slope_df1["min"], slope_df1["LS_density"])
    line_NLS, = ax1.plot(slope_df1["min"], slope_df1["NLS_density"])
    line_fr = ax2.plot(slope_df1["min"], slope_df1["frequency_ratio"])
    plt.tight_layout()
    plt.show()

    auc_folds = fra.get_auc()
