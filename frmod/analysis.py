"""
Frequency ratio model analysis.

Perform a landslide susceptibility analysis with the frequency ratio method.

@author: Dávid Gerzsenyi
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import frmod.utils as utils


def get_freq_ratios_classic(vr,
                            mask,
                            binc=20,
                            nodata=-99999.,
                            categorical=False,
                            ls_marker=1,
                            nls_marker=0):
    """
    Get the frequency ratio of the landslide parts and the whole area.

    Parameters
    ----------
    vr : Array
        The array of the analyzed variable. Numeric values.
    mask : Array
        The array of the mask.
    binc : int, optional
        Bin count for the histogram of the non-categorical variables.
        The default is 20.
    nodata : int / float, optional
        The nodata value of the vr grid.
        The default is -99999..
    categorical : bool, optional
        Set true if the analysed variable raster is categorical.
        Categories must be marked with unique integers in the rasters.
        The dafault is False.
    ls_marker : int, optional
        The value marking the landslide parts.
        The default is 1.
    nls_marker : int, optional
        The value marking the non-landslide parts.
        The default is 0.

    Returns
    -------
    frequency_ratios : Array
        The frequency ratio values. Length: number of bins.
    hst_bins : Array
        Array containing the edges of the bins. Length: number of bins + 1.
    fr_stat_df : pandas.DataFrame
        DataFrame with the statistics.
        Columns:
            'min', 'max', 'LS_density', 'NLS_density', 'frequency_ratio'.

    """
    vr = np.ma.masked_where((vr == nodata), vr)
    in_all = np.logical_or(mask == ls_marker, mask == nls_marker)
    ls_area = np.count_nonzero(mask == ls_marker)
    all_area = np.count_nonzero(mask == nls_marker) + ls_area
    if categorical:
        bin_edges = np.unique(vr[in_all])
        bin_edges = np.append(bin_edges, bin_edges[-1] + 1)
        ls_hst = np.histogram(vr[mask == ls_marker],
                              bins=bin_edges,
                              density=False)
        all_hst = np.histogram(vr[in_all],
                               bins=bin_edges,
                               density=False)
    else:
        glob_lim = (vr[in_all].min(),
                    vr[in_all].max())
        ls_hst = np.histogram(vr[mask == ls_marker], bins=binc,
                              range=glob_lim, density=False)
        all_hst = np.histogram(vr[in_all], bins=binc,
                               range=glob_lim, density=False)

    # Histogram density for the landslide part
    ls_hst_d = ls_hst[0] / ls_area
    # Histogram density for the non-landslide part
    all_hst_d = all_hst[0] / all_area
    frequency_ratios = ls_hst_d / all_hst_d

    hst_bins = all_hst[1]
    mn = hst_bins[:-1]
    mx = hst_bins[1:]

    # Create a pd.DataFrame for the bins, densities, and the frequency ratio
    data = [mn, mx, ls_hst_d, all_hst_d, frequency_ratios]
    # columns = ["Min", "Max", "LS_density", "NLS_density", "frequency_ratio"]
    data = {'min': mn,
            'max': mx,
            'LS_density': ls_hst_d,
            'NLS_density': all_hst_d,
            'frequency_ratio': frequency_ratios}
    fr_stat_df = pd.DataFrame(data=data)
    return frequency_ratios, hst_bins, fr_stat_df

def get_freq_ratios(vr,
                    mask,
                    binc=100,
                    nodata=-9999.,
                    categorical=False,
                    normalize=False,
                    ls_marker=1,
                    nls_marker=0):
    """
    Get the frequency ratio of the landslide and non-landslide parts.

    Parameters
    ----------
    vr : Array
        The array of the analyzed variable. Numeric values.
    mask : Array
        The array of the mask.
    binc : int, optional
        Bin count for the histogram of the non-categorical variables.
        The default is 100.
    categorical : bool, optional
        Set true if the analysed variable raster is categorical.
        Categories must be marked with unique integers in the rasters.
        The dafault is False.
    normalize : bool, optional
        Set True for normalized weights (0, 1)
        The default is False.
    ls_marker : int, optional
        The value marking the landslide parts.
        The default is 1.
    nls_marker : int, optional
        The value marking the non-landslide parts.
        The default is 0.

    Returns
    -------
    frequency_ratios : Array
        The frequency ratio values. Length: number of bins.
    hst_bins : Array
        Array containing the edges of the bins. Length: number of bins + 1.
    fr_stat_df : pandas.DataFrame
        DataFrame with the statistics.

    """
    ls_area = np.count_nonzero(mask == ls_marker)
    nls_area = np.count_nonzero(mask == nls_marker)
    if categorical:
        bin_edges = np.unique(vr[(vr != nodata)])
        bin_edges = np.append(bin_edges, bin_edges[-1] + 1)
        ls_hst = np.histogram(vr[mask == ls_marker],
                              bins=bin_edges,
                              density=False)
        nls_hst = np.histogram(vr[mask == nls_marker],
                               bins=bin_edges,
                               density=False)
    else:
        glob_lim = (vr[vr != nodata].min(),
                    vr[vr != nodata].max())
        ls_hst = np.histogram(vr[mask == ls_marker], bins=binc,
                              range=glob_lim, density=False)
        nls_hst = np.histogram(vr[mask == nls_marker], bins=binc,
                               range=glob_lim, density=False)
    # Histogram density for the landslide part
    ls_hst_d = ls_hst[0] / ls_area
    # Histogram density for the non-landslide part
    nls_hst_d = nls_hst[0] / nls_area
    # Histogram bins
    hst_bins = ls_hst[1]
    mn = hst_bins[:-1]
    mx = hst_bins[1:]
    # Frequency ratios
    fr = ls_hst_d / nls_hst_d
    if normalize:
        frequency_ratios = (fr - fr.min()) / (fr.max() - fr.min())
    else:
        frequency_ratios = fr
    # Create a pd.DataFrame for the bins, densities, and the frequency ratio
    data = [mn, mx, ls_hst_d, nls_hst_d, frequency_ratios]
    # columns = ["Min", "Max", "LS_density", "NLS_density", "frequency_ratio"]
    data = {'min': mn,
            'max': mx,
            'LS_density': ls_hst_d,
            'NLS_density': nls_hst_d,
            'frequency_ratio': fr}
    fr_stat_df = pd.DataFrame(data=data)
    return frequency_ratios, hst_bins, fr_stat_df


# TODO: a name change is due to reclass_array
# TODO Use more generic variable names. This can reclass any array.
def reclass_raster(vr, f_ratios, bin_edges, verbose=False):
    """
    Create an array with the frequency ratios.

    Parameters
    ----------
    vr : Array
        Array of the analysed variable to be reclassified.
    f_ratios : Array
        The frequency ratio values.
        Length: number of bins.
    bin_edges : Array
        Array containing the edges of the bins.
        Length: number of bins + 1.
    verbose : bool
        Set True to print the bin ranges and reclass values.

    Returns
    -------
    reclassed : Array
        Reclassified array with the appropriate frequency ratio values.

    """
    reclassed = np.ones(vr.shape) * -99999
    for i in range(0, len(f_ratios)):
        # Reclassifying the raster by assigning the frequency ratio
        # values of each bin to the corresponding raster values.
        mn = bin_edges[:-1][i]
        mx = bin_edges[1:][i]
        vrange = mx - mn
        to_reclass = (vr >= mn) & (vr < mx)
        reclassed[to_reclass] = f_ratios[i]
        if verbose:
            print("Min: {} Max: {} Range: {} Ratio: {}".format(
                mn, mx, vrange, f_ratios[i])
                 )
    return reclassed


def show_grid(grid, nodata, name='Grid', **kwargs):
    """
    Plot a grid, nodata values are masked.

    Plot a grid (2D array). Values equal to nodata will be masked.
    **kwargs are keywords passed to matplotplit.pyplot.imshow that
    is used for plotting the masked grid. Eg. color maps (cmap).

    Parameters
    ----------
    grid : Array
        Grid to plot.
    nodata : int / float
        Nodata value of the grid. Nodata values will be masked.
    name : str, optional
        The title of the plot. The default is 'Grid'.

    Returns
    -------
    None.

    """
    masked_grid = np.ma.masked_where((grid == nodata), grid)
    plt.figure()
    plt.title(name)
    plt.imshow(masked_grid, **kwargs)
    plt.colorbar()
    plt.show()


class VRaster():
    """Variable raster, input for frequency ratio analysis."""

    def __init__(self, name, path, bins=10, categorical=False):
        """
        Create the VRaster object.

        Parameters
        ----------
        name : str
            Name of the VRaster.
        path : str
            Path to the GDAL-compatible raster file.
        bins : int, optional
            Number of histogram bins. The default is 10.
        categorical : bool, optional
            True if it is a categorical variable, eg: geology.
            The default is False.

        Returns
        -------
        None.

        """
        self.name = name
        self.path = path
        self.bins = bins
        self.categorical = categorical
        self.nodata = utils.get_nodata_value(path)
        # Convert input grid to array
        self.grid = utils.raster2array(path)
        # Calculate basic statistics for the grid
        self.min = min(self.grid[self.grid != self.nodata])
        self.max = max(self.grid[self.grid != self.nodata])
        self.limits = (self.min, self.max)

    def show(self, **kwargs):
        """
        Plot the VRaster.grid.

        Parameters
        ----------
        **kwargs :
            Keywords passed to show_grid.

        Returns
        -------
        None.

        """
        show_grid(self.grid, self.nodata, name=self.name, **kwargs)

    def show_info(self):
        """
        Show basic information about the VRaster.grid.

        Returns
        -------
        None.

        """
        valid_values = self.grid[self.grid != self.nodata]
        if self.categorical:
            print("Categorical!")
        else:
            average = np.mean(valid_values)
            sdev = np.std(valid_values)
            print("Name: {} Limits: {}".format(self.name, self.limits))
            print("Mean: {} Standard deviation: {}".format(average, sdev))


class LandslideMask():
    """LandslideMask."""

    def __init__(self, name, path, ls_marker=1, nls_marker=0, fold_count=5):
        """
        Create a LandslideMask object.

        Parameters
        ----------
        name : str
            Name of the landslide mask.
        path : str
            Path to the file used as the mask in the analysis.
        ls_marker : int, optional
            Value marking the landslide pixels.
            The default is 1.
        nls_marker : int, optional
            Value marking the non-landslide pixels.
            Must be different from the nodata value.
            The default is 0.
        fold_count : int, optional
            The number of cross validation folds. The default is 5.

        Returns
        -------
        None.

        """
        # Name of the LandslideMask
        self.name = name
        # Path to the input mask file.
        self.path = path
        # NoData value of the input mask file.
        self.nodata = utils.get_nodata_value(path)
        # 2D array representation of the input mask file.
        self.grid = utils.raster2array(path)
        # Value marking the landslide cells.
        self.ls_marker = ls_marker
        # Value marking the non-landslide cells. Different from nodata!
        self.nls_marker = nls_marker
        # Number of folds
        self.fold_count = fold_count
        # (fold_count no. of (2, n) size arrays) with the split positions
        self.split_locations = self.get_splits(folds=self.fold_count)
        # List of (train_areas, valid_positions) ndarray pairs.
        self.t_v = [self.get_train_area(a) for a in self.split_locations]
        # Training areas (1), validation cells count as non-landslide.
        # Input mask for get_freq_ratios.
        self.train_areas = [t_v[0] for t_v in self.t_v]
        # Positions of the validation cells.
        self.valid_positions = [t_v[1] for t_v in self.t_v]

    def get_splits(self, folds=5):
        """
        Split the mask for cross validation.

        Parameters
        ----------
        folds : TYPE, optional
            Number of desired folds. The default is 5.

        Returns
        -------
        split_locations : list of ndarrays
            (fold_count no. of (2, n) size arrays) with the split positions.

        """
        valid = np.array(np.nonzero(self.grid == self.ls_marker))
        valid_transposed = valid.T
        np.random.shuffle(valid_transposed)
        valid_transposed_split = np.array_split(valid_transposed, folds)
        split_locations = [i.T for i in valid_transposed_split]
        return split_locations

    def get_train_area(self, split_to_omit):
        """
        Get the train_area grid and the validation cell positions.

        Parameters
        ----------
        split_to_omit : list
            List of the position of the validation cells. It is used
            to construct the valid_position array. Cells marked here
            are turned into non-landslide cells.

        Returns
        -------
        train_area : list
            List of train area grids. Similar format to the self.grid.
        valid_position : list
            Lists the positions of the validation cells for the folds.
            Positions are given as two arrays:
                1st: row index increasing from top to bottom.
                2nd: column index increasing from left to right.

        """
        train_area = np.copy(self.grid)
        valid_position = (split_to_omit[0], split_to_omit[1])
        train_area[valid_position] = self.nls_marker
        return train_area, valid_position

    def show(self, cmap='Accent'):
        """
        Plot the LandslideMask.grid.

        Parameters
        ----------
        cmap : str, optional
            Must be the name of a built-in Matplotlib colormap.
            The default is 'Accent'.

        Returns
        -------
        None.

        """
        show_grid(self.grid, self.nodata, name=self.name, cmap=cmap)


class FRAnalysis():
    """Frequency ratio analysis of a LandslideMask and a list of VRasters.

    The steps of the analysis:

    1. run_analysis() is run upon instantiation for each fold of the
        ls_mask and each VRaster. This yields the frequency ratio statistics
        and the reclassified grids.
    2. get_result() Compute the susceptibility grid and other related data.
    3. get_src() Compute the success rate curves.
        The success rate curve is the cumulative frequency distribution
        of the landslide cells in the susceptibility categories.
        Computed for each fold.
    4. get_auc() Compute the area under the success rate curves for scoring.
        Computed for each fold, then averaged for an overall score.
        Smaller values are better.
    """

    def __init__(self, ls_mask, var_list, classic_mode=True):
        """
        Create the FRAnalysis object.

        Create the FRAnalysis object and its instanced variables.
        The last step of the instantiation is creating the rc_folds list.
        The rc_folds holds the reclassified VRaster.grid-s for the folds 
        of the ls_mask for each VRaster in the var_list.

        Parameters
        ----------
        ls_mask : LandslideMask
            The LandslideMask object for the analysis.
        var_list : list
            List of VRaster objects to be analyzed.
        classic_mode : bool
            True: get frequency ratios with get_freq_ratios_classic
            False: get likelihood ratios with get_freq_ratios

        Returns
        -------
        None.

        """
        # Mode
        self.classic_mode=classic_mode
        # Input LandslideMask
        self.ls_mask = ls_mask
        # List of input VRasters
        self.var_list = var_list
        # Number of VRasters
        self.var_count = len(var_list)

        # AUC, area under the success rate curve
        # Sum of the success_rates of the folds
        self.auc_folds = []
        # Mean AUC
        self.auc_mean = None
        # Standard deviations of the AUCs
        self.auc_std = None

        # Percentile bins of the fresult
        self.ranks = None
        # Fold_susceptibility, the susceptibility grids of the folds
        self.fold_susceptibility = None
        self.fold_percentiles = []
        # Final result, the mean estimated susceptibility map over the folds.
        self.fresult = None
        # Final result in the percentile form
        self.percentile_grid = None

        # Susceptibility value bins for the percentiles
        self.valid_perc = []

        # The distributions of % ranks for the validation cells.
        self.v_dist = []
        # Bins for the above v_dist for conversion to percentiles
        self.v_bins = []

        # Frequency ratio analysis results for each VRaster and fold.
        self.stats = {}

        # Frequency ratio analysis results for each VRaster and fold.
        # keyword: VRaster.name
        # value: list of pd.DataFrames, 1 DF / fold
        self.fr_stats_full = {}

        # List of success rates for the folds.
        self.success_rates = None
        # pandas DataFrame of the success rates for the folds.
        self.src_df = None

        # Reclassified grids for each variable and fold
        # Shape: [var_count, ls_mask.fold_count, rows, columns]
        self.rc_folds = [self.run_analysis(vraster, self.ls_mask)
                         for vraster in self.var_list]

    def run_analysis(self, vrr, lsm):
        """
        Frequency ratio analysis of one VRaster and the LandslideMask folds.

        Parameters
        ----------
        vrr : VRaster
            Supplies the grid and other parameters for the
            get_freq_ratios function.
        lsm : LandslideMask
            Supplies the input masks for the get_freq_ratios functions.

        Returns
        -------
        rc_folds : ndarray
            The reclassified grids after the frequency ratio analysis of the
            VRaster and each fold of the LandslideMask.
            Shape: (self.var_count, lsm.fold_count, vrr.shape[0], vrr.shape[1])

        """
        # Array for storing the reclassified VRaster.grids for the folds
        rc_folds = []
        # Create arrays for the statistics
        all_frq_ratios = []
        all_hst_bins = []
        all_folds_statistics = []

        # Run the analysis with the training areas of the different folds
        # Iterating over the train_areas of the folds
        for msk in lsm.train_areas:
            if self.classic_mode:
                fr_data = get_freq_ratios_classic(vr=vrr.grid,
                                                  mask=msk,
                                                  binc=vrr.bins,
                                                  nodata=vrr.nodata,
                                                  categorical=vrr.categorical)
            else:
                fr_data = get_freq_ratios(vr=vrr.grid,
                                          mask=msk,
                                          binc=vrr.bins,
                                          nodata=vrr.nodata,
                                          categorical=vrr.categorical,
                                          normalize=False
                                          )
            frq_ratios = fr_data[0]
            hst_bins = fr_data[1]
            # pd.DataFrame with the densities and the frequency ratios
            all_folds_statistics.append(fr_data[2])
            # Prepare the statistics DataFrame
            all_frq_ratios.append(frq_ratios)
            all_hst_bins.append(hst_bins)
            # Reclassify the VRaster.grid
            reclassed = reclass_raster(vrr.grid, frq_ratios, hst_bins)
            # Append the reclassified VRaster.grid to rc_folds
            rc_folds.append(reclassed)

        # Adding all_folds_statistics DF to the fr_stats_full dict
        self.fr_stats_full[vrr.name] = all_folds_statistics

        # Creating the pd.DataFrame for the frequency ratio statistics
        mn = all_hst_bins[0][:-1]
        mx = all_hst_bins[0][1:]
        stat_df = pd.DataFrame({(vrr.name+'_min'): mn, (vrr.name+'_max'): mx})
        for i in range(0, lsm.fold_count):
            cname = "fold_"+str(i+1)
            stat_df.insert(i+2, cname, all_frq_ratios[i])
        # Add the stats pd.DataFrame to the stats dict
        self.stats[vrr.name] = stat_df
        return rc_folds

    def get_result(self, product=False):
        """
        Get the susceptibility estimation and ranks.
        
        Parameters
        ----------
        product : bool, optional
            If true: the final susceptibility is calculated as the product
            of the susceptibility grids of the folds.
            If false: the final susceptibility is calculated as the average
            of the susceptibility grids of the folds.
            Default is false.

        Returns
        -------
        self.ranks : ndarray
            Numpy array (101,) with the percentile bins.

        """
        rc_folds = self.rc_folds
        valid_positions = self.ls_mask.valid_positions
        result = []
        # TODO remove percentile_result if really not used
        percentile_result = []
        # percentile bins
        percentile_bins = [x * 0.01 for x in range(0, 101)]
        # Iterate over the folds of the ls_mask.
        for i in range(0, self.ls_mask.fold_count):
            if product:
                fold_result = np.ones(rc_folds[0][0].shape)
            else:
                fold_result = np.zeros(rc_folds[0][0].shape)

            # Iterate over the reclassified grids of fold #i
            for j in rc_folds:
                if product:
                    fold_result *= j[i]
                else:
                    # Average the reclassified  grids of fold #i.
                    fold_result += j[i] / self.var_count

            # Percentile bin edges for conversion to percentiles
            valid_perc = np.quantile(fold_result[fold_result >= 0],
                                     percentile_bins,
                                     interpolation='nearest')
# TODO 
# Users of the modes 'nearest', 'lower', 'higher', or 'midpoint'
# are encouraged to review the method they used. (Deprecated NumPy 1.22)

            self.valid_perc.append(valid_perc)
            v_to_score = fold_result[valid_positions[i]]
            v_dist, v_bins = np.histogram(v_to_score,
                                          valid_perc,
                                          density=False)

            v_dist = np.array(v_dist)
            v_dist = v_dist / sum(v_dist)
            # Append the probability density function to vdist
            self.v_dist.append(v_dist)
            self.v_bins.append(v_bins)
            result.append(fold_result)

        # fresult: cell by cell average of the result array
        fresult = sum(result) / self.ls_mask.fold_count

        # Assign the results from the validation folds to the location
        # of the landslide mask and create the percentile grids for the folds
        for i in range(0, self.ls_mask.fold_count):
            fresult[valid_positions[i]] = result[i][valid_positions[i]]
            result[i][result[i] < 0] = -99999
            fp = reclass_raster(result[i],
                                percentile_bins[1:],
                                self.valid_perc[i])
            fp[fp >= 0] = fp[fp >= 0] * 100
            self.fold_percentiles.append(fp)

        # Set all invalid values (<0) to -99999 (nodata)
        fresult[fresult < 0] = -99999

        self.fold_susceptibility = result
        self.fresult = fresult
        self.ranks = np.quantile(fresult[fresult >= 0], percentile_bins)
        return self.ranks

    def get_src(self):
        """
        Get success rate arrays for the folds.

        Returns
        -------
        success_rates : ndarray
            Cumulative sum of the v_dist arrays of the folds.

        """
        success_rates = []
        for i in self.v_dist:
            success_rates.append(np.cumsum(i))
        self.success_rates = success_rates
        # Create the dataframe
        percentiles = [i + 1 for i in range(0, 100)]
        src_df = pd.DataFrame({"percentile": percentiles})
        for i in range(0, self.ls_mask.fold_count):
            cname = "fold_"+str(i+1)
            src_df.insert(i+1, cname, success_rates[i])
        self.src_df = src_df
        return success_rates

    def get_auc(self):
        """
        Get AUC, area under the success rate curve.

        Returns
        -------
        list
            AUC values for the folds.

        """
        self.auc_folds = []
        for i in self.success_rates:
            auc_of_fold = np.sum(i)
            self.auc_folds.append(auc_of_fold)
            print("Auc: {}".format(auc_of_fold))
        self.auc_mean = np.mean(self.auc_folds)
        self.auc_std = np.std(self.auc_folds)
        print("Mean score: {}; Std: {}".format(self.auc_mean, self.auc_std))
        return self.auc_folds

    def get_percentile_grid(self, show=False, **kwargs):
        """
        Reclass the fresult to get its percentile form.

        Parameters
        ----------
        show : bool, optional
            Set True to plot the percentile_grid after computing it.
            The default is False.
        **kwargs :
            Keywords passed to show_grid.

        Returns
        -------
        None.

        """
        percent = [i for i in range(1, 101)]
        percentile_grid = reclass_raster(self.fresult, percent, self.ranks)
        self.percentile_grid = percentile_grid
        if show:
            show_grid(grid=percentile_grid,
                      nodata=-99999,
                      name="Susceptibility (percentiles)",
                      **kwargs)

    def save_src(self, folder="./output/", fname="src.csv"):
        """
        Save the success rates as csv.

        Parameters
        ----------
        folder : str, optional
            Path to the output folder. The default is "./output/".
        fname : str, optional
            Output filename with extension. The default is "src.csv".

        Returns
        -------
        None.

        """
        if os.path.isdir(folder) is False:
            os.makedirs(folder)
        output_path = folder+fname
        self.src_df.to_csv(output_path)

    def save_stats(self, folder="./output/", tag=""):
        """
        Save the frequency ratio statistics for each VRaster and fold.

        Parameters
        ----------
        folder : str, optional
            Path to the output folder. The default is "./output/".
        tag : str, optional
            Tag inserted to the beginning of the file name.
            The default is "".

        Returns
        -------
        None.

        """
        output_path = folder+tag
        # Create the folder directory if it doesn't exist.
        if os.path.isdir(folder) is False:
            os.makedirs(folder)

        for k, v in self.stats.items():
            v.to_csv(output_path+"_{}.csv".format(str(k)))

    def show_results(self, **kwargs):
        """
        Plot fresult with show_grid.

        Plot fresult, the raw estimated landslide susceptibility.
        **kwargs can be keywords passed to matplotlib.pyplot.imshow

        Returns
        -------
        None.

        """
        if self.fresult.any(None):
            show_grid(grid=self.fresult,
                      nodata=-99999,
                      name='Estimated susceptibility',
                      **kwargs)
        else:
            print("Use get_result() first!")

    def plot_success_rates(self):
        """
        Plot the success rate curves.

        The success rate curve shows the cumulative distribution of
        test landslide cells (pixels) in the susceptibility categories.

        Returns
        -------
        fig : figure.Figure
            A figure for the success rate curves.
        ax : matplotlib.axes.__subplots.AxesSubplot
           The axes of the figure.

        """
        fig, ax = plt.subplots()
        label = 1
        plt.title("Success rate curve")
        for i in self.success_rates:
            ax.plot(i, label=label)
            label += 1
        ax.set_xlim(left=0, right=99)
        ax.set_ylim(bottom=0, top=1.0)
        ax.set_xlabel("Susceptibility, percentile")
        ax.set_ylabel("Proportion of validation pixels")
        diag_line, = ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
        ax.legend()
        plt.show()
        return fig, ax

    def plot_var_fold_fr(self, name, fold):
        """
        Plot the densities and the frequency ratio for a fold of one variable.

        The function handles a pd.DataFrame from the fr_stats_full dict.
        The df DataFrame is selected in the following way:
        df = self.fr_stats_full[name][fold]

        Parameters
        ----------
        name : str
            A key in fr_stats_full.
        fold : int
            Index of the fold in the self.fr_stats_full[name] list.

        Returns
        -------
        fig : figure.Figure
            A figure with two subplots on top of each other.
        ax1 : matplotlib.axes.__subplots.AxesSubplot
            1st axes of the figure. Located in the top part.
        ax2 : matplotlib.axes.__subplots.AxesSubplot
            2nd axes of the figure. Located in the bottom part.
        line_LS : matplotlib.lines.Line2D
            The "landslide" line of the 1st axes.
        line_NLS : matplotlib.lines.Line2D
            The "non-landslide" line of the 1st axes.
        line_fr : matplotlib.lines.Line2D
            The frequency ratio curve on the 2nd axes.
            "landslide" line / "non-landslide" line

        """
        df = self.fr_stats_full[name][fold]
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.set_xlabel("Distribution of {} values in the LS and NLS areas".
                       format(name))
        ax2.set_xlabel("{} - fold: {}".format(name, (fold + 1)))
        line_LS, = ax1.plot(df["min"], df["LS_density"], marker="x")
        line_NLS, = ax1.plot(df["min"], df["NLS_density"], marker="x")
        line_fr = ax2.plot(df["min"], df["frequency_ratio"], marker="x")
        ax1.set_ylim(bottom=0)
        ax2.set_ylim(bottom=0)
        ax1.set_xlim(left=df["min"][0])
        ax2.set_xlim(left=df["min"][0])
        ax2.set_ylabel("Frequency ratio")
        plt.tight_layout()
        plt.show()
        # Return the parts of the figure
        return fig, ax1, ax2, line_LS, line_NLS, line_fr
