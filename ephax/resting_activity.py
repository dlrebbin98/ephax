import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import numpy as np
import pandas as pd
import copy
import re
import os
import imageio
import scipy
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize, LogNorm, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable, get_cmap
from matplotlib.patches import Patch
from ipywidgets import interact, widgets
from scipy.stats import powerlaw, gaussian_kde, binned_statistic, sem, binned_statistic_2d, linregress, norm, f_oneway, chi2, tukey_hsd, lognorm, kstest, shapiro
from scipy.ndimage import maximum_filter, label
from scipy.spatial.distance import cdist
from scipy.signal import find_peaks
from scipy.interpolate import griddata
from scipy.fftpack import dct, idct
from tqdm import tqdm
from joblib import Parallel, delayed
from select_electrodes import assign_n_proximate_electrodes, plot_assigned_layout
from helper_functions import load_spikes, get_activity_sorted_electrodes, assign_r_distance, assign_r_distance_all, assign_r_theta_distance, load_spikes_data, load_spikes_npz, likelihood_ratio_test, log_likelihood
from r_function import correlation_function, correlation_function_w_sum, get_summed_hz_distribution
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from spikes_pooled import plot_layout_grid_with_fr_pooled
from spikes import calculate_ifr
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore", message="Intel MKL WARNING")
warnings.filterwarnings("ignore", message="RuntimeWarning: overflow encountered")

# New modularized helpers
from ra.viz import truncate_colormap as _truncate_colormap, plot_cofiring_heatmap as _plot_cofiring_heatmap
from ra.compute import (
    ifr_peaks as _ifr_peaks,
    fit_ifr_gmm as _fit_ifr_gmm,
    cofiring_proportions as _cofiring_proportions,
    cofiring_vs_distance_by_delay as _cofiring_vs_distance_by_delay,
    aggregate_cofiring_heatmap as _aggregate_cofiring_heatmap,
)

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """Compatibility wrapper now delegating to ra.viz.truncate_colormap."""
    return _truncate_colormap(cmap, minval=minval, maxval=maxval, n=n)



def compare_model_fits(residuals_dict):
    """
    Perform statistical tests to compare models.
    Parameters:
    - residuals_dict: Dictionary containing residuals for each model.
    """
    # Extract residuals and corresponding group labels
    residuals_combined = []
    group_labels = []
    
    for label, residuals in residuals_dict.items():
        # Flatten the residuals and add to the combined list
        residuals_combined.extend(residuals)
        # Extend the group labels to match the number of residuals
        group_labels.extend([label] * len(residuals))
    
    # Convert to numpy arrays
    residuals_combined = np.array(residuals_combined)
    group_labels = np.array(group_labels)
    
    # Ensure all values are finite
    valid_indices = np.isfinite(residuals_combined)
    residuals_combined = residuals_combined[valid_indices]
    group_labels = group_labels[valid_indices]

    # ANOVA Test
    try:
        # Group the residuals by their label for ANOVA
        groups = [residuals_combined[group_labels == label] for label in residuals_dict.keys()]
        f_stat, p_value = f_oneway(*groups)
        print(f"ANOVA F-statistic: {f_stat}, p-value: {p_value}")
    except Exception as e:
        print(f"Error during ANOVA test: {e}")
        return

    # Post-hoc Tukey's HSD if ANOVA is significant
    if p_value < 0.05:
        try:
            # Convert group labels to categorical codes to work properly with Tukey's HSD
            group_labels_numeric = pd.Categorical(group_labels).codes

            # Perform Tukey's HSD test
            tukey_result = tukey_hsd(residuals_combined, group_labels_numeric)
            print("Tukey's HSD Results:")
            print(tukey_result)
        except Exception as e:
            print(f"Error running Tukey's HSD: {e}")


def plot_activity_rank(spikes_data, layout, top_n=100):
    # Count the number of spikes per electrode
    spike_counts = {electrode: 0 for electrode in layout['electrode']}
    
    for channel in spikes_data['channel']:
        # Find the electrode corresponding to the current channel
        matching_indices = np.where(layout['channel'] == channel)[0]
        
        if len(matching_indices) > 0:
            electrode = layout['electrode'][matching_indices[0]]
            spike_counts[electrode] += 1

    # Sort electrodes by spike count
    sorted_electrodes = sorted(spike_counts.items(), key=lambda item: item[1], reverse=True)

    # Return the top N electrodes
    sorted_counts = [count for electrode, count in sorted_electrodes[:top_n]]
    
    plt.plot(range(top_n), sorted_counts)
    plt.yscale('log')
    plt.title('Ranked Firing Counts')
    plt.xlabel('Rank')
    plt.ylabel('AP Count')
    plt.show()

def calculate_cofiring_proportions(spikes_df, stim_times, window_size=0.001, delay=0.0, ref_electrode=None):
    """Compatibility wrapper delegating to ra.compute.cofiring_proportions."""
    return _cofiring_proportions(spikes_df, stim_times, window_size=window_size, delay=delay, ref_electrode=ref_electrode)

def compute_spike_distance_cofiring(spikes_data, layout, ref_electrode, start_time, end_time, window_size, delays):
    """Compatibility wrapper delegating to ra.compute.cofiring_vs_distance_by_delay."""
    return _cofiring_vs_distance_by_delay(spikes_data, layout, ref_electrode, start_time, end_time, window_size, delays)

def compute_spike_distance_hz(spikes_data, layout, ref_electrode, start_time, end_time):
    # Convert to DataFrame
    spikes_df = pd.DataFrame(spikes_data)
    layout_df = pd.DataFrame(layout)

    # Assign radial distances
    spikes_df, layout_df = assign_r_distance(spikes_df, layout_df, ref_electrode)

    # Filter spikes within the time window
    spikes_df_during = spikes_df[(spikes_df['time'] >= start_time) & (spikes_df['time'] <= end_time)]
    
    # Calculate the firing times of the reference electrode
    firing_times = spikes_df_during[spikes_df_during['electrode'] == ref_electrode]['time']
    print(f"Electrode {ref_electrode} fired {len(firing_times)} times in total.")

    # Calculate duration
    duration = end_time - start_time

    # Group by electrode and count the number of spikes, excluding the reference electrode
    electrode_counts = spikes_df_during[spikes_df_during['electrode'] != ref_electrode].groupby('electrode').size()

    # Calculate firing rates (Hz) and convert to np.array
    avg_hz = np.array(electrode_counts / duration)

    # Get distances for each electrode in the same order as the avg_hz array
    relevant_electrodes = electrode_counts.index
    electrode_distances = np.array(layout_df.set_index('electrode').loc[relevant_electrodes, 'distance'])

    return avg_hz, electrode_distances


def normalize_against_t0(heatmap_data, delays):
    """Deprecated: normalization now handled inside ra.viz.plot_cofiring_heatmap."""
    t0_index = np.where(delays == 0)[0]
    if len(t0_index) == 0:
        return heatmap_data
    t0_index = t0_index[0]
    base = heatmap_data[t0_index, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.divide(heatmap_data, base, out=np.zeros_like(heatmap_data), where=base != 0)

def process_electrode(spikes_data_list, layout_list, ref_electrode, start_times, end_times, window_size, delays):
    """Deprecated shim: returns (Z, distance_bins) for a single reference electrode."""
    heatmap = _aggregate_cofiring_heatmap(
        spikes_data_list, layout_list, [ref_electrode], start_times, end_times, window_size=window_size, delays=delays
    )
    return heatmap.Z, heatmap.distance_bins

def plot_avg_cofiring_heatmap(spikes_data_list, layout_list, ref_electrodes, start_times, end_times, window_size=20, delays=np.linspace(-20, 20, 21), normalize=False):
    """Compute and plot averaged co-firing heatmap (delegates to ra)."""
    heatmap = _aggregate_cofiring_heatmap(
        spikes_data_list, layout_list, ref_electrodes, start_times, end_times, window_size=window_size, delays=delays
    )
    _plot_cofiring_heatmap(heatmap, normalize=normalize, show=True)


######################### AGGREGATED HISTOGRAM OF IFR VALUES #########################

def get_ifr_peaks(spikes_data_list, start_times, end_times, max_electrodes=210, log_scale=True):
    """Compatibility wrapper returning the old tuple from ra.compute.ifr_peaks."""
    peaks = _ifr_peaks(spikes_data_list, start_times, end_times, max_electrodes=max_electrodes, log_scale=log_scale)
    all_ifr_values = peaks.values
    x_grid = peaks.kde_x
    kde_values = peaks.kde_y
    peak_positions = peaks.peaks_x
    peak_densities = peaks.peaks_y
    hz_list = list(peaks.peaks_hz)
    return all_ifr_values, x_grid, kde_values, peak_positions, peak_densities, hz_list

def fit_gaussians_to_ifr_data(all_ifr_values, log_scale=True, n_components=None, plot=True):
    """Compatibility wrapper delegating to ra.compute.fit_ifr_gmm (no plotting)."""
    fit = _fit_ifr_gmm(all_ifr_values, log_scale=log_scale, n_components=n_components)
    return fit.means_hz if log_scale else fit.means_hz, fit.std, fit.weights

def get_ifr_stats(
    spikes_data_list, layout_list, ref_electrodes, start_times, end_times, 
    log=False, v_eph=0.1, v_ax=0.45, lambda_eph=100000, std=0.15
):
    # Collect results in parallel
    results = Parallel(n_jobs=-1)(
        delayed(calculate_avg_firing_rate_distance)(
            spikes_data, layout, start_time, end_time
        )
        for spikes_data, layout, start_time, end_time in zip(spikes_data_list, layout_list, start_times, end_times)
    )

    # Concatenate results from all reference electrodes
    all_avg_firing_rates = np.concatenate([result[0] for result in results])
    all_distances = np.concatenate([result[1] for result in results])
    max_distance = 3500

    # Calculate distance bins based on the actual data
    if log:
        distance_bins = np.logspace(
            np.log10(min(all_distances)), np.log10(max_distance), num=38
        )
    else:
        distance_bins = np.linspace(min(all_distances), max_distance, num=38)

    # Bin data by distances and calculate mean and standard error
    bin_means, bin_edges, _ = binned_statistic(
        all_distances, all_avg_firing_rates, statistic=np.nanmean, bins=distance_bins
    )
    bin_std_err, _, _ = binned_statistic(
        all_distances, all_avg_firing_rates,
        statistic=lambda x: np.std(x) / np.sqrt(len(x)), bins=distance_bins
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Remove bins with NaN values
    valid_bins = ~np.isnan(bin_means)
    bin_centers = bin_centers[valid_bins]
    bin_means = bin_means[valid_bins]
    bin_std_err = bin_std_err[valid_bins]

    # Get hz_list
    all_ifr_values = np.array(get_ifr_peaks(spikes_data_list, start_times, end_times)[0])
    means, std_devs, weights = fit_gaussians_to_ifr_data(all_ifr_values)
    means = np.array(means)
    weights = np.array(weights)
    weights = weights[(means > 30) & (means < 1000)]
    gamma_hz_list = np.sort(means[(means > 30) & (means < 1000)])
    weights = weights / weights.min()
    
    print(f'Included IFR Peaks: {gamma_hz_list}')

######################### P(COFIRING) W/O TEMPORAL DIM #########################

def calculate_pcofiring_distance(spikes_data, layout, start_time, end_time, window_size, max_electrodes=1000, delay=-10):
    cofiring_proportions = []
    electrode_distances = []
    ref_electrodes = get_activity_sorted_electrodes([spikes_data], start=10, stop=max_electrodes, start_time=0, end_time=600)

    spikes_df = pd.DataFrame(spikes_data)
    layout_df = pd.DataFrame(layout)

    spikes_df, layout_df = assign_r_distance_all(spikes_df, layout_df, ref_electrodes)

    delay_sec = delay / 1000

    for ref_electrode in ref_electrodes:
        # Filter spikes within the desired time window
        spikes_df_during = spikes_df[(spikes_df['time'] >= start_time) & (spikes_df['time'] <= end_time)].copy()

        # Get firing times for the reference electrode
        firing_times = spikes_df_during[spikes_df_during['electrode'] == ref_electrode]['time'].reset_index(drop=True)

        # Calculate instantaneous firing rates (IFR)
        if len(firing_times) > 1:
            # Calculate the time intervals and the IFRs (in Hz)
            intervals = firing_times.diff().dropna()
            IFRs = 1 / intervals

            # Add IFRs as a new column in the spikes DataFrame
            spikes_df_during.loc[spikes_df_during['electrode'] == ref_electrode, 'IFR'] = IFRs

            # Filter based on IFR > 20 Hz (ignoring the first spike which has no IFR)
            spikes_df_during = spikes_df_during[(spikes_df_during['electrode'] == ref_electrode)]

            # Use the filtered firing times for the next steps
            filtered_firing_times = spikes_df_during['time']

        else:
            filtered_firing_times = firing_times  # No filtering needed if there is only one spike

        # Use filtered firing times to calculate co-firing proportions
        proportions = calculate_cofiring_proportions(
            spikes_df_during, 
            filtered_firing_times, 
            window_size=window_size, 
            delay=delay_sec, 
            ref_electrode=ref_electrode
        )
        
        # Collect co-firing proportions and distances
        for electrode, proportion in proportions.items():
            if electrode != ref_electrode:
                cofiring_proportions.append(proportion)
                electrode_distances.append(layout_df.loc[layout_df['electrode'] == electrode, 'distance'].values[0])

    # Convert to numpy arrays
    cofiring_proportions = np.array(cofiring_proportions)
    electrode_distances = np.array(electrode_distances)

    return cofiring_proportions, electrode_distances


def calculate_bic(n, residuals, k):
    """
    Calculate the Bayesian Information Criterion (BIC).
    Parameters:
    - n: number of data points
    - residuals: array of residuals from the model
    - k: number of parameters in the model
    Returns:
    - BIC value
    """
    rss = np.sum(residuals ** 2)  # Residual sum of squares
    bic = n * np.log(rss / n) + k * np.log(n)
    return bic

def plot_avg_cofiring_models(
    spikes_data_list, layout_list, start_times, end_times, max_electrodes,
    window_size=0.004, delay=-20, log=False, v_eph=0.1, v_ax=0.45, lambda_eph=100000, std=0.15, plot_peaks_only=True
):
    # Collect results in parallel
    results = Parallel(n_jobs=-1)(
        delayed(calculate_pcofiring_distance)(
            spikes_data, layout,
            start_time, end_time, window_size, max_electrodes, delay
        )
        for spikes_data, layout, start_time, end_time in zip(spikes_data_list, layout_list, start_times, end_times)
    )

    # Concatenate results from all reference electrodes
    all_cofiring_proportions = np.concatenate([result[0] for result in results])
    all_distances = np.concatenate([result[1] for result in results])
    print(len(all_distances))
    if len(all_distances) == 0:
        print("No co-firing distances found; skipping model fitting.")
        return
    min_distance = 50
    max_distance = 3500

    # Calculate distance bins based on the actual data
    if log:
        distance_bins = np.logspace(
            np.log10(min_distance), np.log10(max_distance), num=50
        )
    else:
        distance_bins = np.linspace(min_distance, max_distance, num=50)

    # Bin data by distances and calculate mean and standard error
    bin_means, bin_edges, _ = binned_statistic(
        all_distances, all_cofiring_proportions, statistic=np.nanmean, bins=distance_bins
    )
    bin_std_err, _, _ = binned_statistic(
        all_distances,
        all_cofiring_proportions,
        statistic=lambda x: np.std(x) / np.sqrt(len(x)),
        bins=distance_bins
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Remove bins with NaN values
    valid_bins = ~np.isnan(bin_means)
    bin_centers = bin_centers[valid_bins]
    bin_means = bin_means[valid_bins]
    bin_std_err = bin_std_err[valid_bins]

    # Get hz_list
    all_ifr_values = np.array(get_ifr_peaks(spikes_data_list, start_times, end_times)[0])
    means, std_devs, weights = fit_gaussians_to_ifr_data(all_ifr_values)
    means = np.array(means)
    weights = np.array(weights)
    gamma_hz_list = means[(means>20)&(means<1000)]
    weights = weights[(means>20)&(means<1000)]
    weights = weights / weights.min()
    
    print(f'Included IFR Peaks: {gamma_hz_list}')

    # Ensure velocities are in micrometers per second
    v_eph_um_s = v_eph * 1e6  # Convert from m/s to μm/s
    v_ax_um_s = v_ax * 1e6 
    std_um_s = std * 1e6   # Convert from m/s to μm/s

    # Create a fine-grained distance array for the correlation function
    r_um = np.linspace(min_distance, max_distance, 1000)  # Smooth curve

    # Create and plot the Full Model:
    total_r_full = np.zeros_like(r_um)
    for hz, w in zip(gamma_hz_list, weights):
        total_r_full += correlation_function(r_um, hz, v_eph_um_s, v_ax_um_s, lambda_eph) * w

    # Fit and assess the full and reduced models
    X = r_um.reshape(-1, 1)
    y = np.interp(r_um, bin_centers, bin_means)

    # Compute residuals and log-likelihood for the full model
    full_residuals = y - total_r_full
    logL_full = log_likelihood(full_residuals, len(gamma_hz_list))

    # Prepare to store the results of each reduced model and their BIC & MI
    reduced_models = []
    upper_cis = []
    lower_cis = []
    bic_values = []
    mi_values = []
    lrt_values = []
    residuals_dict = {}
    current_gamma_list = list(gamma_hz_list)

    while len(current_gamma_list) > 0:
        # Create the correlation curve using the current list of frequencies
        total_r_reduced = np.zeros_like(r_um)
        steps = 50
        total_r_reduced_list = [np.zeros_like(r_um) for _ in range(50)]
        current_weights = weights[:len(current_gamma_list)]
        
        for hz, w in zip(current_gamma_list, current_weights):
            total_r_reduced += correlation_function(r_um, hz, v_eph_um_s, v_ax_um_s, lambda_eph) * w
            for idx, v in enumerate(np.linspace(v_ax_um_s - std_um_s, v_ax_um_s + std_um_s, steps)):
                total_r_reduced_list[idx] += correlation_function(r_um, hz, v_eph_um_s, v, lambda_eph) * w
        
        # Compute residuals and log-likelihood for reduced model
        reduced_residuals = y - total_r_reduced
        logL_reduced = log_likelihood(reduced_residuals, len(current_gamma_list))
        # Likelihood Ratio Test
        df = len(gamma_hz_list) - len(current_gamma_list)  # Degrees of freedom difference
        LRT_stat, p_value = likelihood_ratio_test(logL_full, logL_reduced, df)
        lrt_values.append((LRT_stat, p_value))

        # Compute residuals and BIC
        residuals = y - total_r_reduced
        print(len(residuals[np.isfinite(residuals)]))
        bic = calculate_bic(len(y), residuals, len(current_gamma_list))
        bic_values.append(bic)
        
        # Compute MI
        mi = mutual_info_regression(total_r_reduced.reshape(-1, 1), y)
        mi_values.append(mi[0])
        
        # Save the model result for plotting later
        reduced_models.append((current_gamma_list.copy(), total_r_reduced))
        stacked_array = np.stack(total_r_reduced_list) 
        upper_ci = np.max(stacked_array, axis=0)
        stacked_array = np.stack(total_r_reduced_list) 
        lower_ci = np.min(stacked_array, axis=0)
        upper_cis.append(upper_ci)
        lower_cis.append(lower_ci)
        residuals_dict[f'Reduced Model {len(current_gamma_list)} Hz'] = np.array(residuals.tolist(), dtype = float)
        
        # Remove the highest frequency from the list
        #current_gamma_list = current_gamma_list[:-1]
        current_gamma_list = []

    # Create Uniform Model (spatially unbiased)
    uniform_model = np.full_like(r_um, np.mean(all_cofiring_proportions))

    # Calculate BIC & MI for the uniform model
    residuals_uniform = y - uniform_model
    uniform_bic = calculate_bic(len(y), residuals_uniform, 1)
    uniform_mi = mutual_info_regression(uniform_model.reshape(-1, 1), y)[0]

    # Plot the data
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot the error bars on the primary y-axis
    # Plot the error bars on the primary y-axis
    ax1.plot(bin_centers, bin_means, color='blue', label='Mean Co-Firing Rate')
    ax1.fill_between(
        bin_centers,
        bin_means - bin_std_err,
        bin_means + bin_std_err,
        alpha=0.4,
        color='blue'
    )
    ax1.set_xlabel('Distance from Electrode ($\mu m$)')
    ax1.set_ylabel('Mean Co-Firing Probability')
    ax1.set_title('Averaged Co-Firing Probability vs. Distance')

    # Get the y-axis limits to set the extent of the shading
    ymin, ymax = ax1.get_ylim()

    # Create a colormap and normalizer for frequencies
    cmap = plt.cm.viridis
    # Truncate the colormap to use only the top half
    cmap = truncate_colormap(cmap, 0.4, 0.9) 
    norm = plt.Normalize(gamma_hz_list.min(), gamma_hz_list.max())

    # Compute the inverse of the difference in inverse velocities (for r calculation)
    inv_velocity_diff = (1 / v_eph_um_s) - (1 / v_ax_um_s)
    velocity_factor = 1 / inv_velocity_diff  # Units: μm/s


    # Initialize lists for custom legend entries
    handles = []
    labels = []

    # Shade the first positive peak for each frequency with varying intensity
    for hz, w in zip(gamma_hz_list, weights):
        # Compute start and end Delta_t
        delta_t_start = (3 / (4 * hz))  # Start at 3π/2 zero crossing into positive
        delta_t_end = (5 / (4 * hz))    # End at next zero crossing into negative

        # Compute corresponding distances
        r_start = delta_t_start * velocity_factor  # Units: μm
        r_end = delta_t_end * velocity_factor      # Units: μm

        # Find indices where r_um is between r_start and r_end
        peak_indices = np.where((r_um >= r_start) & (r_um <= r_end))[0]

        if peak_indices.size == 0:
            continue  # Skip if no points in the range

        # Get the corresponding distance range and correlation values
        r_peak = r_um[peak_indices]
        R = correlation_function(r_peak, hz, v_eph_um_s, v_ax_um_s, lambda_eph)

        # Normalize the correlation function values to [0, 1]
        R_normalized = (R - R.min()) / (R.max() - R.min())

        max_value = np.max(R_normalized)
        max_indices = np.where(R_normalized == max_value)[0]
        max_positions = r_um[max_indices]
        print(f'Peaks for {hz} Hz at {max_distance/1000*max_positions}')

        # Create a 2D array for shading
        Z = np.tile(R_normalized, (2, 1))

        # Map the frequency to a color
        color = cmap(norm(hz))

        # Create a custom colormap from white to the frequency color
        custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', ['white', color])

        # Use imshow to plot the shading
        extent = [r_peak[0], r_peak[-1], ymin, ymax]
        im = ax1.imshow(Z, extent=extent, origin='lower', aspect='auto', cmap=custom_cmap, alpha=0.5, label=f'First peak of SC for {round(hz)} Hz')

        # Create a custom legend handle
        patch = Patch(color=color, label=f'First peak of SC for {round(hz)} Hz')
        handles.append(patch)
        labels.append(f'First peak of SC for {round(hz)} Hz')


    # Create a secondary y-axis for the correlation function
    ax2 = ax1.twinx()
    ax2.set_ylabel('Synergy Correlation Function (a.u.)')

    # Plot the full model
    #ax2.plot(r_um, total_r_full, color='red', linestyle='--', label='Full Model')

    # Create a colormap and normalizer for frequencies
    cmap = plt.cm.viridis
    # Truncate the colormap to use only the top half
    cmap = truncate_colormap(cmap, 0.5, 0.85) 

    # Plot the reduced models
    for idx, (frequencies, model_curve) in enumerate(reduced_models):
        if len(reduced_models) > 1:
            color = cmap(1 - idx / (len(reduced_models) - 1))
        else:
            color = cmap(0.5)
        print(idx / len(reduced_models))
        frequencies = [round(f) for f in frequencies]
        ax2.plot(r_um, model_curve, linestyle='--', color=color, label=f'Model with {frequencies} Hz')
        
        ax2.fill_between(
        r_um,
        lower_cis[idx],
        upper_cis[idx],
        alpha=0.2,
        color=color
        )

    # Plot the uniform model
    #ax2.plot(r_um, uniform_model, color='green', linestyle='-', label='Uniform Model', alpha=0.6)

    # Display BIC and MI comparisons
    print(f'Full Model BIC: {bic_values[0]}, MI: {mi_values[0]}')
    print(f'Uniform Model BIC: {uniform_bic}, MI: {uniform_mi}')
    for idx, (bic, mi, (LRT_value, p_value)) in enumerate(zip(bic_values, mi_values, lrt_values)):
        print(f'Reduced Model {idx + 1} BIC: {bic}, MI: {mi}, LRT: {LRT_value}, p:{p_value}')

    # Add legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.show()




######################### F(DISTANCE) = HZ #########################

def calculate_avg_firing_rate_distance(spikes_data, layout, start_time, end_time, max_electrodes=210):

    ref_electrodes = get_activity_sorted_electrodes([spikes_data], start=10, stop=max_electrodes, start_time=0, end_time=600)
    layout_df = pd.DataFrame(layout)
    spikes_df = pd.DataFrame(spikes_data)
    
    # Assign distances to all valid ref_electrodes
    spikes_df, distances_df = assign_r_distance_all(spikes_df, layout_df, ref_electrodes)
    
    # Filter spikes within the time window
    spikes_df_during = spikes_df[(spikes_df['time'] >= start_time) & (spikes_df['time'] <= end_time)]
    
    # Compute firing rates per electrode
    duration = end_time - start_time
    firing_counts = spikes_df_during['electrode'].value_counts().reset_index()
    firing_counts.columns = ['electrode', 'counts']
    firing_counts['firing_rate'] = firing_counts['counts'] / duration
    
    # Merge firing rates with distances
    firing_rates_df = pd.merge(firing_counts, distances_df, on='electrode', how='inner')
    
    # Remove rows where electrode == ref_electrode
    firing_rates_df = firing_rates_df[firing_rates_df['electrode'] != firing_rates_df['ref_electrode']]
    
    # Extract firing rates and distances
    avg_firing_rates = firing_rates_df['firing_rate'].values
    electrode_distances = firing_rates_df['distance'].values
    
    return avg_firing_rates, electrode_distances

def plot_avg_firing_rate_models(
    spikes_data_list, layout_list, start_times, end_times, max_electrodes,
    log=False, v_eph=0.1, v_ax=0.45, lambda_eph=100000, std=0.15
):
    # Collect results in parallel
    results = Parallel(n_jobs=-1)(
        delayed(calculate_avg_firing_rate_distance)(
            spikes_data, layout, start_time, end_time, max_electrodes
        )
        for spikes_data, layout, start_time, end_time in zip(spikes_data_list, layout_list, start_times, end_times)
    )

    # Concatenate results from all reference electrodes
    all_avg_firing_rates = np.concatenate([result[0] for result in results])
    all_distances = np.concatenate([result[1] for result in results])
    min_distance = 50
    max_distance = 3500

    # Calculate distance bins based on the actual data
    if log:
        distance_bins = np.logspace(
            np.log10(min_distance), np.log10(max_distance), num=100
        )
    else:
        distance_bins = np.linspace(min_distance, max_distance, num=100)

    # Bin data by distances and calculate mean and standard error
    bin_means, bin_edges, _ = binned_statistic(
        all_distances, all_avg_firing_rates, statistic=np.nanmean, bins=distance_bins
    )
    bin_std_err, _, _ = binned_statistic(
        all_distances, all_avg_firing_rates,
        statistic=lambda x: np.std(x) / np.sqrt(len(x)), bins=distance_bins
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Remove bins with NaN values
    valid_bins = ~np.isnan(bin_means)
    bin_centers = bin_centers[valid_bins]
    bin_means = bin_means[valid_bins]
    bin_std_err = bin_std_err[valid_bins]

    # Get hz_list
    all_ifr_values = np.array(get_ifr_peaks(spikes_data_list, start_times, end_times)[0])
    means, std_devs, weights = fit_gaussians_to_ifr_data(all_ifr_values)
    means = np.array(means)
    weights = np.array(weights)
    gamma_hz_list = np.sort(means[(means > 30) & (means < 1000)])
    weights = weights[(means > 30) & (means < 1000)]
    weights = weights / weights.min()
    
    print(f'Included IFR Peaks: {gamma_hz_list}')

    # Ensure velocities are in micrometers per second
    v_eph_um_s = v_eph * 1e6  # Convert from m/s to μm/s
    v_ax_um_s = v_ax * 1e6    # Convert from m/s to μm/s
    std_um_s = std * 1e6

    # Create a fine-grained distance array for the correlation function
    r_um = np.linspace(min_distance, max_distance, 1000)  # Smooth curve

    # Create and plot the Full Model:
    total_r_full = np.zeros_like(r_um)
    for hz, w in zip(gamma_hz_list, weights):
        total_r_full += correlation_function(r_um, hz, v_eph_um_s, v_ax_um_s, lambda_eph) * w

    # Fit and assess the full and reduced models
    X = r_um.reshape(-1, 1)
    y = np.interp(r_um, bin_centers, bin_means)

    # Compute residuals and log-likelihood for the full model
    full_residuals = y - total_r_full
    logL_full = log_likelihood(full_residuals, len(gamma_hz_list))

    # Prepare to store the results of each reduced model and their BIC & MI
    reduced_models = []
    upper_cis = []
    lower_cis = []
    bic_values = []
    mi_values = []
    lrt_values = []
    residuals_dict = {}
    current_gamma_list = list(gamma_hz_list)

    while len(current_gamma_list) > 0:
        # Create the correlation curve using the current list of frequencies
        total_r_reduced = np.zeros_like(r_um)
        steps = 50
        total_r_reduced_list = [np.zeros_like(r_um) for _ in range(50)]
        current_weights = weights[:len(current_gamma_list)]
        
        for hz, w in zip(current_gamma_list, current_weights):
            total_r_reduced += correlation_function(r_um, hz, v_eph_um_s, v_ax_um_s, lambda_eph) * w
            for idx, v in enumerate(np.linspace(v_ax_um_s - std_um_s, v_ax_um_s + std_um_s, steps)):
                total_r_reduced_list[idx] += correlation_function(r_um, hz, v_eph_um_s, v, lambda_eph) * w
        
        # Compute residuals and log-likelihood for reduced model
        reduced_residuals = y - total_r_reduced
        logL_reduced = log_likelihood(reduced_residuals, len(current_gamma_list))
        print(f"Log-likelihood of full model: {logL_full}")
        print(f"Log-likelihood of reduced model: {logL_reduced}")
        # Likelihood Ratio Test
        df = len(gamma_hz_list) - len(current_gamma_list)  # Degrees of freedom difference
        LRT_stat, p_value = likelihood_ratio_test(logL_full, logL_reduced, df)
        lrt_values.append((LRT_stat, p_value))

        # Compute residuals and BIC
        residuals = y - total_r_reduced
        print(len(residuals[np.isfinite(residuals)]))
        bic = calculate_bic(len(y), residuals, len(current_gamma_list))
        bic_values.append(bic)
        
        # Compute MI
        mi = mutual_info_regression(total_r_reduced.reshape(-1, 1), y)
        mi_values.append(mi[0])
        
        # Save the model result for plotting later
        reduced_models.append((current_gamma_list.copy(), total_r_reduced))
        stacked_array = np.stack(total_r_reduced_list) 
        upper_ci = np.max(stacked_array, axis=0)
        stacked_array = np.stack(total_r_reduced_list) 
        lower_ci = np.min(stacked_array, axis=0)
        upper_cis.append(upper_ci)
        lower_cis.append(lower_ci)
        residuals_dict[f'Reduced Model {len(current_gamma_list)} Hz'] = np.array(residuals.tolist(), dtype = float)
        
        # Remove the highest frequency from the list
        #current_gamma_list = current_gamma_list[:-1]
        current_gamma_list = []

    # Plot the data
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot the error bars on the primary y-axis
    ax1.plot(bin_centers, bin_means, color='blue', label='Mean Firing Rate')
    ax1.fill_between(
        bin_centers,
        bin_means - bin_std_err,
        bin_means + bin_std_err,
        alpha=0.4,
        color='blue'
    )
    ax1.set_xlabel('Distance from Electrode ($\mu m$)')
    ax1.set_ylabel('Mean Firing Rate (Hz)')
    ax1.set_title('Averaged Firing Rate vs. Distance with First Synergy Coefficient Peaks')

    # Get the y-axis limits to set the extent of the shading
    ymin, ymax = ax1.get_ylim()

    # Create a colormap and normalizer for frequencies
    cmap = plt.cm.viridis
    # Truncate the colormap to use only the top half
    cmap = truncate_colormap(cmap, 0.4, 0.9) 
    norm = plt.Normalize(gamma_hz_list.min(), gamma_hz_list.max())

    # Compute the inverse of the difference in inverse velocities (for r calculation)
    inv_velocity_diff = (1 / v_eph_um_s) - (1 / v_ax_um_s)
    velocity_factor = 1 / inv_velocity_diff  # Units: μm/s


    # Initialize lists for custom legend entries
    handles = []
    labels = []

    # Shade the first positive peak for each frequency with varying intensity
    for hz, w in zip(gamma_hz_list, weights):
        # Compute start and end Delta_t
        delta_t_start = (3 / (4 * hz))  # Start at 3π/2 zero crossing into positive
        delta_t_end = (5 / (4 * hz))    # End at next zero crossing into negative

        # Compute corresponding distances
        r_start = delta_t_start * velocity_factor  # Units: μm
        r_end = delta_t_end * velocity_factor      # Units: μm

        # Find indices where r_um is between r_start and r_end
        peak_indices = np.where((r_um >= r_start) & (r_um <= r_end))[0]

        if peak_indices.size == 0:
            continue  # Skip if no points in the range

        # Get the corresponding distance range and correlation values
        r_peak = r_um[peak_indices]
        R = correlation_function(r_peak, hz, v_eph_um_s, v_ax_um_s, lambda_eph)

        # Normalize the correlation function values to [0, 1]
        R_normalized = (R - R.min()) / (R.max() - R.min())

        max_value = np.max(R_normalized)
        max_indices = np.where(R_normalized == max_value)[0]
        max_positions = r_um[max_indices]
        print(f'Peaks for {hz} Hz at {max_distance/1000*max_positions}')

        # Create a 2D array for shading
        Z = np.tile(R_normalized, (2, 1))

        # Map the frequency to a color
        color = cmap(norm(hz))

        # Create a custom colormap from white to the frequency color
        custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', ['white', color])

        # Use imshow to plot the shading
        extent = [r_peak[0], r_peak[-1], ymin, ymax]
        im = ax1.imshow(Z, extent=extent, origin='lower', aspect='auto', cmap=custom_cmap, alpha=0.5, label=f'First peak of SC for {round(hz)} Hz')

        # Create a custom legend handle
        patch = Patch(color=color, label=f'First peak of SC for {round(hz)} Hz')
        handles.append(patch)
        labels.append(f'First peak of SC for {round(hz)} Hz')



    # Compare models using statistical tests (only if more than one model)
    if len(residuals_dict) > 1:
        compare_model_fits(residuals_dict)
    else:
        print("Skipped statistical model comparison: need at least two models.")



    # Create a secondary y-axis for the correlation function
    ax2 = ax1.twinx()
    ax2.set_ylabel('Synergy Coefficient')
    # Adjust the secondary y-axis limits if needed
    ax2.set_ylim(np.min(total_r_full), np.max(total_r_full))

    # Plot the full model
    #ax2.plot(r_um, total_r_full, color='red', linestyle='--', label='Full Model')

    # Create a colormap and normalizer for frequencies
    cmap = plt.cm.viridis
    # Truncate the colormap to use only the top half
    cmap = truncate_colormap(cmap, 0.5, 0.85) 

    # Plot the reduced models
    for idx, (frequencies, model_curve) in enumerate(reduced_models):
        if len(reduced_models) > 1:
            color = cmap(1 - idx / (len(reduced_models) - 1))
        else:
            color = cmap(0.5)
        print(idx / len(reduced_models))
        frequencies = [round(f) for f in frequencies]
        ax2.plot(r_um, model_curve, linestyle='--', color=color, label=f'Model with {frequencies} Hz')
        
        ax2.fill_between(
            r_um,
            lower_cis[idx],
            upper_cis[idx],
            alpha=0.2,
            color=color
        )

    # Display BIC and MI comparisons
    print(f'Full Model BIC: {bic_values[0]}, MI: {mi_values[0]}')
    for idx, (bic, mi, (LRT_stat, p_value)) in enumerate(zip(bic_values, mi_values, lrt_values)):
        print(f'Reduced Model {idx + 1} BIC: {bic}, MI: {mi}, LRT: {LRT_stat}, p: {p_value}')

    # Add legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.show()


######################### MOST ACTIVE ELECTRODE DISTANCE HISTOGRAM #########################


def get_proportion_outside_boundary(x1, y1, x2, y2, max_x, max_y):
    """
    Calculate the proportion of the circle (with radius r = distance between (x1, y1) and (x2, y2))
    that falls outside the rectangular boundary defined by (0, max_x) and (0, max_y).
    
    Parameters:
    - x1, y1: Coordinates of the first electrode.
    - x2, y2: Coordinates of the second electrode.
    - max_x, max_y: Dimensions of the electrode array boundary.
    
    Returns:
    - Weighting factor: Inverse of the proportion of the circle outside the boundary.
    """
    # Distance between electrodes
    dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    # If the distance is 0, the proportion is trivially 1 (no boundary effect)
    if dist == 0:
        return 1.0
    
    # Define the circle radius as the distance
    radius = dist
    
    # Create a circle around the midpoint between (x1, y1) and (x2, y2)
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2

    # Check if parts of the circle fall outside the boundary
    # Approximate by sampling points on the circle
    num_points = 100
    theta = np.linspace(0, 2 * np.pi, num_points)
    circle_x = mid_x + radius * np.cos(theta)
    circle_y = mid_y + radius * np.sin(theta)

    # Count how many points fall outside the boundaries
    outside_count = np.sum((circle_x < 0) | (circle_x > max_x) | (circle_y < 0) | (circle_y > max_y))
    proportion_outside = outside_count / num_points

    # Weight is the inverse of the proportion of the circumference inside the boundary
    proportion_inside = max(0.01, 1 - proportion_outside)  # Prevent division by 0
    weight = 1 / proportion_inside

    weight /= (2 * np.pi * radius)
    weight /= 2

    return weight

def process_spike_data(spikes_data, layout, start_time, end_time, max_electrodes):
    # Get the most active electrodes
    most_active_electrodes = get_activity_sorted_electrodes([spikes_data], 10, max_electrodes, 0, 600)
    
    # Convert layout to DataFrame if not already
    layout_df = pd.DataFrame(layout)
    if not {'electrode', 'x', 'y'}.issubset(layout_df.columns):
        layout_df.columns = ['electrode', 'x', 'y']
    
    # Filter the layout to include only the most active electrodes
    active_layout = layout_df[layout_df['electrode'].isin(most_active_electrodes)]
    
    # Extract coordinates of the most active electrodes
    coords = active_layout[['x', 'y']].values
    num_electrodes = coords.shape[0]
    
    # Determine the maximum x and y boundaries
    max_x = layout_df['x'].max()
    max_y = layout_df['y'].max()

    # Compute pairwise distances with weight adjustments
    weighted_distances = []
    weights = []
    for i in range(num_electrodes):
        for j in range(i + 1, num_electrodes):
            # Coordinates of the electrodes
            x1, y1 = coords[i]
            x2, y2 = coords[j]
            
            # Compute Euclidean distance
            dist = np.linalg.norm([x2 - x1, y2 - y1])
            
            # Calculate weighting factor based on boundary effects
            weight = get_proportion_outside_boundary(x1, y1, x2, y2, max_x, max_y)
            
            # Append distance and its associated weight
            weighted_distances.append(dist)
            weights.append(weight)
    
    return weighted_distances, weights

def plot_distance_hist(spikes_data_list, layout_list, start_times, end_times, max_electrodes, finite_size_correction=True):

    min_distance = 50
    max_distance = 3500
    lambda_eph = 100000
    # Ensure velocities are in micrometers per second
    v_eph = 0.1
    v_ax = 0.45
    v_eph_um_s = v_eph * 1e6  # Convert from m/s to μm/s
    v_ax_um_s = v_ax * 1e6    # Convert from m/s to μm/s



    # Use Parallel to process all datasets
    results = Parallel(n_jobs=-1)(
        delayed(process_spike_data)(
            spikes_data, layout, start_time, end_time, max_electrodes
        )
        for spikes_data, layout, start_time, end_time in zip(spikes_data_list, layout_list, start_times, end_times)
    )

    # Flatten the lists of distances and weights
    all_distances = []
    all_weights = []
    for distances, weights in results:
        all_distances.extend(distances)
        all_weights.extend(weights)

    # Get weighted gamma frequency list
    all_ifr_values = np.array(get_ifr_peaks(spikes_data_list, start_times, end_times)[0])
    means, std_devs, weights = fit_gaussians_to_ifr_data(all_ifr_values)
    means = np.array(means)
    weights = np.array(weights)
    gamma_hz_list = means[(means>30)&(means<1000)]
    weights = weights[(means>30)&(means<1000)]
    weights = weights/weights.min()

    # Create a fine-grained distance array for the correlation function
    r_um = np.linspace(min_distance, max_distance, 1000)  # Smooth curve

    # Calculate the correlation values over the fine-grained distance array
    total_r = np.zeros_like(r_um)
    for hz, w in zip(gamma_hz_list, weights):
        total_r += correlation_function(r_um, hz, v_eph_um_s, v_ax_um_s, lambda_eph) * w



    # Plot the weighted histogram of distances
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histogram with weighting
    ax.hist(all_distances, bins=74, weights=all_weights if finite_size_correction else None, alpha=0.4)

    # Get the y-axis limits to set the extent of the shading
    ymin, ymax = ax.get_ylim()

    # Create a colormap and normalizer for frequencies
    cmap = plt.cm.viridis
    # Truncate the colormap to use only the top half
    cmap = truncate_colormap(cmap, 0.4, 0.9) 
    norm = plt.Normalize(gamma_hz_list.min(), gamma_hz_list.max())

    # Compute the inverse of the difference in inverse velocities (for r calculation)
    inv_velocity_diff = (1 / v_eph_um_s) - (1 / v_ax_um_s)
    velocity_factor = 1 / inv_velocity_diff  # Units: μm/s


    # Initialize lists for custom legend entries
    handles = []
    labels = []

    # Shade the first positive peak for each frequency with varying intensity
    for hz, w in zip(gamma_hz_list, weights):
        # Compute start and end Delta_t
        delta_t_start = (3 / (4 * hz))  # Start at 3π/2 zero crossing into positive
        delta_t_end = (5 / (4 * hz))    # End at next zero crossing into negative

        # Compute corresponding distances
        r_start = delta_t_start * velocity_factor  # Units: μm
        r_end = delta_t_end * velocity_factor      # Units: μm

        # Find indices where r_um is between r_start and r_end
        peak_indices = np.where((r_um >= r_start) & (r_um <= r_end))[0]

        if peak_indices.size == 0:
            continue  # Skip if no points in the range

        # Get the corresponding distance range and correlation values
        r_peak = r_um[peak_indices]
        R = correlation_function(r_peak, hz, v_eph_um_s, v_ax_um_s, lambda_eph)

        # Normalize the correlation function values to [0, 1]
        R_normalized = (R - R.min()) / (R.max() - R.min())

        max_value = np.max(R_normalized)
        max_indices = np.where(R_normalized == max_value)[0]
        max_positions = r_um[max_indices]
        print(f'Peaks for {hz} Hz at {max_distance/1000*max_positions}')


        # Create a 2D array for shading
        Z = np.tile(R_normalized, (2, 1))

        # Map the frequency to a color
        color = cmap(norm(hz))

        # Create a custom colormap from white to the frequency color
        custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', ['white', color])

        # Use imshow to plot the shading
        extent = [r_peak[0], r_peak[-1], ymin, ymax]
        im = ax.imshow(Z, extent=extent, origin='lower', aspect='auto', cmap=custom_cmap, alpha=0.5, label=f'First peak of SC for {round(hz)} Hz')

        # Create a custom legend handle
        patch = Patch(color=color, label=f'First peak of SC for {round(hz)} Hz')
        handles.append(patch)
        labels.append(f'First peak of SC for {round(hz)} Hz')

    # Collect histogram's handle and label
    hist_handles, hist_labels = ax.get_legend_handles_labels()

    # Combine all handles and labels
    all_handles = hist_handles + handles
    all_labels = hist_labels + labels

    # Set plot limits and labels
    ax.set_xlim(0, max_distance)
    ax.set_xlabel('Distance between Most Active Electrodes (µm)')
    ax.set_ylabel('Weighted Count')
    ax.set_title('Weighted Histogram of Distances Between Most Active Electrodes Across Recordings')

    # Create the legend
    ax.legend(all_handles, all_labels, loc='lower right')

    plt.show()


######################### POLAR LAYOUT #########################

def save_theta_results_to_csv(heatmap_data, distance_bins, theta_bins, delays, output_filename='results.csv'):
    # Prepare data for CSV
    data = []
    for i, delay in enumerate(delays):
        for d_idx in range(heatmap_data.shape[1]):
            for t_idx in range(heatmap_data.shape[2]):
                data.append({
                    'delay': delay,
                    'distance_bin': distance_bins[d_idx],
                    'radian_bin': theta_bins[t_idx],
                    'probability': heatmap_data[i, t_idx, d_idx]
                })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save DataFrame to CSV
    df.to_csv(output_filename, index=False)

def compute_spike_distance_theta_cofiring(spikes_data, layout, ref_electrode, start_time, end_time, window_size, delays):
    spikes_df = pd.DataFrame(spikes_data)
    layout_df = pd.DataFrame(layout)

    spikes_df, layout_df = assign_r_theta_distance(spikes_df, layout_df, ref_electrode)

    spikes_df_during = spikes_df[(spikes_df['time'] >= start_time) & (spikes_df['time'] <= end_time)]
    
    firing_times = spikes_df_during['time'][spikes_df_during['electrode'] == ref_electrode]
    print(f"Electrode {ref_electrode} fired {len(firing_times)} times in total.")

    cofiring_proportions_by_delay = {delay: [] for delay in delays}
    electrode_distances_by_delay = {delay: [] for delay in delays}
    electrode_thetas_by_delay = {delay: [] for delay in delays}

    for delay in delays:
        delay_sec = delay / 1000
        proportions = calculate_cofiring_proportions(spikes_df_during, firing_times, window_size=window_size / 10000, delay=delay_sec, ref_electrode=ref_electrode)
        
        for electrode, proportion in proportions.items():
            if electrode == ref_electrode:
                continue

            cofiring_proportions_by_delay[delay].append(proportion)
            electrode_distances_by_delay[delay].append(layout_df.loc[layout_df['electrode'] == electrode, 'distance'].values[0])
            electrode_thetas_by_delay[delay].append(layout_df.loc[layout_df['electrode'] == electrode, 'theta'].values[0])

    return cofiring_proportions_by_delay, electrode_distances_by_delay, electrode_thetas_by_delay

def process_electrode_theta(spikes_data_list, layout_list, ref_electrode, start_time, end_time, window_size, delays):
    heatmap_data_sum = np.zeros((len(delays), 30, 30))  # Sum of heatmap data across recordings
    count_data = np.zeros((len(delays), 30, 30))  # To count the number of valid entries for each bin
    for spikes_data, layout in zip(spikes_data_list, layout_list):
        proportions_by_delay, distances_by_delay, thetas_by_delay = compute_spike_distance_theta_cofiring(spikes_data, layout, ref_electrode, start_time, end_time, window_size, delays)

        for i, delay in enumerate(delays):
            all_distances = np.array(distances_by_delay[delay])
            all_thetas = np.array(thetas_by_delay[delay])
            all_cofiring_proportions = np.array(proportions_by_delay[delay])

            distance_bins = np.linspace(min(all_distances), max(all_distances), num=31)
            theta_bins = np.linspace(-np.pi, np.pi, num=31)  # Full 360 degrees

            bin_means, _, _, _ = binned_statistic_2d(all_distances, all_thetas, all_cofiring_proportions, statistic='mean', bins=[distance_bins, theta_bins])
            valid_bins = np.logical_not(np.isnan(bin_means))

            # Accumulate sum of heatmap data and count for averaging
            heatmap_data_sum[i, valid_bins] += bin_means[valid_bins]
            count_data[i, valid_bins] += 1

    # Calculate the average heatmap data, handling cases where count_data is zero
    with np.errstate(divide='ignore', invalid='ignore'):
        heatmap_data_avg = np.true_divide(heatmap_data_sum, count_data)
        heatmap_data_avg[~np.isfinite(heatmap_data_avg)] = 0  # Set infinities and NaNs to 0

    return heatmap_data_avg, distance_bins, theta_bins

def create_spike_distance_theta_gif(spikes_data_list, layout_list, ref_electrodes, start_time, end_time, window_size=10, delays=np.linspace(-80, 160, 121), normalize=False, output_filename='spike_density_theta.gif'):
    results = Parallel(n_jobs=-1)(
        delayed(process_electrode_theta)(spikes_data_list, layout_list, ref_electrode, start_time, end_time, window_size, delays)
        for ref_electrode in ref_electrodes
    )

    avg_heatmap_data = np.mean([result[0] for result in results], axis=0)
    distance_bins = results[0][1]
    theta_bins = results[0][2][:-1]  # Remove the last bin edge to match the bin count

    if normalize:
        avg_heatmap_data = normalize_against_t0(avg_heatmap_data, delays)

    # Save results to CSV
    file_name = f'spike_theta_{ref_electrodes[0]}'
    save_theta_results_to_csv(avg_heatmap_data, distance_bins, theta_bins, delays, output_filename=file_name+'.csv')
    filenames = []

    vmin = np.min(avg_heatmap_data)
    vmax = np.max(avg_heatmap_data)

    # Ensure theta bins are in radians for polar plotting
    theta_bins = np.linspace(-np.pi, np.pi, avg_heatmap_data.shape[2] + 1)

    # Generate GIF frames with polar plots
    filenames = []
    for i, delay in enumerate(delays):
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
        
        # Create the polar plot
        theta_grid, r_grid = np.meshgrid(theta_bins, distance_bins)
        cax = ax.pcolormesh(theta_grid, r_grid, avg_heatmap_data[i], cmap=get_cmap('magma'), vmin=vmin, vmax=vmax)
        
        ax.set_theta_zero_location('N')  # Set the zero location to North (top)
        ax.set_theta_direction(-1)       # Set the theta direction to clockwise
        ax.tick_params(colors='white') 
        ax.set_title(f'Firing Probability of {len(ref_electrodes)} Electrodes around Electrode {ref_electrodes[0]} \nTime Delay: {delay:.2f} ms')
        
        # Add colorbar
        fig.colorbar(cax, ax=ax, label='Probability')
        
        # Save each frame
        filename = f'frame_{i:03d}.png'
        filenames.append(filename)
        plt.savefig(filename)
        plt.close()

    # Create GIF
    with imageio.get_writer(file_name + '.gif', mode='I', duration=2) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    # Clean up temporary files
    for filename in filenames:
        os.remove(filename)



######################### GRID LAYOUT #########################

def save_grid_results_to_csv(heatmap_data, x_bins, y_bins, delays, output_filename='results.csv'):
    # Prepare data for CSV
    data = []
    for i, delay in enumerate(delays):
        for y_idx in range(heatmap_data.shape[1]):
            for x_idx in range(heatmap_data.shape[2]):
                data.append({
                    'delay': delay,
                    'x_bin': x_bins[x_idx],
                    'y_bin': y_bins[y_idx],
                    'probability': heatmap_data[i, y_idx, x_idx]
                })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save DataFrame to CSV
    df.to_csv(output_filename, index=False)

def compute_spike_distance_grid(spikes_data, layout, ref_electrode, start_time, end_time, window_size, delays):
    spikes_df = pd.DataFrame(spikes_data)
    layout_df = pd.DataFrame(layout)

    spikes_df, layout_df = assign_r_distance(spikes_df, layout_df, ref_electrode)

    spikes_df_during = spikes_df[(spikes_df['time'] >= start_time) & (spikes_df['time'] <= end_time)]
    firing_times = spikes_df_during['time'][spikes_df_during['electrode'] == ref_electrode]

    print(f"Electrode {ref_electrode} fired {len(firing_times)} times in total.")

    # Define the spatial grid
    x_min, x_max = layout_df['x'].min(), layout_df['x'].max()
    y_min, y_max = layout_df['y'].min(), layout_df['y'].max()
    x_bins = np.arange(x_min, x_max + 100, 100)
    y_bins = np.arange(y_min, y_max + 100, 100)

    cofiring_proportions_by_delay = {delay: np.zeros((len(y_bins) - 1, len(x_bins) - 1)) for delay in delays}
    cofiring_counts_by_delay = {delay: np.zeros((len(y_bins) - 1, len(x_bins) - 1)) for delay in delays}

    for delay in delays:
        delay_sec = delay / 1000
        proportions = calculate_cofiring_proportions(spikes_df_during, firing_times, window_size=window_size / 10000, delay=delay_sec, ref_electrode=ref_electrode)
        
        for electrode, proportion in proportions.items():
            if electrode == ref_electrode:
                continue

            # Check if the electrode exists in the layout
            electrode_location = layout_df.loc[layout_df['electrode'] == electrode, ['x', 'y']]
            if electrode_location.empty:
                print(f"Warning: Electrode {electrode} not found in layout.")
                continue  # Skip this electrode since it's not in the layout

            x, y = electrode_location.values[0]
            x_idx = np.digitize(x, x_bins) - 1
            y_idx = np.digitize(y, y_bins) - 1

            if 0 <= x_idx < len(x_bins) - 1 and 0 <= y_idx < len(y_bins) - 1:
                cofiring_proportions_by_delay[delay][y_idx, x_idx] += proportion
                cofiring_counts_by_delay[delay][y_idx, x_idx] += 1

    # Normalize the cofiring proportions to avoid exceeding 1
    for delay in delays:
        with np.errstate(divide='ignore', invalid='ignore'):
            cofiring_proportions_by_delay[delay] = np.divide(cofiring_proportions_by_delay[delay], cofiring_counts_by_delay[delay])
            cofiring_proportions_by_delay[delay][~np.isfinite(cofiring_proportions_by_delay[delay])] = 0  # Set infinities and NaNs to 0

    return cofiring_proportions_by_delay, x_bins, y_bins

def process_electrode_grid(spikes_data_list, layout_list, ref_electrode, start_time, end_time, window_size, delays):
    grid_size = None  # We will define the grid size based on x_bins and y_bins
    heatmap_data_sum = None  # Initialize later when grid size is known
    count_data = None  # Initialize later when grid size is known

    for spikes_data, layout in zip(spikes_data_list, layout_list):
        proportions_by_delay, x_bins, y_bins = compute_spike_distance_grid(spikes_data, layout, ref_electrode, start_time, end_time, window_size, delays)
        
        if grid_size is None:
            # Initialize heatmap and count data with correct grid size
            grid_size = (len(y_bins) - 1, len(x_bins) - 1)
            heatmap_data_sum = np.zeros((len(delays), *grid_size))
            count_data = np.zeros((len(delays), *grid_size))

        for i, delay in enumerate(delays):
            heatmap_data_sum[i] += proportions_by_delay[delay]
            count_data[i] += 1

    with np.errstate(divide='ignore', invalid='ignore'):
        heatmap_data_avg = np.true_divide(heatmap_data_sum, count_data)
        heatmap_data_avg[~np.isfinite(heatmap_data_avg)] = 0  # Set infinities and NaNs to 0

    return heatmap_data_avg, x_bins, y_bins

def create_spike_distance_grid_gif(spikes_data_list, layout_list, ref_electrodes, start_time, end_time, window_size=10, delays=np.linspace(-80, 160, 121), normalize=False):
    results = Parallel(n_jobs=-1)(
        delayed(process_electrode_grid)(spikes_data_list, layout_list, ref_electrode, start_time, end_time, window_size, delays)
        for ref_electrode in ref_electrodes
    )

    avg_heatmap_data = np.mean([result[0] for result in results], axis=0)
    x_bins = results[0][1]
    y_bins = results[0][2]

    if normalize:
        avg_heatmap_data = normalize_against_t0(avg_heatmap_data, delays)

    # Save results to CSV
    file_name = f'spike_grid_{ref_electrodes[0]}'
    save_grid_results_to_csv(avg_heatmap_data, x_bins, y_bins, delays, output_filename=file_name+'.csv')

    filenames = []

    vmin = np.min(avg_heatmap_data)
    vmax = np.max(avg_heatmap_data)

    for i, delay in enumerate(delays):
        fig, ax = plt.subplots(figsize=(8, 8))
        cax = ax.imshow(avg_heatmap_data[i], cmap=get_cmap('magma'), extent=[x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]], vmin=vmin, vmax=vmax, origin='lower')
        ax.set_xlabel('X Coordinate (in micrometer)')
        ax.set_ylabel('Y Coordinate (in micrometer)')
        ax.set_title(f'Firing Frequency Ratio of {len(ref_electrodes)} Electrodes around {ref_electrodes[0]}\nTime Delay: {delay:.2f} ms')
        
        fig.colorbar(cax, ax=ax, label='Firing Frequency Ratio')

        # Save each frame
        filename = f'frame_{i:03d}.png'
        filenames.append(filename)
        plt.gca().set_aspect('equal')
        plt.savefig(filename)
        plt.close()

    # Create GIF
    with imageio.get_writer(file_name+'.gif', mode='I', duration=0.2) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    # Clean up temporary files
    for filename in filenames:
        os.remove(filename)



######################### DCT #########################

def compute_interpolated_firing_rate_grid(spikes_data, layout, start_time, end_time, grid_size=50):
    """
    Computes the interpolated firing rate grid for a single recording.
    """
    # Step 1: Prepare the data and calculate firing rates
    spikes_df = pd.DataFrame(spikes_data)
    layout_df = pd.DataFrame(layout)

    # Map channels to electrodes
    channel_to_electrode = layout_df.set_index('channel')['electrode'].to_dict()

    # Filter data within the specified time range
    valid_indices = (spikes_df['time'] >= start_time) & (spikes_df['time'] <= end_time)
    filtered_channels = spikes_df['channel'][valid_indices]

    # Map channels to electrodes in the filtered data
    filtered_electrodes = filtered_channels.map(channel_to_electrode)

    # Calculate average firing rate for each electrode
    unique_electrodes, electrode_counts_in_recording = np.unique(filtered_electrodes, return_counts=True)
    time_duration = end_time - start_time
    firing_rates = dict(zip(unique_electrodes, electrode_counts_in_recording / time_duration))

    # Prepare the layout map and firing rates
    map_electrodes = [electrode for electrode in layout_df['electrode'] if electrode in firing_rates]
    map_firing_rates = [firing_rates[electrode] for electrode in map_electrodes]
    map_x = layout_df.set_index('electrode').loc[map_electrodes, 'x'].values
    map_y = layout_df.set_index('electrode').loc[map_electrodes, 'y'].values

    # Define the grid
    x_n = int(np.ceil(max(layout_df['x'])/grid_size)) - 1
    y_n = int(np.ceil(max(layout_df['y'])/grid_size)) - 1

    x_min, x_max = 1, x_n * grid_size
    y_min, y_max = 1, y_n * grid_size
    x_bins = np.linspace(x_min, x_max, x_n + 1)
    y_bins = np.linspace(y_min, y_max, y_n + 1)

    # Create the grid to store average firing rates
    grid_firing_rates = np.zeros((x_n, y_n))
    grid_counts = np.zeros((x_n, y_n))

    # Known positions and values for interpolation
    known_points = []
    known_values = []

    for x, y, rate in zip(map_x, map_y, map_firing_rates):
        x_idx = np.digitize(x, x_bins) - 1
        y_idx = np.digitize(y, y_bins) - 1
        if 0 <= x_idx < x_n and 0 <= y_idx < y_n:
            grid_firing_rates[x_idx, y_idx] += rate
            grid_counts[x_idx, y_idx] += 1
            known_points.append((x_idx, y_idx))
            known_values.append(rate)

    # Calculate the average firing rate for each cell and interpolate missing values
    with np.errstate(invalid='ignore', divide='ignore'):
        grid_avg_firing_rates = np.divide(grid_firing_rates, grid_counts, where=grid_counts != 0)

    # Prepare grid coordinates for interpolation
    grid_x, grid_y = np.meshgrid(range(x_n), range(y_n), indexing='ij')
    points = np.array(known_points)
    values = np.array(known_values)

    # Interpolate to fill in missing values
    interpolated_grid = griddata(points, values, (grid_x, grid_y), method='cubic', fill_value=0)

    # Replace NaNs or zero areas with interpolated values
    grid_avg_firing_rates = np.where(grid_counts == 0, interpolated_grid, grid_avg_firing_rates)

    return grid_avg_firing_rates, x_min, x_max, y_min, y_max


def plot_firing_rate_grid(grid_avg_firing_rates, x_min, x_max, y_min, y_max):
    """
    Plots the firing rate grid with sanitization to avoid plotting issues.
    """
    # Sanitize data to avoid issues with LogNorm
    grid_avg_firing_rates = np.nan_to_num(grid_avg_firing_rates, nan=0.0, posinf=0.0, neginf=0.0)
    grid_avg_firing_rates[grid_avg_firing_rates == 0] = 1e-6  # Replace zeros with a small positive value

    # Exclude invalid values for normalization
    valid_values = grid_avg_firing_rates[(grid_avg_firing_rates > 1e-6)]
    if valid_values.size == 0:
        raise ValueError("No valid firing rates available for normalization.")
    vmin = np.nanmin(valid_values)
    vmax = np.nanmax(valid_values)

    # Ensure LogNorm works correctly
    if vmin <= 0:
        vmin = 1e-6

    # Plot the grid with average firing rates
    fig, ax = plt.subplots(figsize=(10, 6))
    norm = LogNorm(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap('magma')
    cax = ax.imshow(grid_avg_firing_rates.T, origin='lower', cmap=cmap, norm=norm,
                    extent=[x_min, x_max, y_min, y_max])
    ax.set_xlabel(r'X-coordinate ($\mu m$)')
    ax.set_ylabel(r'Y-coordinate ($\mu m$)')
    ax.set_title('Average Firing Rate')
    ax.set_facecolor('black')

    # Create a subplot for the colorbar
    cbar = fig.colorbar(cax, ax=ax, orientation='vertical')
    cbar.set_label('Average Firing Rate (Hz)')

    plt.grid(True)
    plt.gca().set_aspect('equal')
    plt.show()

def plot_dct_of_firing_rate_grid(grid_avg_firing_rates):
    """
    Applies a Discrete Cosine Transform (DCT) to the interpolated grid and plots the absolute values of the DCT coefficients.
    """
    # Apply 2D Discrete Cosine Transform (DCT)
    dct_transform = dct(dct(grid_avg_firing_rates.T, norm='ortho').T, norm='ortho')

    # Plot the absolute values of the DCT coefficients
    fig, ax = plt.subplots(figsize=(10, 6))
    cax = ax.imshow(np.abs(dct_transform), origin='lower', cmap='magma', aspect='auto')
    ax.set_xlabel('DCT X-coefficient')
    ax.set_ylabel('DCT Y-coefficient')
    ax.set_title('Magnitude of DCT Coefficients')
    cbar = fig.colorbar(cax, ax=ax, orientation='vertical')
    cbar.set_label('Magnitude')
    plt.grid(True)
    plt.gca().set_aspect('equal')
    plt.show()





def interpolated_grids_to_dct(interpolated_grids):
    if len(interpolated_grids) == 0:
        raise ValueError("No interpolated grids provided for DCT averaging.")
    
    # Step 1: Compute the absolute DCT matrices for each grid
    dct_matrices = []
    for grid in interpolated_grids:
        dct_transform = dct(dct(grid.T, norm='ortho').T, norm='ortho')
        #dct_transform = np.abs(dct_transform)
        dct_transform[0, 0] = 0
        dct_transform = dct_transform / np.max(dct_transform)
        dct_matrices.append(dct_transform)
    
    # Step 2: Sum and Average the DCT matrices
    dct_sum = np.sum(dct_matrices, axis=0)
    dct_avg = dct_sum / len(dct_matrices)

    # Step 3: Normalize the averaged DCT (optional)
    dct_avg_normalized = dct_avg / np.max(dct_avg)  # Normalize to [0, 1] range

    return dct_avg_normalized

def plot_average_dct_coefficients(interpolated_grids):
    """
    Computes and plots the average of the absolute DCT matrices across multiple recordings.
    The resulting DCT coefficients are normalized and then visualized.
    
    Parameters:
    - interpolated_grids: list of 2D numpy arrays (each one being an interpolated firing rate grid).
    """

    dct_avg_normalized = interpolated_grids_to_dct(interpolated_grids)

    fig, ax = plt.subplots(figsize=(10, 6))
    cax = ax.imshow(np.abs(dct_avg_normalized), origin='lower', cmap='OrRd', aspect='auto')
    ax.set_xlabel('DCT X-coefficient')
    ax.set_ylabel('DCT Y-coefficient')
    ax.set_title('Average Magnitude of DCT Coefficients Across Recordings')
    cbar = fig.colorbar(cax, ax=ax, orientation='vertical')
    cbar.set_label('Normalized Magnitude')
    plt.grid(True)
    plt.gca().set_aspect('equal')
    plt.show()

def reconstruct_grid_from_top_dct_components(dct_transform, stop_rank, start_rank=0, plot_distribution=False):
    """
    Reconstructs the firing rate grid from the top `n_components` DCT coefficients.
    
    Parameters:
    - dct_transform: 2D numpy array (the original DCT).
    - stop_rank: int, rank of the last component to include (exclusive).
    - start_rank: int, rank of the first component to include (default is 0).
    - plot_distribution: bool, if True, plots the sorted distribution of DCT coefficients' magnitudes.
    
    Returns:
    - reconstructed_grid: 2D numpy array of the reconstructed grid.
    """
    if start_rank > stop_rank - 1:
        raise ValueError("Choose a value for start_rank smaller than stop_rank.")
    
    # Step 1: Identify the top coefficients based on the magnitude
    flattened_indices = np.dstack(np.unravel_index(np.argsort(-np.abs(dct_transform).ravel()), dct_transform.shape))[0]
    
    # Plot the distribution of sorted coefficients' magnitudes if requested
    if plot_distribution:
        sorted_magnitudes = np.sort(np.abs(dct_transform).ravel())[::-1]
        plt.figure(figsize=(8, 5))
        plt.plot(sorted_magnitudes, marker='o', linestyle='-', color='b', alpha=0.7)
        plt.xlabel('Sorted Coefficient Rank')
        plt.ylabel('Coefficient Magnitude')
        plt.title('Distribution of DCT Coefficients\' Magnitudes')
        plt.yscale('log')  # Log scale can help visualize the drop-off
        plt.grid(True, which="both", ls="--")
        plt.show()

    # Step 2: Create a new DCT matrix to retain only the top components
    top_dct = np.zeros_like(dct_transform)
    for i in range(start_rank, min(stop_rank, len(flattened_indices))):
        x_idx, y_idx = flattened_indices[i]
        top_dct[x_idx, y_idx] = dct_transform[x_idx, y_idx]

    # Step 3: Reconstruct the grid using the inverse DCT (IDCT)
    reconstructed_grid = idct(idct(top_dct.T, norm='ortho').T, norm='ortho')

    print(reconstructed_grid.shape)

    return reconstructed_grid



def plot_reconstructed_firing_rate_grid(reconstructed_grid, x_min, x_max, y_min, y_max):
    """
    Plots the reconstructed firing rate grid.
    
    Parameters:
    - reconstructed_grid: 2D numpy array, the grid reconstructed from DCT components.
    - x_min, x_max, y_min, y_max: float, coordinates for plotting.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    cax = ax.imshow(reconstructed_grid.T, origin='lower', cmap='magma', extent=[x_min, x_max, y_min, y_max])
    ax.set_xlabel(r'X-coordinate ($\mu m$)')
    ax.set_ylabel(r'Y-coordinate ($\mu m$)')
    ax.set_title('Reconstructed Firing Rate Grid')
    cbar = fig.colorbar(cax, ax=ax, orientation='vertical')
    cbar.set_label('Averge Firing Rate Contribution (Hz)')
    plt.grid(True)
    plt.gca().set_aspect('equal')
    plt.show()

def extract_and_plot_spatial_frequencies_from_dct(dct_transform, n_components_stop, n_components_start, array_dims=(3800, 2100), axis='x', num_points=1000):
    """
    Reconstructs a spatial signal by summing cosine waves derived from the top `n_components` DCT coefficients.
    
    Parameters:
    - dct_transform: 2D numpy array of the DCT coefficients.
    - n_components: int, the number of top components to consider.
    - array_dims: tuple, the dimensions of the electrode array in micrometers (x, y).
    - axis: str, 'x' or 'y', indicating along which axis the reconstructed signal should be plotted.
    - num_points: int, number of points to use for plotting the reconstructed signal.
    
    Returns:
    - None (displays a plot)
    """
    # Step 1: Flatten the DCT coefficients and sort by magnitude to get the top `n_components`
    flattened_indices = np.dstack(np.unravel_index(np.argsort(-np.abs(dct_transform).ravel()), dct_transform.shape))[0]

    # Determine grid sizes
    num_x, num_y = dct_transform.shape
    length_x, length_y = array_dims
    
    # Prepare to sum cosine waves
    signal = np.zeros(num_points)
    x_space = np.linspace(0, length_x if axis == 'x' else length_y, num_points)

    # Step 2: Process each of the top `n_components`
    for i in range(n_components_start, n_components_stop):
        x_idx, y_idx = flattened_indices[i]
        coeff_value = dct_transform[x_idx, y_idx]

        # Determine frequency based on index
        freq_x = x_idx / num_x * (1 / (2 * (length_x / num_x)))  # Frequency in cycles per micrometer
        freq_y = y_idx / num_y * (1 / (2 * (length_y / num_y)))  # Frequency in cycles per micrometer

        # Calculate phase shift; phase is inferred from the sign of the DCT component
        amplitude = np.abs(coeff_value)
        phase = np.angle(coeff_value)  # Get the phase from the complex representation

        # Define cosine wave contribution
        if axis == 'x':
            wave = amplitude * np.cos(2 * np.pi * freq_x * x_space + phase)
        else:
            wave = amplitude * np.cos(2 * np.pi * freq_y * x_space + phase)

        # Add to the signal
        signal += wave

    max_value = np.max(signal)
    max_indices = np.where(signal == max_value)[0]
    max_positions = x_space[max_indices]

    print(f'Peaks at {max_positions}')
    # Step 3: Plot the reconstructed signal
    plt.figure(figsize=(10, 6))
    plt.plot(x_space, signal, 'b-', label=f'Reconstructed Signal along {axis}-axis')
    plt.xlabel(f'{axis}-axis Position (μm)')
    plt.ylabel('Amplitude (a.u.)')
    plt.title(f'Reconstructed Spatial Signal from Top DCT Components')
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_layout_grid_with_fr(spikes_data_list, layout_list, start_times, end_times):
    interpolated_grids = []
    for spikes_data, layout, start_time, end_time in zip(spikes_data_list, layout_list, start_times, end_times):
        grid, x_min, x_max, y_min, y_max = compute_interpolated_firing_rate_grid(spikes_data, layout, start_time, end_time)
        interpolated_grids.append(grid)

    # Plot the average DCT across all interpolated grids
    plot_average_dct_coefficients(interpolated_grids)

    n_components_stop = 3
    n_components_start = 0

    # Assuming `firing_rate_grid` is your 2D data array
    #reconstruct_from_top_dft_components_with_coeff_plot(interpolated_grids[0], n_components=20, array_dims=(3800, 2100), num_points=100)

    dct_avg_normalized = interpolated_grids_to_dct(interpolated_grids)
    reconstructed_grid = reconstruct_grid_from_top_dct_components(dct_avg_normalized, n_components_stop, n_components_start, plot_distribution=True)
    plot_reconstructed_firing_rate_grid(reconstructed_grid, x_min, x_max, y_min, y_max)

    extract_and_plot_spatial_frequencies_from_dct(dct_avg_normalized, n_components_stop, n_components_start, array_dims=(3800, 2100))

def plot_layout_grid_with_fr_each(spikes_data_list, layout_list, start_times, end_times):
    for spikes_data, layout, start_time, end_time in zip(spikes_data_list, layout_list, start_times, end_times):
        # Compute the interpolated grid
        grid, x_min, x_max, y_min, y_max = compute_interpolated_firing_rate_grid(spikes_data, layout, start_time, end_time)

        # Plot the interpolated grid
        plot_firing_rate_grid(grid, x_min, x_max, y_min, y_max)

        # Plot the DCT coefficients
        plot_dct_of_firing_rate_grid(grid)

        dct_transform = dct(dct(grid.T, norm='ortho').T, norm='ortho')
        reconstructed_grid = reconstruct_grid_from_top_dct_components(dct_transform, 1)
        plot_reconstructed_firing_rate_grid(reconstructed_grid, x_min, x_max, y_min, y_max)




def plot_log_avg_firing_rate_distribution(spikes_data_list, layout_list, start_times, end_times, bins=30):
    """
    Plots the frequency distribution of average Hz per electrode aggregated across recordings in logspace,
    fits a normal distribution to the log-transformed data, and prints the mean, std in Hz and p-value.
    
    Parameters:
    - spikes_data_list: List of spike data arrays for each recording.
    - layout_list: List of layout data corresponding to each recording.
    - start_times: List of start times for each recording.
    - end_times: List of end times for each recording.
    - bins: Number of bins to use in the histogram (default is 30).
    """
    # Dictionary to store total firing rate and counts for each electrode
    electrode_firing_rate = {}
    
    for spikes_data, layout, start_time, end_time in zip(spikes_data_list, layout_list, start_times, end_times):
        # Convert to DataFrame for easier processing
        spikes_df = pd.DataFrame(spikes_data)
        layout_df = pd.DataFrame(layout)
        
        # Filter spikes within the desired time window
        spikes_df_during = spikes_df[(spikes_df['time'] >= start_time) & (spikes_df['time'] <= end_time)]

        # Get unique electrodes
        unique_electrodes = layout_df['electrode'].unique()

        # Calculate average firing rate for each electrode
        for electrode in unique_electrodes:
            firing_times = spikes_df_during['time'][spikes_df_during['electrode'] == electrode].reset_index(drop=True)
            num_spikes = len(firing_times)
            time_duration = end_time - start_time

            # Calculate average Hz
            if time_duration > 0:
                avg_hz = num_spikes / time_duration
                if electrode in electrode_firing_rate:
                    electrode_firing_rate[electrode].append(avg_hz)
                else:
                    electrode_firing_rate[electrode] = [avg_hz]

    # Compute the average firing rate per electrode across recordings
    avg_firing_rates = [np.mean(rates) for rates in electrode_firing_rate.values()]

    # Apply log transformation to the data (ensure no zero or negative values)
    log_avg_firing_rates = np.log10(np.array(avg_firing_rates) + 1e-8)  # Small offset to avoid log(0)

    # Fit a normal distribution to the log-transformed data
    mu, sigma = norm.fit(log_avg_firing_rates)

    # Convert mean and std back to Hz
    mean_hz = 10 ** mu
    std_hz = (10 ** (mu + sigma)) - mean_hz
    print(f"Fitted normal distribution on log-transformed data: mean = {mu:.3f} (Hz: {mean_hz:.3f}), std dev = {sigma:.3f} (Hz: {std_hz:.3f})")

    # Perform Kolmogorov-Smirnov test to assess goodness of fit
    ks_stat, p_value = kstest(log_avg_firing_rates, 'norm', args=(mu, sigma))
    print(f"Kolmogorov-Smirnov Test p-value: {p_value:.5f}")

    # Perform Shapiro-Wilk test to assess goodness of fit
    ks_stat, p_value = shapiro(log_avg_firing_rates)
    print(f"Shapiro-Wilk Test p-value: {p_value:.5f}")

    # Generate the PDF for the fitted distribution
    x = np.linspace(min(log_avg_firing_rates), max(log_avg_firing_rates), 1000)
    pdf = norm.pdf(x, mu, sigma)

    # Plot the histogram of log-transformed average firing rates
    plt.figure(figsize=(10, 6))
    plt.hist(log_avg_firing_rates, bins=bins, color='blue', alpha=0.7, edgecolor='black', density=True)
    plt.xlabel('Log(Average Firing Rate) (log10 Hz)')
    plt.ylabel('Density')
    plt.title('Frequency Distribution of Log-Transformed Average Firing Rate with Normal Fit')
    plt.grid(True, linestyle='--', alpha=0.6)

    # Overlay the normal fit
    plt.plot(x, pdf, 'r-', linewidth=2, label='Normal Fit')
    plt.legend()

    plt.show()





########### LYAPUNOV STABILITY #############




def analyze_firing_activity(spikes_data, layout, start_time, end_time, bin_width=0.01):
    """
    Analyze firing activity before attempting Lyapunov calculation
    """
    print("\nAnalyzing firing activity...")

    # Get active electrodes
    active_electrodes = np.unique(spikes_data['electrode'])
    print(f"Total electrodes: {len(active_electrodes)}")
    
    # Calculate firing rates
    bins = np.arange(start_time, end_time + bin_width, bin_width)
    
    activity_stats = []
    for electrode in active_electrodes:
        # Get spike times for this electrode
        spike_times = spikes_data['time'][spikes_data['electrode'] == electrode]
        spike_times = spike_times[(spike_times >= start_time) & (spike_times <= end_time)]
        
        if len(spike_times) > 0:
            # Calculate firing rate
            counts, _ = np.histogram(spike_times, bins=bins)
            rates = counts / bin_width
            
            # Calculate statistics
            active_bins = np.sum(rates > 0)
            mean_rate = np.mean(rates)
            max_rate = np.max(rates)
            
            activity_stats.append({
                'electrode': electrode,
                'n_spikes': len(spike_times),
                'active_bins': active_bins,
                'percent_active': (active_bins / len(rates)) * 100,
                'mean_rate': mean_rate,
                'max_rate': max_rate
            })
    
    # Convert to DataFrame for analysis
    stats_df = pd.DataFrame(activity_stats)
    
    print("\nFiring statistics:")
    print(f"Mean spikes per electrode: {stats_df['n_spikes'].mean():.2f}")
    print(f"Mean active time percentage: {stats_df['percent_active'].mean():.2f}%")
    print(f"Mean firing rate: {stats_df['mean_rate'].mean():.2f} Hz")
    print(f"Max firing rate: {stats_df['max_rate'].max():.2f} Hz")
    
    # Identify electrodes with sufficient activity
    active_threshold = 0.1  # 0.1% of bins must have activity
    sufficiently_active = stats_df[stats_df['percent_active'] > active_threshold]
    print(f"\nElectrodes with >{active_threshold}% activity: {len(sufficiently_active)}")
    
    return sufficiently_active['electrode'].values
    
def smooth_ifr_trajectory(ifr_times, ifr_values, start_time, end_time, bin_width=0.01, sigma=0.05):
    """
    Convert step-function IFR into smoothed trajectory in log space
    
    Args:
        ifr_times: Array of times from calculate_ifr
        ifr_values: Array of IFR values from calculate_ifr
        start_time: Start time of analysis window
        end_time: End time of analysis window
        bin_width: Width of time bins (default 10ms)
        sigma: Standard deviation for Gaussian kernel in seconds
    """
    # Create regular time bins
    bins = np.arange(start_time, end_time + bin_width, bin_width)
    
    # Convert to log space, handling zeros
    min_rate = 0.00001  # minimum rate
    log_values = np.log10(ifr_values + min_rate)
    
    # Interpolate log IFR values onto regular grid
    interpolated = np.interp(bins, ifr_times, log_values)
    
    # Create Gaussian kernel
    kernel_width = int(4 * sigma / bin_width)  # 4 sigma on each side
    kernel_times = np.arange(-kernel_width, kernel_width + 1) * bin_width
    kernel = np.exp(-(kernel_times**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()  # normalize
    
    # Convolve with Gaussian kernel
    smoothed = np.convolve(interpolated, kernel, mode='same')
    
    # No need to convert back from log space - we'll keep it logarithmic
    # for the Lyapunov analysis
    
    return bins, smoothed

# Modify prepare_stability_analysis to use smoothed trajectories
def prepare_stability_analysis(spikes_data_list, layout_list, start_times, end_times, plot_heatmap=False):
    """
    Prepares data for stability analysis using smoothed IFR trajectories
    """
    all_stability_data = []
    
    for idx, (spikes_data, layout, start_time, end_time) in enumerate(zip(
        spikes_data_list, layout_list, start_times, end_times)):
        
        print(f"\nProcessing recording {idx+1}/{len(spikes_data_list)}")
        
        # First analyze firing activity
        active_electrodes = analyze_firing_activity(spikes_data, layout, start_time, end_time)
        print(f"\nProceeding with analysis using {len(active_electrodes)} active electrodes")
        
        # Calculate IFR
        ifr_data, _, _ = calculate_ifr(spikes_data, active_electrodes, start_time, end_time)
        
        # Create smoothed trajectories
        smoothed_data = {}
        time_bins = None  # Store time bins for heatmap
        heatmap_data = []  # Collect data for the heatmap
        
        for electrode in ifr_data:
            ifr_times, ifr_values = ifr_data[electrode]
            bins, smoothed = smooth_ifr_trajectory(ifr_times, ifr_values, start_time, end_time)
            smoothed_data[electrode] = (bins, smoothed)
            heatmap_data.append(smoothed)
            if time_bins is None:
                time_bins = bins
        
        if plot_heatmap:
            plt.figure(figsize=(12, 8))
            plt.imshow(
                np.array(heatmap_data),
                aspect='auto',
                extent=[time_bins[0], time_bins[-1], 0, len(active_electrodes)],
                cmap='hot',
                interpolation='nearest',
                origin='lower'
            )
            plt.colorbar(label="IFR (Hz)")
            plt.ylabel("Electrodes")
            plt.xlabel("Time (s)")
            plt.title(f"Heatmap of IFR Rates (Recording {idx+1})")
            plt.yticks(range(len(active_electrodes)), labels=active_electrodes)
            plt.show()
        
        # Rest of the processing remains the same, but use smoothed_data instead of ifr_data
        layout_df = pd.DataFrame(layout)
        coords_dict = {
            electrode: layout_df[layout_df['electrode'] == electrode][['x', 'y']].iloc[0].values 
            for electrode in active_electrodes
        }
        
        for i, ref_electrode in enumerate(active_electrodes):
            if ref_electrode in smoothed_data:
                ref_coords = coords_dict[ref_electrode]
                
                for target_electrode in active_electrodes[i+1:]:
                    if target_electrode in smoothed_data:
                        target_coords = coords_dict[target_electrode]
                        distance = np.sqrt(np.sum((ref_coords - target_coords)**2))
                        
                        pair_data = {
                            'recording_idx': idx,
                            'ref_electrode': ref_electrode,
                            'target_electrode': target_electrode,
                            'distance': distance,
                            'ref_ifr': smoothed_data[ref_electrode],
                            'target_ifr': smoothed_data[target_electrode]
                        }
                        all_stability_data.append(pair_data)
    
    return all_stability_data

def compute_lyapunov(embed1, embed2, max_points=1000, min_neighbors=5, max_neighbors=10, max_dt=10):
    """
    Added checks for numerical stability
    """
    min_length = min(len(embed1), len(embed2))
    
    # Random sampling
    if min_length > max_points:
        indices = np.random.choice(min_length - 5, max_points, replace=False)
        indices = np.sort(indices)
        embed1 = embed1[indices]
        embed2 = embed2[indices]
        min_length = max_points
    
    # Check embeddings before normalization
    #print("\nEmbedding statistics before normalization:")
    #print(f"Embed1 - min: {np.min(embed1):.6f}, max: {np.max(embed1):.6f}, mean: {np.mean(embed1):.6f}")
    #print(f"Embed2 - min: {np.min(embed2):.6f}, max: {np.max(embed2):.6f}, mean: {np.mean(embed2):.6f}")
    
    # Check for zero variance
    std1 = np.std(embed1, axis=0)
    std2 = np.std(embed2, axis=0)
    if np.any(std1 == 0) or np.any(std2 == 0):
        print("Warning: Zero variance in embeddings")
        return np.nan
    
    # Normalize embeddings with safety checks
    joint_embed = np.hstack([embed1, embed2])
    means = np.mean(joint_embed, axis=0)
    stds = np.std(joint_embed, axis=0)
    stds[stds == 0] = 1  # Prevent division by zero
    joint_embed = (joint_embed - means) / stds
    
    # Check for invalid values
    if np.any(np.isnan(joint_embed)) or np.any(np.isinf(joint_embed)):
        print("Warning: Invalid values in normalized embeddings")
        return np.nan
    
    # Compute distances
    distances = scipy.spatial.distance.pdist(joint_embed)
    distances = scipy.spatial.distance.squareform(distances)
    
    # Check if we have any non-zero distances
    if np.all(distances == 0):
        print("Warning: All distances are zero")
        return np.nan
    
    non_zero_distances = distances[distances > 0]
    if len(non_zero_distances) == 0:
        print("Warning: No non-zero distances found")
        return np.nan
        
    #print(f"\nDistance statistics:")
    #print(f"Min non-zero distance: {np.min(non_zero_distances):.6f}")
    #print(f"Mean distance: {np.mean(distances):.6f}")
    #print(f"Max distance: {np.max(distances):.6f}")
    
    epsilon = 0.2
    divergence_rates = []
    
    n_points_with_neighbors = 0
    n_points_with_valid_divergence = 0
    
    for i in range(min_length - max_dt):
        neighbors = np.where((distances[i] > 0) & (distances[i] < epsilon))[0]
        neighbors = neighbors[neighbors < min_length - max_dt]
        
        if len(neighbors) >= min_neighbors:
            n_points_with_neighbors += 1
            
            if len(neighbors) > max_neighbors:
                neighbors = neighbors[np.argsort(distances[i, neighbors])[:max_neighbors]]
            
            initial_distances = distances[i, neighbors]
            future_distances = np.array([distances[i + dt, neighbors] for dt in range(1, max_dt)])
            
            with np.errstate(divide='ignore', invalid='ignore'):
                ratios = future_distances / initial_distances[None, :]
                valid_mask = (ratios > 0)
                
                if np.any(valid_mask):
                    logs = np.log(ratios + 1e-10)
                    dt_array = np.arange(1, max_dt)[:, None]
                    divs = np.mean(logs * valid_mask, axis=1) / dt_array
                    valid_divs = divs[~np.isnan(divs)]
                    
                    if len(valid_divs) > 0:
                        divergence_rates.extend(valid_divs)
                        n_points_with_valid_divergence += 1
    
    #print(f"\nAnalysis statistics:")
    #print(f"Points with enough neighbors: {n_points_with_neighbors}/{min_length-max_dt}")
    #print(f"Points with valid divergence: {n_points_with_valid_divergence}/{min_length-max_dt}")
    
    if divergence_rates:
        mean_rate = np.mean(divergence_rates)
        #print(f"\nDivergence rates:")
        #print(f"Mean: {mean_rate:.6f}")
        #print(f"Min: {np.min(divergence_rates):.6f}")
        #print(f"Max: {np.max(divergence_rates):.6f}")
        return mean_rate
    else:
        #print("\nNo valid divergence rates found")
        return np.nan


def create_delay_vectors(time_series, tau, m):
    """
    Create delay vectors ensuring consistent length
    """
    N = len(time_series) - (m-1)*tau
    if N <= 0:
        return np.array([])
        
    vectors = np.zeros((N, m))
    for i in range(m):
        vectors[:, i] = time_series[i*tau : i*tau + N]
    
    return vectors

from typing import Optional

def bin_and_analyze_stability(
    all_stability_data,
    bin_size=200,
    max_distance=3200,
    tau=10,
    m=5,
    max_pairs_per_bin=20000,
    random_state: Optional[int] = 42,
    stratified: bool = True,
):
    print("\nStarting stability analysis...")
    
    distances = [pair['distance'] for pair in all_stability_data]
    bins = np.arange(0, max_distance, bin_size)
    print(f"Created {len(bins)-1} distance bins")
    
    print("Binning pairs...")
    binned_pairs = {i: [] for i in range(len(bins)-1)}
    for pair in all_stability_data:
        bin_idx = np.digitize(pair['distance'], bins) - 1
        if bin_idx in binned_pairs:
            binned_pairs[bin_idx].append(pair)
    
    print("Computing stability metrics...")
    stability_metrics = {}
    total_bins = len(binned_pairs)
    
    rng = np.random.default_rng(random_state)

    for bin_idx, pairs in binned_pairs.items():
        if len(pairs) > 0:
            print(f"\nProcessing bin {bin_idx+1}/{total_bins} ({len(pairs)} pairs)")
            bin_lyap = []
            # Select a subset of pairs to control runtime without biasing toward early recordings
            if len(pairs) > max_pairs_per_bin:
                if stratified:
                    # Sample approximately equal numbers from each recording present in this bin
                    by_rec = {}
                    for p in pairs:
                        by_rec.setdefault(p.get('recording_idx', -1), []).append(p)
                    n_rec = len(by_rec)
                    quota = max(1, max_pairs_per_bin // max(1, n_rec))
                    sampled = []
                    for rec_idx, plist in by_rec.items():
                        k = min(quota, len(plist))
                        idxs = rng.choice(len(plist), size=k, replace=False)
                        sampled.extend([plist[i] for i in idxs])
                    # If under target due to small groups, top-up with random remaining pairs
                    if len(sampled) < max_pairs_per_bin:
                        remaining = [p for p in pairs if p not in sampled]
                        k = min(max_pairs_per_bin - len(sampled), len(remaining))
                        if k > 0:
                            idxs = rng.choice(len(remaining), size=k, replace=False)
                            sampled.extend([remaining[i] for i in idxs])
                    pairs_to_process = sampled[:max_pairs_per_bin]
                else:
                    idxs = rng.choice(len(pairs), size=max_pairs_per_bin, replace=False)
                    pairs_to_process = [pairs[i] for i in idxs]
            else:
                pairs_to_process = list(pairs)

            for pair_idx, pair in enumerate(pairs_to_process):
                
                ref_times, ref_values = pair['ref_ifr']
                target_times, target_values = pair['target_ifr']
                
                
                if len(ref_values) >= m*tau+30 and len(target_values) >= m*tau+30:
                    ref_embedding = create_delay_vectors(ref_values, tau, m)
                    target_embedding = create_delay_vectors(target_values, tau, m)
                    
                    
                    max_analysis_points = 100
                    
                    if len(ref_embedding) > 0 and len(target_embedding) > 0:
                        lyap = compute_lyapunov(ref_embedding, target_embedding, max_points=max_analysis_points)
                        if not np.isnan(lyap):
                            bin_lyap.append(lyap)
            
            if bin_lyap:
                stability_metrics[bin_idx] = {
                    'distance': int((bins[bin_idx] + bins[bin_idx+1])/2),
                    'mean_lyap': np.mean(bin_lyap),
                    'std_lyap': np.std(bin_lyap),
                    'n_pairs': len(bin_lyap),
                    'raw_lyap': bin_lyap
                }
                print(f"Bin {bin_idx+1}: Mean Lyapunov = {np.mean(bin_lyap):.4f} ± {(1.96*np.std(bin_lyap)/np.sqrt(len(bin_lyap))):.4f} (n={len(bin_lyap)})")
            else:
                print(f"No valid Lyapunov exponents for bin {bin_idx+1}")
    
    print("Analysis complete!")
    return stability_metrics, bins


def plot_exponent_difference_heatmap(stability_metrics, bins, max_bins=35, exponent_cap=2, kde_points=200):
    """
    Create a heatmap showing differences in the distribution of Lyapunov exponents 
    from the first bin's KDE distribution. Distance on the x-axis, exponent on the y-axis.
    We limit to the first `max_bins` distance bins and cap the exponent range at `exponent_cap`.
    Uses a KDE for a smooth distribution estimate.

    Parameters
    ----------
    stability_metrics : dict
        Contains stability metrics and 'raw_lyap' for each bin.
    bins : array-like
        The bin edges for distance.
    max_bins : int, optional
        Maximum number of distance bins to include.
    exponent_cap : float, optional
        Maximum Lyapunov exponent value to visualize.
    kde_points : int, optional
        Number of points at which to evaluate the KDE.
    """

    # Sort stability_metrics by key (assuming they are bin indices)
    sorted_indices = sorted(stability_metrics.keys())
    # Take the first `max_bins` bins
    selected_indices = sorted_indices[:max_bins]

    # Extract distances and raw exponent values for the selected bins
    distances = [stability_metrics[i]['distance'] for i in selected_indices]
    raw_exponents_by_bin = [stability_metrics[i]['raw_lyap'] for i in selected_indices]

    # Determine exponent range
    all_exponents = np.concatenate(raw_exponents_by_bin)
    exponent_min = all_exponents.min()
    exponent_max = min(all_exponents.max(), exponent_cap)

    # Create a grid of exponent values at which we will evaluate the KDE
    exponent_values = np.linspace(exponent_min, exponent_max, kde_points)

    # Compute KDE for each bin
    kde_distributions = []
    for exps in raw_exponents_by_bin:
        if len(exps) > 1:
            kde = gaussian_kde(exps)
            kde_pdf = kde(exponent_values)
        else:
            # If there's only one or zero exps, the KDE is not well-defined; 
            # handle by setting a uniform-like very low density or skip
            kde_pdf = np.zeros_like(exponent_values)
        # Normalize (gaussian_kde should already produce a pdf that integrates to 1, 
        # but we can double-check by numerical integration if desired.)
        # For now, we trust gaussian_kde normalization.
        kde_distributions.append(kde_pdf)

    # The first bin is the reference
    reference_distribution = kde_distributions[0]

    # Compute differences
    distributions_diff = np.array([dist - reference_distribution for dist in kde_distributions])

    # Create a DataFrame: Rows = exponent bins (y), Columns = distances (x)
    df_diff = pd.DataFrame(distributions_diff.T, index=exponent_values, columns=distances)

    yticks = np.linspace(0.2, 2, 19)

    tick_positions = [np.argmin(np.abs(exponent_values - val)) for val in yticks]

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        df_diff,
        cmap='RdBu_r',
        center=0,
        xticklabels=np.round(df_diff.columns, 2),
        yticklabels=False,  # We'll set yticklabels manually
        cbar_kws={'label': 'Difference in Probability Density'},
        ax=ax
    )

    # Now set the yticks (row indices) and yticklabels (actual exponent values)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(np.round(yticks, 2))

    ax.set_xlabel('Distance (μm)')
    ax.set_ylabel('Lyapunov Exponent')
    ax.set_title('KDE-based Distribution Differences from First Bin')
    plt.tight_layout()
    plt.show()

def plot_stability_distributions(stability_metrics, bins):
    """
    Plot raw distributions and difference from average distribution using KDE.
    """
    # Setup figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Get data for all bins
    distances = [metrics['distance'] for metrics in stability_metrics.values()]
    raw = [metrics['raw_lyap'] for metrics in stability_metrics.values()]
    
    # For violin plots (raw distributions)
    data = []
    for distance, raw_values in zip(distances, raw):
        for value in raw_values:
            data.append({"Distance": distance, "Lyapunov": value})
    raw_df = pd.DataFrame(data)
    
    # Plot raw distributions
    max_lyap = 2
    sns.violinplot(data=raw_df, x="Distance", y="Lyapunov", ax=ax1)
    ax1.set_title('Raw Lyapunov Exponent Distributions')
    ax1.set_xlabel('Distance (μm)')
    ax1.set_ylabel('Lyapunov Exponent')
    ax1.set_ylim(top=max_lyap)
    
    # Calculate KDE for average distribution
    lyap_grid = np.linspace(min(raw_df['Lyapunov']), max(raw_df['Lyapunov']), 100)
    avg_kde = scipy.stats.gaussian_kde(raw_df['Lyapunov'])
    avg_density = avg_kde(lyap_grid)

    mean = np.mean(raw_df['Lyapunov'], axis=0)
    ci = 1.96 * np.std(raw_df['Lyapunov'], axis=0) / np.sqrt(len(raw_df['Lyapunov']))
    print(f'Average LE: {mean:.4} ± {ci:.4f}')
    
    # Calculate KDE differences for heatmap
    diffs = []
    for values in raw:
        if len(values) > 1:  # Need at least 2 points for KDE
            kde = scipy.stats.gaussian_kde(values)
            density = kde(lyap_grid)
            diff = density - avg_density
        else:
            diff = np.zeros_like(lyap_grid)
        diffs.append(diff)
    
    # Create heatmap with centered colormap
    diffs_array = np.array(diffs).T
    vmax = np.max(np.abs(diffs_array))
    im = ax2.imshow(diffs_array,
                    aspect='auto',
                    origin='lower',
                    extent=[min(distances), max(distances), 
                           min(lyap_grid), max_lyap],
                    cmap='RdBu_r',
                    vmin=-vmax,
                    vmax=vmax)
    
    plt.colorbar(im, ax=ax2, label='Difference from Average KDE')
    ax2.set_title('Difference from Average Distribution (KDE)')
    ax2.set_xlabel('Distance (μm)')
    ax2.set_ylabel('Lyapunov Exponent')
    
    plt.tight_layout()
    plt.show()

def _fdr_bh(pvals: np.ndarray) -> np.ndarray:
    p = np.array(pvals, dtype=float)
    mask = np.isfinite(p)
    m = np.sum(mask)
    if m == 0:
        return p
    order = np.argsort(p[mask])
    ranked = p[mask][order]
    adj = np.empty_like(ranked)
    prev = 1.0
    # Benjamini-Hochberg step-up
    for i in range(m - 1, -1, -1):
        adj_i = ranked[i] * m / (i + 1)
        prev = min(prev, adj_i)
        adj[i] = prev
    out = np.full_like(p, np.nan)
    out_idx = np.where(mask)[0][order]
    out[out_idx] = adj
    return out

def plot_stability_ci_with_significance(
    stability_metrics,
    bins,
    alpha: float = 0.05,
    correction: str = "fdr_bh",
    test: str = "welch",
):
    """
    Plot mean ± 95% CI of Lyapunov exponents per distance bin and annotate
    significance (two-sided) against the global average distribution.
    - test: 'welch' (two-sample Welch t-test on means) or 'ks' (KS test on distributions)
    - correction: 'fdr_bh', 'bonferroni', or 'none'
    """
    # Collect per-bin arrays and distances in order of bin index
    bin_indices = sorted(stability_metrics.keys())
    distances = [stability_metrics[i]["distance"] for i in bin_indices]
    arrays = [np.asarray(stability_metrics[i]["raw_lyap"], dtype=float) for i in bin_indices]

    means = np.array([np.nan if len(a) == 0 else np.mean(a) for a in arrays], dtype=float)
    ses = np.array([
        np.nan if len(a) < 2 else np.std(a, ddof=1) / np.sqrt(len(a)) for a in arrays
    ], dtype=float)
    cis = 1.96 * ses

    # Build global pool excluding each bin for testing
    pvals = []
    for i, a in enumerate(arrays):
        if len(a) < 2:
            pvals.append(np.nan)
            continue
        others = [arrays[j] for j in range(len(arrays)) if j != i and len(arrays[j]) > 1]
        if not others:
            pvals.append(np.nan)
            continue
        pooled = np.concatenate(others)
        if test == "welch":
            from scipy.stats import ttest_ind
            _, p = ttest_ind(a, pooled, equal_var=False)
        elif test == "ks":
            from scipy.stats import ks_2samp
            _, p = ks_2samp(a, pooled, alternative="two-sided", mode="auto")
        else:
            raise ValueError("Unsupported test; use 'welch' or 'ks'.")
        pvals.append(float(p))
    pvals = np.array(pvals, dtype=float)

    # Multiple testing correction
    if correction == "fdr_bh":
        adj_p = _fdr_bh(pvals)
    elif correction == "bonferroni":
        m = np.sum(np.isfinite(pvals))
        adj_p = np.minimum(1.0, pvals * m)
    else:
        adj_p = pvals

    def star(p):
        if not np.isfinite(p):
            return ""
        return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < alpha else ""

    stars = [star(p) for p in adj_p]

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.errorbar(distances, means, yerr=cis, fmt="o-", capsize=3, color="C0", label="Mean ± 95% CI")
    ax.set_xlabel("Distance (μm)")
    ax.set_ylabel("Lyapunov Exponent")
    ax.set_title(
        f"Lyapunov Mean ± 95% CI per Distance Bin (test={test}, correction={correction})"
    )
    # Annotate stars above upper CI
    if np.any(np.isfinite(means)):
        y_hi = means + cis
        y_min = np.nanmin(means - cis)
        y_max = np.nanmax(means + cis)
        y_off = 0.05 * (y_max - y_min if np.isfinite(y_max - y_min) and (y_max - y_min) > 0 else 1.0)
        for x, y, s in zip(distances, y_hi, stars):
            if s:
                ax.text(x, y + y_off, s, ha="center", va="bottom", fontsize=10, color="k")
    ax.legend()
    plt.tight_layout()
    plt.show()

def compute_synergy_coefficients(distances, hz=46):
    """
    Compute synergy coefficients for given distances
    """
    # Parameters
    v_eph = 0.1  # mm/ms = 100 μm/ms
    v_ax = 0.45  # mm/ms = 450 μm/ms
    lambda_eph = 100000  # Large value to make exponential negligible
    
    # Convert distances to numpy array if not already
    distances = np.array(distances)
    
    # Compute correlation function
    synergy_coeffs = correlation_function(distances, hz, v_eph, v_ax, lambda_eph)
    
    return synergy_coeffs






if __name__ == "__main__":

    file_info = [
        ('2407', 'control_0.raw.h5', 0, 1800, 0),
        ('2407', 'control_1.raw.h5', 0, 1800, 0),
        ('2407', 'control_2.raw.h5', 0, 1800, 0),
        ('2407', 'control_0.raw.h5', 0, 600, 1),
        ('2407', 'control_1.raw.h5', 0, 1800, 1),
        ('2407', 'control_2.raw.h5', 0, 1800, 1),
        ('240725', '50hz.raw.h5', 0, 600, 2),
        ('240726', '50hz.raw.h5', 0, 600, 2),
        ('2407', '50hz_0.raw.h5', 0, 600, 2),
        ('2407', '50hz_1.raw.h5', 0, 600, 2),
        ('240725', '50hz.raw.h5', 0, 600, 3), #13260 instead of 13360
        ('240726', '50hz.raw.h5', 0, 600, 3),
        ('240725', '100hz.raw.h5', 0, 600, 4),
        ('240726', '100hz.raw.h5', 0, 600, 4),
        ('2407', '100hz_0.raw.h5', 0, 600, 4),
        ('2407', '100hz_1.raw.h5', 0, 600, 4),
        ('2407', '100hz_2.raw.h5', 0, 600, 4),
        ('2407', '100hz_1.raw.h5', 0, 1800, 5),
        ('2407', '100hz_0.raw.h5', 0, 600, 5),
        ('2407', '100hz_2.raw.h5', 0, 1800, 5),
    ]

    file_info = [
        ('2407', 'control_0.raw.h5', 0, 600, 0),
        #('2407', 'control_1.raw.h5', 0, 1800, 0),
        #('2407', 'control_2.raw.h5', 0, 1800, 0),
        #('2407', 'control_0.raw.h5', 0, 600, 1),
        #('2407', 'control_1.raw.h5', 0, 1800, 1),
        ('2407', 'control_2.raw.h5', 0, 600, 1),
        ('240725', '50hz.raw.h5', 0, 600, 2),
        #('240726', '50hz.raw.h5', 0, 600, 2),
        #('2407', '50hz_0.raw.h5', 0, 600, 2),
        #('2407', '50hz_1.raw.h5', 0, 600, 2),
        ('240725', '50hz.raw.h5', 0, 600, 3), #13260 instead of 13360
        #('240726', '50hz.raw.h5', 0, 600, 3),
        ('240725', '100hz.raw.h5', 0, 600, 4),
        #('240726', '100hz.raw.h5', 0, 600, 4),
        #('2407', '100hz_0.raw.h5', 0, 600, 4),
        #('2407', '100hz_1.raw.h5', 0, 600, 4),
        #('2407', '100hz_2.raw.h5', 0, 600, 4),
        ('2407', '100hz_1.raw.h5', 0, 600, 5),
        #('2407', '100hz_0.raw.h5', 0, 600, 5),
        #('2407', '100hz_2.raw.h5', 0, 1800, 5),
    ]

    file_info = [
        ('2407', 'control_0.raw.h5', 0, 600, 0),
        #('2407', 'control_1.raw.h5', 0, 1800, 0),
        #('2407', 'control_2.raw.h5', 0, 1800, 0),
        #('2407', 'control_0.raw.h5', 0, 600, 1),
        #('2407', 'control_1.raw.h5', 0, 1800, 1),
        ('2407', 'control_2.raw.h5', 0, 600, 1),
        ('240725', '50hz.raw.h5', 0, 600, 2),
        #('240726', '50hz.raw.h5', 0, 600, 2),
        #('2407', '50hz_0.raw.h5', 0, 600, 2),
        #('2407', '50hz_1.raw.h5', 0, 600, 2),
        ('240725', '50hz.raw.h5', 0, 600, 3), #13260 instead of 13360
        #('240726', '50hz.raw.h5', 0, 600, 3),
        ('240725', '100hz.raw.h5', 0, 600, 4),
        #('240726', '100hz.raw.h5', 0, 600, 4),
        #('2407', '100hz_0.raw.h5', 0, 600, 4),
        #('2407', '100hz_1.raw.h5', 0, 600, 4),
        #('2407', '100hz_2.raw.h5', 0, 600, 4),
        ('2407', '100hz_1.raw.h5', 0, 600, 5),
        #('2407', '100hz_0.raw.h5', 0, 600, 5),
        #('2407', '100hz_2.raw.h5', 0, 1800, 5),
    ]

    start_s = 2400
    end_s = 3000


    # all divs for one well
    well = 4
    file_info = [(div, start_s, end_s, well) for div in [30, 33, 34, 35, 36, 37, 38, 40, 41]]

    # all divs for all wells
    file_info = [(div, start_s, end_s, well) for div, well in zip([30, 33, 34, 35, 36, 37, 38, 40, 41], range(5))]

    file_info = [
        #('2407', 'control_0.raw.h5', 0, 600, 0),
        #('2407', 'control_1.raw.h5', 0, 1800, 0),
        #('2407', 'control_2.raw.h5', 0, 1800, 0),
        #('2407', 'control_0.raw.h5', 0, 600, 1),
        #('2407', 'control_1.raw.h5', 0, 1800, 1),
        #('2407', 'control_2.raw.h5', 0, 600, 1),
        #('240725', '50hz.raw.h5', 0, 600, 2),
        #('240726', '50hz.raw.h5', 0, 600, 2),
        #('2407', '50hz_0.raw.h5', 0, 600, 2),
        #('2407', '50hz_1.raw.h5', 0, 600, 2),
        #('240725', '50hz.raw.h5', 0, 600, 3), #13260 instead of 13360
        #('240726', '50hz.raw.h5', 0, 600, 3),
        #('240725', '100hz.raw.h5', 0, 600, 4),
        #('240726', '100hz.raw.h5', 0, 600, 4),
        #('2407', '100hz_0.raw.h5', 0, 600, 4),
        #('2407', '100hz_1.raw.h5', 0, 600, 4),
        #('2407', '100hz_2.raw.h5', 0, 600, 4),
        #('2407', '100hz_1.raw.h5', 0, 600, 5),
        #('2407', '100hz_0.raw.h5', 0, 600, 5),
        #('2407', '100hz_2.raw.h5', 0, 1800, 5),
    ]

    file_info = [
        ('2407', 'control_0.raw.h5', 0, 1800, 0),
        ('2407', 'control_1.raw.h5', 0, 1800, 0),
        ('2407', 'control_2.raw.h5', 0, 1800, 0),
        ('2407', 'control_0.raw.h5', 0, 600, 1),
        ('2407', 'control_1.raw.h5', 0, 1800, 1),
        ('2407', 'control_2.raw.h5', 0, 1800, 1),
        ('240725', '50hz.raw.h5', 0, 600, 2),
        ('240726', '50hz.raw.h5', 0, 600, 2),
        ('2407', '50hz_0.raw.h5', 0, 600, 2),
        ('2407', '50hz_1.raw.h5', 0, 600, 2),
        ('240725', '50hz.raw.h5', 0, 600, 3), #13260 instead of 13360
        ('240726', '50hz.raw.h5', 0, 600, 3),
        ('240725', '100hz.raw.h5', 0, 600, 4),
        ('240726', '100hz.raw.h5', 0, 600, 4),
        ('2407', '100hz_0.raw.h5', 0, 600, 4),
        ('2407', '100hz_1.raw.h5', 0, 600, 4),
        ('2407', '100hz_2.raw.h5', 0, 600, 4),
        ('2407', '100hz_1.raw.h5', 0, 1800, 5),
        ('2407', '100hz_0.raw.h5', 0, 600, 5),
        ('2407', '100hz_2.raw.h5', 0, 1800, 5),
    ]

    #sf, spikes_data_list, layout_list, start_times, end_times = load_spikes_npz(file_info)
    sf, spikes_data_list, layout_list, start_times, end_times = load_spikes_data(file_info)

    print(spikes_data_list[0]['time'])
    

    #selected_electrodes = assign_n_proximate_electrodes(layout_list[0], ref_electrode, 10)
    selected_electrodes = get_activity_sorted_electrodes([spikes_data_list[0]], 10, 410, 0, 600)
    #plot_assigned_layout(selected_electrodes, layout_list[2])
    max_electrodes = 200

    # Start, stop and step size in ms for cofiring calculation
    start_ms = -20
    stop_ms = 20
    step_ms = 10
    window_size = int(step_ms*sf/1000)
    delays = np.linspace(start_ms, stop_ms, int((stop_ms-start_ms)/step_ms)+1)

    #plot_log_avg_firing_rate_distribution(spikes_data_list, layout_list, start_times, end_times, bins=30)
    #plot_layout_grid_with_fr_pooled(spikes_data_list, layout_list, start_times, end_times)
    #plot_avg_cofiring_heatmap(spikes_data_list, layout_list, selected_electrodes,  start_times, end_times, window_size, delays, normalize=False)
    #create_spike_distance_theta_gif(spikes_data_list, layout_list, selected_electrodes, start_times[0], end_times[0], window_size, delays, normalize=False)
    #create_spike_distance_grid_gif(spikes_data_list, layout_list, selected_electrodes, start_times[0], end_times[0], window_size, delays, normalize=False)
    #plot_avg_cofiring_models(spikes_data_list, layout_list, start_times, end_times, max_electrodes, log=False)
    #plot_avg_firing_rate_models(spikes_data_list, layout_list, start_times, end_times, max_electrodes, log=False)
    #get_ifr_stats(spikes_data_list, layout_list, selected_electrodes, start_times, end_times, log=False)
    #plot_distance_hist(spikes_data_list, layout_list, start_times, end_times, max_electrodes, finite_size_correction=True)

    #plot_layout_grid_with_fr_each(spikes_data_list, layout_list, start_times, end_times)
    #plot_layout_grid_with_fr(spikes_data_list, layout_list, start_times, end_times)
    
    all_stability_data = prepare_stability_analysis(spikes_data_list, layout_list, start_times, end_times)
    stability_metrics, bins = bin_and_analyze_stability(all_stability_data)
    plot_stability_distributions(stability_metrics, bins)
    plot_stability_ci_with_significance(stability_metrics, bins)
