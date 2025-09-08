import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import numpy as np
import pandas as pd
import copy
import re
import os
import imageio
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize, LogNorm
from matplotlib.cm import ScalarMappable, get_cmap
from ipywidgets import interact, widgets
from scipy.stats import powerlaw, gaussian_kde, binned_statistic, sem, t
from helper_functions import load_spikes

def correlation_function(r, lambda_eph, f, v_eph):
    return np.exp(-(r / lambda_eph)**3) * np.cos(4 * np.pi * f * r / v_eph)

def plot_spike_distance_scatter_discounted(spikes_data_list, layout_list, stimulated_electrode, start_time, end_time):
    cumulative_firing_rates = {}
    electrode_counts = {}
    electrode_distances = {}

    for spikes_data, layout in zip(spikes_data_list, layout_list):
        spikes_df = pd.DataFrame(spikes_data)
        layout_df = pd.DataFrame(layout)

        # Ensure the stimulated electrode is in the layout
        if stimulated_electrode not in layout_df['electrode'].values:
            raise ValueError(f"No data found for stimulated electrode: {stimulated_electrode}")
        stimulated_coords = layout_df.loc[layout_df['electrode'] == stimulated_electrode, ['x', 'y']].iloc[0]

        # Calculate distances from each electrode to the stimulated electrode
        layout_df['distance'] = np.sqrt((layout_df['x'] - stimulated_coords['x'])**2 + 
                                        (layout_df['y'] - stimulated_coords['y'])**2)
        
        # Remove entries with zero distance
        layout_df = layout_df[layout_df['distance'] > 0]

        # Map channels to distances and electrodes
        channel_to_distance = layout_df.set_index('channel')['distance'].to_dict()
        channel_to_electrode = layout_df.set_index('channel')['electrode'].to_dict()
        
        spikes_df['distance'] = spikes_df['channel'].map(channel_to_distance)
        spikes_df['electrode'] = spikes_df['channel'].map(channel_to_electrode)

        # Handling NaNs by dropping them
        spikes_df = spikes_df.dropna(subset=['distance', 'electrode'])

        # Filter data within the specified time range
        spikes_df_during = spikes_df[(spikes_df['time'] >= start_time) & (spikes_df['time'] <= end_time)]
        
        # Filter for the time range previous to stimulation
        spikes_df_pre = spikes_df[spikes_df['time'] < start_time]

        # Calculate firing rates (Hz) for both periods per electrode
        duration_pre = start_time
        duration_during = end_time - start_time
        
        firing_rates_pre = spikes_df_pre.groupby('electrode').size() / duration_pre
        firing_rates_during = spikes_df_during.groupby('electrode').size() / duration_during
        
        # Adjust firing rates by discounting pre-stimulation rates
        firing_rates_discounted = firing_rates_during - firing_rates_pre.reindex(firing_rates_during.index, fill_value=0)
        
        # Accumulate discounted firing rates and counts for each electrode
        for electrode, discounted_rate in firing_rates_discounted.items():
            if discounted_rate < 0:
                continue  # Skip negative values
            if electrode in cumulative_firing_rates:
                cumulative_firing_rates[electrode].append(discounted_rate)
                electrode_counts[electrode] += 1
            else:
                cumulative_firing_rates[electrode] = [discounted_rate]
                electrode_counts[electrode] = 1
                electrode_distances[electrode] = layout_df.loc[layout_df['electrode'] == electrode, 'distance'].values[0]

    # Prepare data for binned statistics
    all_distances = np.array(list(electrode_distances.values()))
    all_discounted_firing_rates = np.array([np.mean(rates) for rates in cumulative_firing_rates.values()])

    # Define bins logarithmically
    bins = np.logspace(np.log10(min(all_distances)), np.log10(max(all_distances)), num=25)

    # Compute means and standard errors for each bin
    bin_means, bin_edges, binnumber = binned_statistic(all_distances, all_discounted_firing_rates, statistic='mean', bins=bins)
    bin_counts, _, _ = binned_statistic(all_distances, all_discounted_firing_rates, statistic='count', bins=bins)
    bin_stds, _, _ = binned_statistic(all_distances, all_discounted_firing_rates, statistic='std', bins=bins)

    # Calculate confidence intervals
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_sems = bin_stds / np.sqrt(bin_counts)
    ci = bin_sems * t.ppf((1 + 0.95) / 2., bin_counts - 1)

    # Create a meshgrid for the background
    x_fit = np.linspace(0, max(all_distances), 500)
    y = np.linspace(0, 1000, 500)  # Dummy y-axis for gradient representation
    X, Y = np.meshgrid(x_fit, y)

    # Parameters for the correlation function
    lambda_eph = 2  # Example value
    f = 1 # Example value
    v_eph = 0.1 # Example value

    # Calculate the correlation values
    R = correlation_function(X/1000, lambda_eph, f, v_eph)

    # Normalize the correlation values to the range [0, 1] for color mapping
    norm = Normalize(vmin=-1, vmax=1)

    # Create the background gradient
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap('RdBu')  # Blue to white to red
    background = ax.imshow(R, aspect='auto', cmap=cmap, norm=norm,
                           extent=[bins.min(), x_fit.max(), y.min(), y.max()], alpha=1)

    # Add a colorbar to indicate correlation values
    cbar = plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, alpha = 1)
    cbar.set_label('Correlation')

    # Plot the scatter plot with confidence intervals
    ax.errorbar(bin_centers, bin_means, yerr=ci, fmt='o', ecolor='r', capsize=5, label='Mean Discounted Firing Rate with CI')
    #ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Distance from Stimulated Electrode ($\mu m$)')
    ax.set_ylabel('Average Firing Rate Increase (Hz)')
    ax.set_title('Average Firing Rate Increase over Baseline by Distance from Stimulation Site')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend()
    plt.show()

def plot_layout_grid_with_fr_pooled(spikes_data_list, layout_list, start_times, end_times, stimulated_electrode=None):
    pooled_firing_rates = {}
    electrode_counts = {}
    combined_layout = {'electrode': [], 'x': [], 'y': [], 'channel': []}

    for layout in layout_list:
        combined_layout['electrode'].extend(layout['electrode'])
        combined_layout['x'].extend(layout['x'])
        combined_layout['y'].extend(layout['y'])
        combined_layout['channel'].extend(layout['channel'])

    combined_layout_df = pd.DataFrame(combined_layout).drop_duplicates(subset='electrode').reset_index(drop=True)
    
    for spikes_data, layout, start_time, end_time in zip(spikes_data_list, layout_list, start_times, end_times):
        spikes_df = pd.DataFrame(spikes_data)
        layout_df = pd.DataFrame(layout)

        # Ensure the stimulated electrode is in the layout
        if stimulated_electrode not in layout_df['electrode'].values:
            print(f"No data found for stimulated electrode: {stimulated_electrode}")
        
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

        # Accumulate firing rates and counts
        for electrode, rate in firing_rates.items():
            if electrode in pooled_firing_rates:
                pooled_firing_rates[electrode] += rate
                electrode_counts[electrode] += 1
            else:
                pooled_firing_rates[electrode] = rate
                electrode_counts[electrode] = 1

    # Calculate the average firing rate for each electrode
    average_firing_rates = {electrode: pooled_firing_rates[electrode] / electrode_counts[electrode] for electrode in pooled_firing_rates}

    # Prepare the layout map and firing rates
    map_electrodes = [electrode for electrode in combined_layout_df['electrode'] if electrode in average_firing_rates]
    map_firing_rates = [average_firing_rates[electrode] for electrode in map_electrodes]

    # Filter the map to only include electrodes with firing rates
    indices_to_keep = [i for i, electrode in enumerate(combined_layout_df['electrode']) if electrode in average_firing_rates]
    map = {}
    for key in combined_layout_df.columns:
        map[key] = combined_layout_df[key].iloc[indices_to_keep].tolist()

    # Define the grid
    grid_size = 50
    x_n = int(np.ceil(max(map['x'])/grid_size)) - 1
    y_n = int(np.ceil(max(map['y'])/grid_size)) - 1

    x_min, x_max = 1, x_n * grid_size
    y_min, y_max = 1, y_n * grid_size
    x_bins = np.linspace(x_min, x_max, x_n+1)
    y_bins = np.linspace(y_min, y_max, y_n+1)
    print(y_bins)

    # Calculate the average firing rate for each grid cell
    grid_firing_rates = np.zeros((x_n, y_n))
    grid_counts = np.zeros((x_n, y_n))

    for i in range(len(map['x'])):
        x_idx = np.digitize(map['x'][i], x_bins) - 1
        y_idx = np.digitize(map['y'][i], y_bins) - 1
        if 0 <= x_idx < x_n and 0 <= y_idx < y_n:
            grid_firing_rates[x_idx, y_idx] += map_firing_rates[i]
            grid_counts[x_idx, y_idx] += 1
    
     
    # Calculate the average firing rate for each cell
    with np.errstate(invalid='ignore', divide='ignore'):
        grid_avg_firing_rates = np.divide(grid_firing_rates, grid_counts, where=grid_counts != 0)
    # Exclude invalid values for normalization
    valid_values = grid_avg_firing_rates[(grid_avg_firing_rates > 0.0001)]
    if valid_values.size == 0:
        raise ValueError("No valid firing rates available for normalization.")
    vmin = np.nanmin(valid_values)
    vmax = np.nanmax(valid_values)
    print(vmin, vmax)

    # Plot the grid with average firing rates
    fig, ax = plt.subplots(figsize=(10, 6))
    norm = LogNorm(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap('magma')
    cax = ax.imshow(grid_avg_firing_rates.T, origin='lower', cmap=cmap, norm=norm,
                    extent=[x_min, x_max, y_min, y_max])
    ax.set_xlabel('X-coordinate ($\mu m$)')
    ax.set_ylabel('Y-coordinate ($\mu m$)')
    ax.set_title('Average Firing Rate')
    ax.set_facecolor('black')

    # Highlight the electrode with a red circle if it exists in the map
    if stimulated_electrode in map['electrode']:
        index = map['electrode'].index(stimulated_electrode)
        ax.scatter(map['x'][index], map['y'][index], s=200, c='none', edgecolors='red', label='Stimulated Electrode')
        ax.legend(loc='upper right')

    # Create a subplot for the colorbar
    cbar = fig.colorbar(cax, ax=ax, orientation='vertical')
    cbar.set_label('Average Firing Rate (Hz)')

    plt.grid(True)
    plt.gca().set_aspect('equal')
    plt.show()

def plot_inter_spike_interval_scatter(spikes_data_list, layout_list, stimulated_electrode, start_time, end_time):
    # Combine spikes_data and layout from all wells
    combined_spikes_data_df = pd.concat([pd.DataFrame(spikes_data) for spikes_data in spikes_data_list], ignore_index=True)
    combined_layout_df = pd.concat([pd.DataFrame(layout) for layout in layout_list], ignore_index=True)
    
    # Filter data into three time intervals
    spikes_df_pre = combined_spikes_data_df[combined_spikes_data_df['time'] < start_time]
    spikes_df_during = combined_spikes_data_df[(combined_spikes_data_df['time'] >= start_time) & (combined_spikes_data_df['time'] <= end_time)]
    spikes_df_post = combined_spikes_data_df[combined_spikes_data_df['time'] > end_time]
    
    # Create a DataFrame from the layout for easy manipulation
    layout_df = combined_layout_df
    
    # Ensure the stimulated electrode is in the layout
    if stimulated_electrode not in layout_df['electrode'].values:
        raise ValueError(f"No data found for stimulated electrode: {stimulated_electrode}")
    stimulated_coords = layout_df.loc[layout_df['electrode'] == stimulated_electrode, ['x', 'y']].iloc[0]
    
    # Calculate distances from each electrode to the stimulated electrode
    layout_df['distance'] = np.sqrt((layout_df['x'] - stimulated_coords['x'])**2 + 
                                    (layout_df['y'] - stimulated_coords['y'])**2)
    
    # Remove entries with zero distance
    layout_df = layout_df[layout_df['distance'] > 0]
    
    # Map channels to distances
    channel_to_distance = layout_df.set_index('channel')['distance'].to_dict()
    spikes_df_pre['y_distance'] = spikes_df_pre['channel'].map(channel_to_distance)
    spikes_df_during['y_distance'] = spikes_df_during['channel'].map(channel_to_distance)
    spikes_df_post['y_distance'] = spikes_df_post['channel'].map(channel_to_distance)
    
    # Handling NaNs by dropping them
    spikes_df_pre = spikes_df_pre.dropna(subset=['y_distance'])
    spikes_df_during = spikes_df_during.dropna(subset=['y_distance'])
    spikes_df_post = spikes_df_post.dropna(subset=['y_distance'])
    
    # Define bins logarithmically
    min_dist = min(spikes_df_during['y_distance'].min(), spikes_df_pre['y_distance'].min(), spikes_df_post['y_distance'].min())
    max_dist = max(spikes_df_during['y_distance'].max(), spikes_df_pre['y_distance'].max(), spikes_df_post['y_distance'].max())
    bins = np.logspace(np.log10(min_dist), np.log10(max_dist), num=40)
    
    # Function to calculate ISI and bin statistics
    def calculate_isi_stats(spikes_df):
        spikes_df['isi'] = np.nan
        for channel in spikes_df['channel'].unique():
            channel_spikes = spikes_df[spikes_df['channel'] == channel]
            channel_spikes = channel_spikes.sort_values(by='time')
            inter_spike_intervals = np.diff(channel_spikes['time'].values)
            spikes_df.loc[channel_spikes.index[:-1], 'isi'] = inter_spike_intervals
        
        means = []
        std_errors = []
        bin_centers = []
        for i in range(len(bins) - 1):
            bin_data = spikes_df[(spikes_df['y_distance'] > bins[i]) & (spikes_df['y_distance'] <= bins[i + 1])]
            bin_isi = bin_data['isi'].dropna()
            if len(bin_isi) > 0:
                means.append(bin_isi.mean())
                std_errors.append(bin_isi.sem())
                bin_centers.append(0.5 * (bins[i] + bins[i + 1]))
        
        return bin_centers, means, std_errors

    # Calculate ISI stats for each interval
    bin_centers_pre, means_pre, std_errors_pre = calculate_isi_stats(spikes_df_pre)
    bin_centers_during, means_during, std_errors_during = calculate_isi_stats(spikes_df_during)
    bin_centers_post, means_post, std_errors_post = calculate_isi_stats(spikes_df_post)
    
    # Plot the mean ISI with error bars
    plt.figure(figsize=(10, 6))
    plt.errorbar(bin_centers_pre, means_pre, yerr=std_errors_pre, fmt='o', capsize=5, label='Before Stimulation', alpha=0.3)
    plt.errorbar(bin_centers_during, means_during, yerr=std_errors_during, fmt='o', capsize=5, label='During Stimulation')
    plt.errorbar(bin_centers_post, means_post, yerr=std_errors_post, fmt='o', capsize=5, label='After Stimulation')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Distance from Stimulated Electrode (micrometers)')
    plt.ylabel('Mean Inter-Spike Interval (seconds)')
    plt.title('Mean Inter-Spike Interval by Distance')
    plt.legend()
    plt.grid(True)
    plt.show()

def calculate_coincidence_proportions(spikes_df, stim_times, window_size=0.001, delay=0.0, stimulated_electrode=None):
    coincidence_counts = {electrode: 0 for electrode in spikes_df['electrode'].unique() if electrode != stimulated_electrode}
    total_stimulations = len(stim_times)

    for stim_time in stim_times:
        window_start = stim_time + delay
        window_end = stim_time + window_size + delay

        coinciding_spikes = spikes_df[(spikes_df['time'] >= window_start) & (spikes_df['time'] <= window_end)]
        
        for electrode, group in coinciding_spikes.groupby('electrode'):
            if electrode != stimulated_electrode:
                coincidence_counts[electrode] += len(group)

    proportions = {electrode: (coincidence_counts[electrode] / total_stimulations)
                   for electrode in coincidence_counts}
    
    return proportions

def plot_spike_distance_coincidence_heatmap(spikes_data_list, event_list, layout_list, stimulated_electrode, start_time, end_time, window_size=1, delays=np.linspace(-1, 3, 41)):
    coincidence_proportions_by_delay = {delay: {} for delay in delays}
    electrode_distances = {}

    for spikes_data, event_data, layout in zip(spikes_data_list, event_list, layout_list):
        spikes_df = pd.DataFrame(spikes_data)
        layout_df = pd.DataFrame(layout)

        # Ensure the stimulated electrode is in the layout
        if stimulated_electrode not in layout_df['electrode'].values:
            raise ValueError(f"No data found for stimulated electrode: {stimulated_electrode}")
        stimulated_coords = layout_df.loc[layout_df['electrode'] == stimulated_electrode, ['x', 'y']].iloc[0]

        # Calculate distances from each electrode to the stimulated electrode
        layout_df['distance'] = np.sqrt((layout_df['x'] - stimulated_coords['x'])**2 + 
                                        (layout_df['y'] - stimulated_coords['y'])**2)

        # Map channels to distances and electrodes
        channel_to_distance = layout_df.set_index('channel')['distance'].to_dict()
        channel_to_electrode = layout_df.set_index('channel')['electrode'].to_dict()
        
        spikes_df['distance'] = spikes_df['channel'].map(channel_to_distance)
        spikes_df['electrode'] = spikes_df['channel'].map(channel_to_electrode)

        # Handling NaNs by dropping them
        spikes_df = spikes_df.dropna(subset=['distance', 'electrode'])

        # Filter data within the specified time range
        spikes_df_during = spikes_df[(spikes_df['time'] >= start_time) & (spikes_df['time'] <= end_time)]
        
        # Extract the stimulation times from event_data and select every second entry starting from index 1
        stim_times = event_data['time'][1:-1:2]
        
        # Calculate coincidence proportions for each electrode and each delay
        for delay in delays:
            delay_sec = delay / 1000  # Convert ms to seconds
            proportions = calculate_coincidence_proportions(spikes_df_during, stim_times, window_size=window_size / 10000, delay=delay_sec, stimulated_electrode=stimulated_electrode)
            
            for electrode, proportion in proportions.items():
                if electrode == stimulated_electrode:
                    continue

                if electrode in coincidence_proportions_by_delay[delay]:
                    coincidence_proportions_by_delay[delay][electrode].append(proportion)
                else:
                    coincidence_proportions_by_delay[delay][electrode] = [proportion]
                    electrode_distances[electrode] = layout_df.loc[layout_df['electrode'] == electrode, 'distance'].values[0]

    # Prepare data for heatmap
    all_distances = np.array(list(electrode_distances.values()))
    distance_bins = np.linspace(min(all_distances), max(all_distances), num=31)
    distance_bin_centers = 0.5 * (distance_bins[:-1] + distance_bins[1:])
    delay_bin_centers = np.array(delays)
    heatmap_data = np.zeros((len(delays), len(distance_bin_centers)))

    for i, delay in enumerate(delays):
        all_coincidence_proportions = np.array([np.mean(proportions) for proportions in coincidence_proportions_by_delay[delay].values() if proportions])
        
        # Compute means for each distance bin
        bin_means, _, _ = binned_statistic(all_distances, all_coincidence_proportions, statistic='mean', bins=distance_bins)
        heatmap_data[i, :] = bin_means

    # Log-transform the proportions
    heatmap_data = np.log10(heatmap_data)
    heatmap_data[np.isnan(heatmap_data)] = -6

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = get_cmap('magma')
    cax = ax.imshow(heatmap_data, aspect='auto', cmap=cmap, extent=[min(distance_bin_centers), max(distance_bin_centers), min(delay_bin_centers), max(delay_bin_centers)], origin='lower')

    ax.set_xlabel('Distance from Stimulated Electrode ($\mu m$)')
    ax.set_ylabel('Delay (ms)')
    ax.set_title('Log Proportion of Coincidence by Distance and Delay')

    # Add colorbar
    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label('Log Proportion')

    plt.show()

def plot_spike_distance_coincidence_heatmap_gif(spikes_data_list, event_list, layout_list, stimulated_electrode, start_time, end_time, window_size=1, delays=np.linspace(-1, 3, 41)):
    coincidence_proportions_by_delay = {delay: {} for delay in delays}
    electrode_coords = {}

    for spikes_data, event_data, layout in zip(spikes_data_list, event_list, layout_list):
        spikes_df = pd.DataFrame(spikes_data)
        layout_df = pd.DataFrame(layout)

        # Ensure the stimulated electrode is in the layout
        if stimulated_electrode not in layout_df['electrode'].values:
            raise ValueError(f"No data found for stimulated electrode: {stimulated_electrode}")
        stimulated_coords = layout_df.loc[layout_df['electrode'] == stimulated_electrode, ['x', 'y']].iloc[0]

        # Map channels to electrodes and coordinates
        channel_to_electrode = layout_df.set_index('channel')['electrode'].to_dict()
        channel_to_coords = layout_df.set_index('channel')[['x', 'y']].to_dict(orient='index')
        
        spikes_df['electrode'] = spikes_df['channel'].map(channel_to_electrode)
        spikes_df['x'] = spikes_df['channel'].map(lambda x: channel_to_coords[x]['x'] if x in channel_to_coords else np.nan)
        spikes_df['y'] = spikes_df['channel'].map(lambda x: channel_to_coords[x]['y'] if x in channel_to_coords else np.nan)

        # Handling NaNs by dropping them
        spikes_df = spikes_df.dropna(subset=['electrode', 'x', 'y'])

        # Filter data within the specified time range
        spikes_df_during = spikes_df[(spikes_df['time'] >= start_time) & (spikes_df['time'] <= end_time)]
        
        # Extract the stimulation times from event_data and select every second entry starting from index 1
        stim_times = event_data['time'][1:-1:2]
        
        # Calculate coincidence proportions for each electrode and each delay
        for delay in delays:
            delay_sec = delay / 1000  # Convert ms to seconds
            proportions = calculate_coincidence_proportions(spikes_df_during, stim_times, window_size=window_size / 10000, delay=delay_sec, stimulated_electrode=stimulated_electrode)
            
            for electrode, proportion in proportions.items():
                if electrode == stimulated_electrode:
                    continue

                if electrode in coincidence_proportions_by_delay[delay]:
                    coincidence_proportions_by_delay[delay][electrode].append(proportion)
                else:
                    coincidence_proportions_by_delay[delay][electrode] = [proportion]
                    electrode_coords[electrode] = layout_df.loc[layout_df['electrode'] == electrode, ['x', 'y']].values[0]

    # Define the spatial grid
    x_min, x_max = min(coord[0] for coord in electrode_coords.values()), max(coord[0] for coord in electrode_coords.values())
    y_min, y_max = min(coord[1] for coord in electrode_coords.values()), max(coord[1] for coord in electrode_coords.values())
    x_bins = np.arange(x_min, x_max + 100, 100)
    y_bins = np.arange(y_min, y_max + 100, 100)
    
    filenames = []

    for i, delay in enumerate(delays):
        heatmap_data = np.zeros((len(y_bins) - 1, len(x_bins) - 1))

        all_coincidence_proportions = np.array([np.mean(proportions) for proportions in coincidence_proportions_by_delay[delay].values() if proportions])
        
        # Place data into the heatmap
        for j, electrode in enumerate(electrode_coords.keys()):
            if electrode in coincidence_proportions_by_delay[delay]:
                x, y = electrode_coords[electrode]
                x_idx = np.digitize(x, x_bins) - 1
                y_idx = np.digitize(y, y_bins) - 1
                heatmap_data[y_idx, x_idx] = np.mean(coincidence_proportions_by_delay[delay][electrode])

        # Log-transform the proportions
        heatmap_data = np.log10(heatmap_data)
        heatmap_data[np.isnan(heatmap_data)] = -6

        # Plot heatmap for current delay
        fig, ax = plt.subplots(figsize=(10, 6))
        cmap = get_cmap('magma')
        cax = ax.imshow(heatmap_data, aspect='auto', cmap=cmap, extent=[x_min, x_max, y_min, y_max], origin='lower')

        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title(f'Log Proportion of Coincidence by X and Y Coordinates\nTime Delay: {delay:.2f} ms')
        ax.set_facecolor("black")

        # Add colorbar
        cbar = fig.colorbar(cax, ax=ax)
        cbar.set_label('Log Proportion')

        # Save frame
        filename = f'frame_{i:03d}.png'
        filenames.append(filename)
        plt.savefig(filename)
        plt.close()

    # Create GIF
    with imageio.get_writer('spike_coincidence_heatmap_4.gif', mode='I', duration=0.1) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    # Remove files
    for filename in filenames:
        os.remove(filename)

def calculate_average_amplitudes(spikes_df, stim_times, window_size=0.001, delay=0.0):
    amplitudes = {electrode: [] for electrode in spikes_df['electrode'].unique() if electrode != stimulated_electrode}
    
    for stim_time in stim_times:
        window_start = stim_time + delay
        window_end = stim_time + window_size + delay

        coinciding_spikes = spikes_df[(spikes_df['time'] >= window_start) & (spikes_df['time'] <= window_end)]
        
        for electrode, group in coinciding_spikes.groupby('electrode'):
            if electrode != stimulated_electrode:
                amplitudes[electrode].extend(group['amplitude'].values)

    avg_amplitudes = {electrode: np.mean(amplitudes[electrode]) for electrode in amplitudes if len(amplitudes[electrode]) > 0}
    
    return avg_amplitudes

def plot_spike_amplitude_heatmap_gif(spikes_data_list, layout_list, stimulated_electrode, start_time, end_time, window_size=10,
                                     delays=np.linspace(-1, 3, 41)):
    amplitudes_by_delay = {delay: {} for delay in delays}
    electrode_coords = {}

    for spikes_data, layout in zip(spikes_data_list, layout_list):
        spikes_df = pd.DataFrame(spikes_data)
        layout_df = pd.DataFrame(layout)

        # Ensure the stimulated electrode is in the layout
        if stimulated_electrode not in layout_df['electrode'].values:
            raise ValueError(f"No data found for stimulated electrode: {stimulated_electrode}")
        stimulated_coords = layout_df.loc[layout_df['electrode'] == stimulated_electrode, ['x', 'y']].iloc[0]

        # Map channels to electrodes and coordinates
        channel_to_electrode = layout_df.set_index('channel')['electrode'].to_dict()
        channel_to_coords = layout_df.set_index('channel')[['x', 'y']].to_dict(orient='index')
        
        spikes_df['electrode'] = spikes_df['channel'].map(channel_to_electrode)
        spikes_df['x'] = spikes_df['channel'].map(lambda x: channel_to_coords[x]['x'] if x in channel_to_coords else np.nan)
        spikes_df['y'] = spikes_df['channel'].map(lambda x: channel_to_coords[x]['y'] if x in channel_to_coords else np.nan)

        # Handling NaNs by dropping them
        spikes_df = spikes_df.dropna(subset=['electrode', 'x', 'y'])

        # Filter data within the specified time range
        spikes_df_during = spikes_df[(spikes_df['time'] >= start_time) & (spikes_df['time'] <= end_time)]
        
        # Identify stimulation times and apply a 10 ms refractory period
        stim_times = spikes_df_during[spikes_df_during['electrode'] == stimulated_electrode]['time'].values
        filtered_stim_times = []
        last_time = -np.inf

        for time in stim_times:
            if time >= last_time + 0.01:  # 10 ms refractory period
                filtered_stim_times.append(time)
                last_time = time

        if len(filtered_stim_times) == 0:
            raise ValueError(f"No stimulation times found for electrode: {stimulated_electrode}")

        # Calculate average amplitudes for each electrode and each delay
        for delay in delays:
            delay_sec = delay / 1000  # Convert ms to seconds
            avg_amplitudes = calculate_average_amplitudes(spikes_df_during, filtered_stim_times, window_size=window_size / 10000, delay=delay_sec)
            
            for electrode, avg_amplitude in avg_amplitudes.items():
                if electrode == stimulated_electrode:
                    continue

                if electrode in amplitudes_by_delay[delay]:
                    amplitudes_by_delay[delay][electrode].append(avg_amplitude)
                else:
                    amplitudes_by_delay[delay][electrode] = [avg_amplitude]
                    electrode_coords[electrode] = layout_df.loc[layout_df['electrode'] == electrode, ['x', 'y']].values[0]

    # Define the spatial grid
    x_min, x_max = min(coord[0] for coord in electrode_coords.values()), max(coord[0] for coord in electrode_coords.values())
    y_min, y_max = min(coord[1] for coord in electrode_coords.values()), max(coord[1] for coord in electrode_coords.values())
    x_bins = np.arange(x_min, x_max + 100, 100)
    y_bins = np.arange(y_min, y_max + 100, 100)
    
    # Determine global min and max values for the color scale
    global_min = float('inf')
    global_max = float('-inf')

    for delay in delays:
        for amplitudes in amplitudes_by_delay[delay].values():
            if amplitudes:
                min_val = np.min(amplitudes)
                max_val = np.max(amplitudes)
                if min_val < global_min:
                    global_min = min_val
                if max_val > global_max:
                    global_max = max_val

    filenames = []

    for i, delay in enumerate(delays):
        heatmap_data = np.zeros((len(y_bins) - 1, len(x_bins) - 1))

        all_avg_amplitudes = np.array([np.mean(amplitudes) for amplitudes in amplitudes_by_delay[delay].values() if amplitudes])
        
        # Place data into the heatmap
        for j, electrode in enumerate(electrode_coords.keys()):
            if electrode in amplitudes_by_delay[delay]:
                x, y = electrode_coords[electrode]
                x_idx = np.digitize(x, x_bins) - 1
                y_idx = np.digitize(y, y_bins) - 1
                heatmap_data[y_idx, x_idx] = np.mean(amplitudes_by_delay[delay][electrode])

        # Plot heatmap for current delay
        fig, ax = plt.subplots(figsize=(10, 6))
        cmap = get_cmap('magma')
        cax = ax.imshow(heatmap_data, aspect='auto', cmap=cmap, extent=[x_min, x_max, y_min, y_max], origin='lower', vmin=global_min, vmax=global_max)

        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title(f'Average Amplitude of Spikes by X and Y Coordinates\nTime Delay: {delay:.2f} ms')

        # Add colorbar
        cbar = fig.colorbar(cax, ax=ax)
        cbar.set_label('Average Amplitude')

        # Save frame
        filename = f'frame_{i:03d}.png'
        filenames.append(filename)
        plt.savefig(filename)
        plt.close()

    # Create GIF
    with imageio.get_writer('spike_amplitude_heatmap.gif', mode='I', duration=0.1) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    # Remove files
    for filename in filenames:
        os.remove(filename)

############## Angular Anisotropy

def calculate_transmission_speeds(spikes_df, stim_times, stimulated_coords):
    speeds = {electrode: [] for electrode in spikes_df['electrode'].unique() if electrode != stimulated_electrode}
    
    for stim_time in stim_times:
        window_start = stim_time
        window_end = stim_time + 0.010  # 10 ms window

        coinciding_spikes = spikes_df[(spikes_df['time'] > window_start) & (spikes_df['time'] <= window_end)]
        
        for electrode, group in coinciding_spikes.groupby('electrode'):
            if electrode != stimulated_electrode:
                distance = np.linalg.norm(np.array([group['x'].values[0], group['y'].values[0]]) - stimulated_coords)  # Convert distance to m
                time_elapsed = (group['time'].values[0] - stim_time)/1000  # time_elapsed is in ms
                speed = distance / time_elapsed  # Speed in mm/ms
                speeds[electrode].append(speed)
    
    avg_speeds = {electrode: np.mean(speeds[electrode]) for electrode in speeds if len(speeds[electrode]) > 0}
    return avg_speeds

def calculate_angles(layout_df, stimulated_coords):
    layout_df['angle'] = np.arctan2(layout_df['y'] - stimulated_coords[1], layout_df['x'] - stimulated_coords[0])
    layout_df['angle'] = np.degrees(layout_df['angle'])
    layout_df['angle'] = (layout_df['angle'] + 360) % 360  # Normalize to [0, 360) degrees
    return layout_df

def process_spike_data_angular(spikes_data_list, layout_list, stimulated_electrode, start_time, end_time): # needs rewriting as to take event data
    speeds_by_angle = {i: [] for i in range(32)}
    electrode_distances = {}
    all_angles = []

    for spikes_data, layout in zip(spikes_data_list, layout_list):
        spikes_df = pd.DataFrame(spikes_data)
        layout_df = pd.DataFrame(layout)

        # Ensure the stimulated electrode is in the layout
        if stimulated_electrode not in layout_df['electrode'].values:
            raise ValueError(f"No data found for stimulated electrode: {stimulated_electrode}")
        stimulated_coords = layout_df.loc[layout_df['electrode'] == stimulated_electrode, ['x', 'y']].iloc[0].values

        # Calculate angles and distances from each electrode to the stimulated electrode
        layout_df = calculate_angles(layout_df, stimulated_coords)
        layout_df['distance'] = np.sqrt((layout_df['x'] - stimulated_coords[0])**2 + 
                                        (layout_df['y'] - stimulated_coords[1])**2)

        # Map electrodes to distances, and angles
        electrode_to_distance = layout_df.set_index('electrode')['distance'].to_dict()
        electrode_to_angle = layout_df.set_index('electrode')['angle'].to_dict()
        electrode_to_coords = layout_df.set_index('electrode')[['x', 'y']].to_dict(orient='index')
        
        spikes_df['electrode'] = spikes_df['channel'].map(dict(zip(layout_df['channel'], layout_df['electrode'])))
        spikes_df['distance'] = spikes_df['electrode'].map(electrode_to_distance)
        spikes_df['angle'] = spikes_df['electrode'].map(electrode_to_angle)
        spikes_df['x'] = spikes_df['electrode'].map(lambda x: electrode_to_coords[x]['x'] if x in electrode_to_coords else np.nan)
        spikes_df['y'] = spikes_df['electrode'].map(lambda x: electrode_to_coords[x]['y'] if x in electrode_to_coords else np.nan)

        # Handling NaNs by dropping them
        spikes_df = spikes_df.dropna(subset=['distance', 'electrode', 'angle', 'x', 'y'])

        # Filter data within the specified time range
        spikes_df_during = spikes_df[(spikes_df['time'] >= start_time) & (spikes_df['time'] <= end_time)]
        
        # Identify stimulation times and apply a 10 ms refractory period
        stim_times = spikes_df_during[spikes_df_during['electrode'] == stimulated_electrode]['time'].values
        filtered_stim_times = []
        last_time = -np.inf

        for time in stim_times:
            if time >= last_time + 0.01:  # 10 ms refractory period
                filtered_stim_times.append(time)
                last_time = time

        if len(filtered_stim_times) == 0:
            raise ValueError(f"No stimulation times found for electrode: {stimulated_electrode}")

        # Calculate transmission speeds for each electrode
        avg_speeds = calculate_transmission_speeds(spikes_df_during, filtered_stim_times, stimulated_coords)
        
        for electrode, speed in avg_speeds.items():
            angle = layout_df.loc[layout_df['electrode'] == electrode, 'angle'].values[0]
            bin_index = int(angle // 11.25)  # Each bin covers 11.25 degrees (32 bins)
            speeds_by_angle[bin_index].append(speed)
            electrode_distances[electrode] = layout_df.loc[layout_df['electrode'] == electrode, 'distance'].values[0]
            all_angles.append(angle)

    return speeds_by_angle, all_angles

def plot_angular_anisotropy(spikes_data_list, layout_list, stimulated_electrode, start_time, end_time): 
    speeds_by_angle, all_angles = process_spike_data_angular(spikes_data_list, layout_list, stimulated_electrode, start_time, end_time)

    # Calculate average speeds and error bars for each angular bin
    avg_speeds = []
    speed_errors = []
    angle_bins = np.arange(0, 360, 11.25)

    for i in range(32):
        if speeds_by_angle[i]:
            avg_speeds.append(np.mean(speeds_by_angle[i]))
            speed_errors.append(np.std(speeds_by_angle[i]) / np.sqrt(len(speeds_by_angle[i])))
        else:
            avg_speeds.append(0)
            speed_errors.append(0)

    # Convert angle_bins to radians for the polar plot
    angle_bins_rad = np.deg2rad(angle_bins)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})
    ax.errorbar(angle_bins_rad, avg_speeds, yerr=speed_errors, fmt='o', capsize=5, linestyle='-', color='b', label='Average Transmission Speed')

    # Set the labels for angular space
    ax.set_xticks(np.pi/180.0 * np.linspace(0, 360, 8, endpoint=False))
    ax.set_xticklabels(['0', '$\pi/4$', '$\pi/2$', '3$\pi/4$', '$\pi$', '5$\pi/4$', '3$\pi/2$', '7$\pi/4$'])
    
    ax.set_ylabel('Transmission Speed (mm/ms)')
    ax.set_title('Angular Anisotropy of Signal Propagation')
    ax.legend()

    plt.show()



if __name__ == "__main__":
    filename = '/Users/danielrebbin/Documents/Academia/UvA/Internship/Wes_Files/Data/Stimulation/2407/ephaptic_stimulation_caspfiguration_100hz_1.raw.h5'

    well = 5

    spikes_data_0, event_data_0, layout_0, sf, stimulated_electrode = load_spikes(filename, well, 0)

    stimulated_electrode = 13360
    start_time = 0
    stop_time = 1800


    spikes_data_list = [spikes_data_0]
    event_list = [event_data_0]
    layout_list = [layout_0]
    start_times = [start_time]
    stop_times = [stop_time]

    #print(event_data_0)
    #spikes_data_list = [spikes_data_6, spikes_data_7, spikes_data_8]
    #layout_list = [layout_6, layout_7, layout_8]

    plot_spike_distance_scatter_discounted(spikes_data_list, layout_list, stimulated_electrode, start_time, stop_time)
    plot_layout_grid_with_fr_pooled(spikes_data_list, layout_list, start_times, stop_times, stimulated_electrode)
    #plot_inter_spike_interval_scatter(spikes_data_list, layout_list, stimulated_electrode, start_time, stop_time)
    #plot_spike_distance_coincidence_heatmap(spikes_data_list, event_list, layout_list, stimulated_electrode, start_time, stop_time, window_size=1)
    #plot_spike_distance_coincidence_heatmap_gif(spikes_data_list, event_list, layout_list, stimulated_electrode, start_time, stop_time)
    #plot_angular_anisotropy(spikes_data_list, layout_list, stimulated_electrode, start_time, stop_time)
    #plot_spike_amplitude_heatmap_gif(spikes_data_list, layout_list, stimulated_electrode, start_time, stop_time)