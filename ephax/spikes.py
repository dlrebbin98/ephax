import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import numpy as np
import pandas as pd
import copy
import tqdm
import re
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize, LogNorm
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from ipywidgets import interact, widgets
from scipy.stats import powerlaw
from scipy.signal import hilbert
from helper_functions import load_spikes, get_activity_sorted_electrodes
from select_electrodes import assign_n_proximate_electrodes

"""
This script shows how to open a raw data file and how to read and interpret the data. 
ATTENTION: The data file format is not considered stable and may change in the future.
"""


def plot_spikes(spikes_data):
    plt.figure(figsize=(10, 6))
    plt.scatter(spikes_data['time'], spikes_data['channel'], s=5, c='b', marker='o', alpha=0.5)
    plt.xlabel('Time (in seconds)')
    plt.ylabel('Channel')
    plt.title('Spike Events Recorded by Channels')
    plt.grid(True)
    plt.show()

def plot_spikes_events(spikes_data, event_data):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(spikes_data['time'], spikes_data['channel'], s=5, c='b', marker='o', alpha=0.5)
    ax.scatter(event_data['time'], [0] * len(event_data['time']), s=5, c='r', marker='x', alpha=0.5)
    ax.set_xlabel('Time (in seconds)')
    ax.set_ylabel('Channel')
    ax.set_title('Spike Events Recorded by Channels')
    ax.grid(True)
    plt.show()

def plot_spikes_sorted(spikes_data, event_data, layout, well, stimulated_electrode):
    
    # Convert spikes_data dictionary to a DataFrame
    spikes_df = pd.DataFrame(spikes_data)
    # Convert spikes_data dictionary to a DataFrame
    events_df = pd.DataFrame(event_data)

    #  Filter spikes data between time points 600 and 700
    #spikes_df = spikes_df[(spikes_df['time'] >= 600) & (spikes_df['time'] <= 700)]
    
    # Filter event data between time points 600 and 700
    #events_df = events_df[(events_df['time'] >= 600) & (events_df['time'] <= 700)]
    
    # Create a DataFrame from the layout for easy manipulation
    layout_df = pd.DataFrame(layout)
    
    # Get coordinates for the stimulated electrode
    if stimulated_electrode not in layout_df['electrode'].values:
        raise ValueError(f"No data found for stimulated electrode: {stimulated_electrode}")
    stimulated_coords = layout_df.loc[layout_df['electrode'] == stimulated_electrode, ['x', 'y']].iloc[0]

    # Calculate distances from each electrode to the stimulated electrode
    layout_df['distance'] = np.sqrt((layout_df['x'] - stimulated_coords['x'])**2 + 
                                    (layout_df['y'] - stimulated_coords['y'])**2)
    
    # Create a dictionary to map channels to their distances
    channel_to_distance = layout_df.set_index('channel')['distance'].to_dict()
    
    # Map the channels in spikes_df to their distances
    spikes_df['y_distance'] = spikes_df['channel'].map(channel_to_distance)
    
    # Create a figure with two subplots: one for the scatter plot, one for the density plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax_density = ax.inset_axes([1., 0, 0.1, 1.], sharey=ax)  # Inset axes for density plot
    
    # Scatter plot
    scatter = ax.scatter(spikes_df['time'], spikes_df['y_distance'], s=5, marker='o', alpha=0.5)
    ax.scatter(events_df['time'], [0] * len(events_df['time']), s=7, c='r', marker='x', alpha=1)
    ax.set_xlabel('Time (in seconds)')
    ax.set_ylabel('Distance from Stimulated Electrode (micrometers)')
    ax.set_title(f'Spike Events Recorded by Distance from Stimulated Electrode in Well {well}')
    ax.grid(True)
    
    # Density plot
    """ sns.kdeplot(y=spikes_df['y_distance'], ax=ax_density, color='r', linewidth=2, fill=True)
    ax_density.set_ylabel('')
    # ax_density.set_yticks([])  # Hide the density plot y-axis ticks
    ax_density.set_xticks([])  # Hide the density plot x-axis ticks
    ax_density.set_xlabel('Density') """

    plt.tight_layout()
    plt.show()

def calculate_ifr(spikes_data, selected_electrodes, start_time=None, end_time=None):
    # Initialize a dictionary to hold spike times for each selected electrode
    electrode_spikes = {el: [] for el in selected_electrodes}

    # Iterate over the spikes once, collecting times for selected electrodes
    for time, electrode in zip(spikes_data['time'], spikes_data['electrode']):
        if electrode in electrode_spikes:
            electrode_spikes[electrode].append(time)

    # Convert lists to numpy arrays for further processing
    for el in electrode_spikes:
        electrode_spikes[el] = np.array(electrode_spikes[el])
    
    if start_time is None:
        start_time = min(spikes_data['time'])
    if end_time is None:
        end_time = max(spikes_data['time'])
    
    def _calculate_ifr_for_electrode(spike_times):
        if len(spike_times) < 2:
            return [start_time, end_time], [0, 0]
        
        ifr_times = [start_time]
        ifr_values = [0]
        
        first_spike = spike_times[0]
        ifr_times.append(first_spike)
        ifr_values.append(0)
        
        for i in range(len(spike_times) - 1):
            current_spike = spike_times[i]
            next_spike = spike_times[i + 1]
            interval = next_spike - current_spike
            
            ifr_times.extend([current_spike, next_spike])
            ifr_values.extend([1/interval, 1/interval])
        
        last_spike = spike_times[-1]
        ifr_times.append(last_spike)
        ifr_values.append(0)
        
        ifr_times.append(end_time)
        ifr_values.append(0)
        
        return np.array(ifr_times), np.array(ifr_values)

    ifr_data = {}
    total_firing = {}
    all_ifr_values = []

    for electrode, spike_times in electrode_spikes.items():
        spike_times = spike_times[(spike_times >= start_time) & (spike_times <= end_time)]
        if len(spike_times) > 0:
            ifr_times, ifr_values = _calculate_ifr_for_electrode(spike_times)
            ifr_data[electrode] = (ifr_times, ifr_values)
            total_firing[electrode] = len(spike_times) / (end_time - start_time)
            all_ifr_values.extend(ifr_values)
    
    return ifr_data, total_firing, np.array(all_ifr_values)

def plot_ifr(spikes_data, sf, start_time, end_time, selected_electrodes, log_scale=False):
    ifr_data, total_firing, all_ifr_values = calculate_ifr(spikes_data, selected_electrodes, start_time, end_time)
    time_points = np.linspace(start_time, end_time, int((end_time - start_time) * sf))
    heatmap_data = []
    valid_electrodes = []

    for i, electrode in enumerate(selected_electrodes):
        if electrode in ifr_data:
            times, ifr = ifr_data[electrode]
            if log_scale:
                ifr = np.where(ifr == 0, 1e-3, ifr)  # Replace 0s with epsilon before log
                ifr = np.log10(ifr)
                ifr = np.where(np.isinf(ifr), -3, ifr)  # Replace -inf with a minimum log value
            heatmap_data.append(np.interp(time_points, times, ifr))
            valid_electrodes.append(electrode)
    
    heatmap_data = np.array(heatmap_data)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plotting the IFR heatmap
    im = ax1.imshow(heatmap_data, aspect='auto', origin='lower', 
                   extent=[start_time, end_time, -0.5, len(valid_electrodes)-0.5],
                   cmap='viridis', interpolation='nearest')
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Channel by Firing Frequency Rank')
    
    if log_scale:
        ax1.set_title(f'Log Instantaneous Firing Rate Across Top {len(valid_electrodes)} electrodes')
        cbar_label = 'Log Instantaneous Firing Rate (Hz)'
    else:
        ax1.set_title(f'Instantaneous Firing Rate Across Top {len(valid_electrodes)} electrodes')
        cbar_label = 'Instantaneous Firing Rate (Hz)'
    
    ax1.set_yticks([0, len(valid_electrodes)-1])
    ax1.set_yticklabels([1, len(valid_electrodes)])
    
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label(cbar_label)

    # Prepare IFR values for histogram
    if log_scale:
        all_ifr_values = all_ifr_values[all_ifr_values > 1e-3]  # Remove zeros replaced by epsilon
        all_ifr_values = np.log10(all_ifr_values)
    
    # Plotting the histogram of IFR values
    ax2.hist(all_ifr_values, bins=50, color='blue', edgecolor='black')
    ax2.set_xlabel('Instantaneous Firing Rate (Hz)' if not log_scale else 'Log Instantaneous Firing Rate (Hz)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Histogram of Instantaneous Firing Rates')

    plt.tight_layout()
    plt.show()


def plot_spike_distance_histogram_discounted(spikes_data, layout, stimulated_electrode, start_time, end_time):
    # Convert spikes_data dictionary to a DataFrame
    spikes_df_pre = pd.DataFrame(spikes_data)
    
    # Filter data within the specified time range
    spikes_df = spikes_df_pre[(spikes_df_pre['time'] >= start_time) & (spikes_df_pre['time'] <= end_time)]

    # Filter for the time range previous to stimulation
    spikes_df_pre = spikes_df_pre[(spikes_df_pre['time'] <= start_time)]
    
    # Create a DataFrame from the layout for easy manipulation
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

    # Map channels to distances
    channel_to_distance = layout_df.set_index('channel')['distance'].to_dict()
    spikes_df['y_distance'] = spikes_df['channel'].map(channel_to_distance)
    channels_missing_coordinates = np.unique(spikes_df['channel'][spikes_df['y_distance'].isna().values])
    spikes_df_pre['y_distance'] = spikes_df_pre['channel'].map(channel_to_distance)

    # Handling NaNs by replacing them with the mean distance
    if spikes_df['y_distance'].isna().any():
        spikes_df = spikes_df[~spikes_df['channel'].isin(channels_missing_coordinates)]

     # Filter data within the specified time range
    spikes_df_during = spikes_df[(spikes_df['time'] >= start_time) & (spikes_df['time'] <= end_time)]
    
    # Filter for the time range previous to stimulation
    spikes_df_pre = spikes_df[spikes_df['time'] < start_time]

    # Calculate firing rates (Hz) for both periods
    duration_pre = start_time
    duration_during = end_time - start_time
    
    firing_rates_pre = spikes_df_pre.groupby('channel').size() / duration_pre
    print(firing_rates_pre)
    firing_rates_during = spikes_df_during.groupby('channel').size() / duration_during
    
    # Adjust firing rates by discounting pre-stimulation rates
    firing_rates_discounted = firing_rates_during - firing_rates_pre.reindex(firing_rates_during.index, fill_value=0)
    spikes_df_during['firing_rate_discounted'] = spikes_df_during['channel'].map(firing_rates_discounted)
    
    # Define bins logarithmically
    min_dist = min(spikes_df_during['y_distance'])
    max_dist = max(spikes_df_during['y_distance'])
    bins = np.logspace(np.log10(min_dist), np.log10(max_dist), num=50)
    
    # Create histogram bins and count spikes in each bin
    counts, bin_edges = np.histogram(spikes_df_during['y_distance'], bins=bins, weights=spikes_df_during['firing_rate_discounted'])
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Count electrodes in each bin
    electrode_counts, _ = np.histogram(layout_df['distance'], bins=bins)
    
    # Normalize spike counts by the number of electrodes in each bin
    normalized_counts = np.divide(counts, electrode_counts, where=electrode_counts != 0)
    normalized_counts = np.nan_to_num(normalized_counts)  # Replace NaNs with zeros

    # Plot the normalized histogram on a logarithmic y-axis
    plt.figure(figsize=(10, 6))
    plt.bar(bin_centers, normalized_counts, width=np.diff(bin_edges), edgecolor='black', alpha=0.7, log=True)
    plt.xscale('log')
    plt.xlabel('Distance from Stimulated Electrode (micrometers)')
    plt.ylabel('Normalized Firing Rate Difference (Hz)')
    plt.title('Normalized Histogram of Firing Rate Differences by Distance')
    plt.grid(True)
    plt.show()

def plot_firing_probability(spikes_data, start_time, end_time, num_bins=10):
    """
    Plots the firing frequency against its probability on a log-log scale, with logarithmic binning.

    :param spikes_data: dict, contains spiking data loaded from the file.
    :param start_time: float, the start time of the interval to analyze.
    :param end_time: float, the end time of the interval to analyze.
    :param num_bins: int, number of logarithmic bins for the histogram of firing rates.
    :return: None
    """
    # Filter spikes to the desired time interval
    mask = (spikes_data['time'] >= start_time) & (spikes_data['time'] <= end_time)
    filtered_times = spikes_data['time'][mask]
    filtered_channels = spikes_data['channel'][mask]
    unique_channels = np.unique(filtered_channels)

    # Calculate the total duration of the interval
    interval_duration = end_time - start_time

    # Calculate average firing rate per neuron across the entire interval
    avg_firing_rates = []
    for channel in unique_channels:
        count_spikes = np.sum(filtered_channels == channel)
        avg_firing_rate = count_spikes / interval_duration
        avg_firing_rates.append(avg_firing_rate)

    avg_firing_rates = np.array(avg_firing_rates)

    # Logarithmic bins for plotting the average firing rates
    if avg_firing_rates.min() > 0:
        log_bins = np.logspace(np.log10(avg_firing_rates.min()), np.log10(avg_firing_rates.max()), num_bins)
    else:
        log_bins = np.logspace(-1, np.log10(avg_firing_rates.max()), num_bins)

    # Calculate the probability distribution using logarithmic bins
    hist, bin_edges = np.histogram(avg_firing_rates, bins=log_bins, density=True)
    probabilities = hist / np.sum(hist)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Plot using logarithmic scales on both axes
    plt.figure(figsize=(8, 6))
    plt.loglog(bin_centers, probabilities, marker='o', linestyle='-', color='b')
    plt.xlabel('Average Firing Frequency (Hz)')
    plt.ylabel('Probability')
    plt.title('Log-Log Plot of Average Firing Frequency vs. Probability (Logarithmic Bins)')
    plt.grid(True, which="both", ls="--")
    plt.show()

    # Fit to a power law model and print the exponent
    fit = powerlaw.fit(avg_firing_rates)
    print("Fitted power-law exponent:", fit[0])

def plot_layout(layout, well):
    plt.figure(figsize=(10, 6))
    plt.scatter(layout['x'], layout['y'], s=5, c='b', marker='o', alpha=0.5)
    """ for i in range(len(layout['electrode'])):
        plt.text(layout['x'][i], layout['y'][i], str(layout['electrode'][i]),
                 fontsize=8, ha='center', va='bottom') """
    print(len(layout['electrode']))
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.title(f'Spatial Layout of Well {well}')
    plt.grid(True)
    plt.show()

def plot_layout_with_fr(spikes_data, layout, start_time, end_time, stimulated_electrode):

    valid_indices = (spikes_data['time'] >= start_time) & (spikes_data['time'] <= end_time)
    filtered_times = spikes_data['time'][valid_indices]
    filtered_channels = spikes_data['channel'][valid_indices]

    map = copy.copy(layout)

    # Calculate average firing rate for each channel
    unique_channels, channel_counts = np.unique(filtered_channels, return_counts=True)
    time_duration = end_time - start_time
    firing_rates = dict(zip(unique_channels, channel_counts / time_duration))

    # Remove channels from firing_rates that are not in layout
    firing_rates = {ch: firing_rates[ch] for ch in firing_rates if ch in map['channel']}

    # Convert the result back to a list if needed
    missing_elements_list = list(set(map['channel']) - set(unique_channels))

    # Find indices of the values in the 'channel' list
    indices_to_delete = [i for i, val in enumerate(map['channel']) if val in missing_elements_list]

    # Delete corresponding values in other lists
    for key in map.keys():
        map[key] = [val for i, val in enumerate(map[key]) if i not in indices_to_delete]
    
    # Prepare color values for each channel based on its firing rate
    color_values = list(firing_rates.values())

    # Plot layout with firing rate as color gradient
    fig, ax = plt.subplots(figsize=(10, 6))
    norm = LogNorm(vmin=min(color_values), vmax=max(color_values))
    cmap = plt.get_cmap('viridis')
    scalar_map = ScalarMappable(norm=norm, cmap=cmap)
    ax.scatter(map['x'], map['y'], s=200, c=color_values, cmap=cmap, norm=norm, alpha=0.7)
    ax.set_xlabel('X-coordinate ($\mu m$)')
    ax.set_ylabel('Y-coordinate ($\mu m$)')
    ax.set_title(f'Spatial Layout of Well {well} with Firing Rates')

    # Highlight the electrode with a red circle if it exists in the map
    target_electrode = stimulated_electrode
    if target_electrode in map['electrode']:
        index = map['electrode'].index(target_electrode)
        ax.scatter(map['x'][index], map['y'][index], s=200, c='none', edgecolors='red', label='Stimulated Electrode')

    # Create a subplot for the colorbar
    cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(scalar_map, cax=cax)
    cbar.set_label('Average Firing Rate (Hz)')

    ax.legend(loc='upper right')
    plt.grid(True)
    plt.show()

def plot_layout_with_fr_3d(spikes_data, layout, start_time, end_time, stimulated_electrode):
    valid_indices = (spikes_data['time'] >= start_time) & (spikes_data['time'] <= end_time)
    filtered_times = spikes_data['time'][valid_indices]
    filtered_channels = spikes_data['channel'][valid_indices]

    map = copy.copy(layout)

    # Calculate average firing rate for each channel
    unique_channels, channel_counts = np.unique(filtered_channels, return_counts=True)
    time_duration = end_time - start_time
    firing_rates = dict(zip(unique_channels, channel_counts / time_duration))

    # Filter out unused channels
    firing_rates = {ch: firing_rates[ch] for ch in firing_rates if ch in map['channel']}
    firing_rates_list = [firing_rates[ch] if ch in firing_rates else 0 for ch in map['channel']]

    # 3D plot
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Normalize color map based on firing rates
    norm = LogNorm(vmin=min(firing_rates.values()), vmax=max(firing_rates.values()))
    cmap = plt.get_cmap('viridis')
    
    # Create 3D bars
    for (x, y, z) in zip(map['x'], map['y'], firing_rates_list):
        ax.bar3d(x, y, 0, 20, 20, z, color=cmap(norm(z)), shade=True)

    ax.set_xlabel('X-coordinate')
    ax.set_ylabel('Y-coordinate')
    ax.set_zlabel('Firing Rate')

    # Highlight the stimulated electrode
    if stimulated_electrode in map['electrode']:
        index = np.where(map['electrode'] == stimulated_electrode)[0][0]
        ax.bar3d(map['x'][index], map['y'][index], 0, 20, 20, firing_rates_list[index], color='red', label='Stimulated Electrode')

    # Color bar to show firing rates
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Average Firing Rate')

    ax.legend()
    #ax.set_aspect('equal')
    plt.show()

def filter_spikes_data(spikes_data, start_time, end_time):
    time_filter = (spikes_data['time'] >= start_time) & (spikes_data['time'] <= end_time)
    filtered_spikes_data = {
        'time': spikes_data['time'][time_filter],
        'channel': spikes_data['channel'][time_filter],
        'amplitude': spikes_data['amplitude'][time_filter]
    }
    return filtered_spikes_data

def compute_phase_synchrony(spikes_data, layout, sf, stimulus_electrode):
    # Convert spike times to phase using a sparse approach
    def spikes_to_phase(spike_times, fs, duration):
        spike_train = np.zeros(duration)
        spike_train[spike_times] = 1
        
        # Hilbert transform to get the analytic signal
        analytic_signal = hilbert(spike_train)
        instantaneous_phase = np.angle(analytic_signal)
        
        return instantaneous_phase

    # Get spike times for the stimulus electrode
    stimulus_spike_times = spikes_data['time'][spikes_data['channel'] == stimulus_electrode]
    duration = int(max(spikes_data['time']) * sf) + 1  # Duration in samples

    # Compute phase for central (stimulus) electrode
    phase_central = spikes_to_phase((stimulus_spike_times * sf).astype(int), sf, duration)

    plv_values = {}
    for electrode in layout['electrode']:
        if electrode != stimulus_electrode:
            spike_times = spikes_data['time'][spikes_data['channel'] == electrode]
            phase_other = spikes_to_phase((spike_times * sf).astype(int), sf, duration)

            # Compute phase difference
            phase_diff = np.angle(np.exp(1j * (phase_central - phase_other)))

            # Compute PLV
            plv = np.abs(np.mean(np.exp(1j * phase_diff)))
            plv_values[electrode] = plv

    return plv_values

if __name__ == "__main__":
    #filename = '/Users/danielrebbin/Documents/Academia/UvA/Internship/Wes_Files/Data/Stimulation/000059/data.raw.h5'
    #filename = '/Users/danielrebbin/Documents/Academia/UvA/Internship/Wes_Files/Data/Stimulation/240729/ephaptic_stimulation_caspfiguration_100hz.raw.h5'
    #filename = '/Users/danielrebbin/Documents/Academia/UvA/Internship/Wes_Files/Data/Stimulation/Artefact/ephaptic_stimulation_caspfiguration_specific_electrode_50hz_240816.raw.h5'
    #filename = '/Users/danielrebbin/Documents/Academia/UvA/Internship/Wes_Files/Data/Stimulation/2407/ephaptic_stimulation_caspfiguration_control_0.raw.h5'
    #filename = '/Users/danielrebbin/Documents/Academia/UvA/Internship/Wes_Files/Data/Stimulation/2407/ephaptic_stimulation_caspfiguration_100hz_2.raw.h5'
    filename = '/Users/danielrebbin/Documents/Academia/UvA/Internship/Wes_Files/Data/Stimulation/2407/control_0.raw.h5'
    well = 0

    spikes_data, event_data, layout, sf, stimulated_electrode = load_spikes(filename, well)
    stimulated_electrode = 13260
    start_time = 200
    end_time = 230


    #plot_layout(layout, well)
    plot_layout_with_fr(spikes_data, layout, start_time, end_time, stimulated_electrode)
    #plot_layout_with_fr_3d(spikes_data, layout, start_time, end_time, stimulated_electrode)
    #plot_spikes(spikes_data)
    #plot_spikes_events(spikes_data, event_data)
    plot_spikes_sorted(spikes_data, event_data, layout, well, stimulated_electrode)
    #plot_spike_distance_histogram_discounted(spikes_data, layout, stimulated_electrode, start_time, end_time)
    #plot_firing_probability(spikes_data, num_bins=10, start_time=start_time, end_time=end_time)

    selected_electrodes = get_activity_sorted_electrodes([spikes_data], 10, 110)
    #selected_electrodes = assign_n_proximate_electrodes(layout, stimulated_electrode, 0)
    plot_ifr(spikes_data, sf, start_time=start_time, end_time=end_time, selected_electrodes=selected_electrodes, log_scale=True)