import matplotlib.pyplot as plt
import h5py
import numpy as np
from scipy.signal import butter, lfilter
from matplotlib.colors import LogNorm, Normalize
from matplotlib.cm import ScalarMappable, viridis
from ephax.helper_functions import load_raw

"""
This script shows how to open a raw data file and how to read and interpret the data. 
ATTENTION: The data file format is not considered stable and may change in the future.
"""

def filter_data(data, low, high, sf, order=5):
    # Determine Nyquist frequency
    nyq = sf / 2

    # Set bands
    low = low / nyq
    high = high / nyq

    # Calculate coefficients
    b, a = butter(order, [low, high], btype='band')

    # Filter signal
    filtered_data = lfilter(b, a, data)

    return filtered_data

def plot_raw(X, t, event_data, n=50):
    # Find the n channels with the highest maximum voltage values
    max_vals = np.max(X, axis=0)
    top_channels_indices = np.argsort(max_vals)[-n:]

    # Plot these channels
    plt.figure(figsize=(15, 10))
    for idx in top_channels_indices:
        plt.plot(t[:X.shape[0]], X[:, idx], label=f'Channel {idx}')
    # Draw vertical lines for events
    for etime in event_data['time']:
        plt.axvline(x=etime, color='r', linestyle='--', linewidth=0.5)
    plt.ylabel('Volts')
    plt.xlabel('Seconds')
    plt.legend(loc='upper right')
    plt.show()

def calculate_deviation(X, sf, event_data, start_frame, stimulation_electrode_id, frame_indices):

    # Extract stimulation electrode location
    stim_electrode_index = np.where(layout['electrode'] == stimulation_electrode_id)[0][0]
    stim_x = layout['x'][stim_electrode_index]
    stim_y = layout['y'][stim_electrode_index]

    # Calculate geometric distances
    distances = np.sqrt((layout['x'] - stim_x)**2 + (layout['y'] - stim_y)**2)
    print("Distances shape:", distances.shape)
    print("X shape:", X.shape)

    # Ensure distances array matches the number of channels in X
    valid_channel_indices = [np.where(layout['channel'] == ch)[0][0] for ch in np.arange(1024) if ch in layout['channel']]
    distances = distances[valid_channel_indices]

    # Calculate deviations
    num_events = len(event_data['time'])
    print("Number of Events:", num_events)
    deviations = np.zeros((len(frame_indices), X.shape[1]))  # Adjust for the number of frames in frame_indices

    for i in range(0, num_events, 2):  # Every second event
        event_time = event_data['time'][i]
        event_frame = int(event_time * sf - start_frame)
        if event_frame - 100 < 0 or event_frame + max(frame_indices) >= X.shape[0]:
            print(f"Skipping event at time {event_time:.2f}s (frame {event_frame}) due to insufficient data range")
            continue

        for idx, frame in enumerate(frame_indices):
            if event_frame + frame < X.shape[0]:
                event_voltage = X[event_frame + frame, :]
                baseline_voltage = np.mean(X[event_frame - 100:event_frame, :], axis=0)
                abs_deviation = np.abs(event_voltage - baseline_voltage)
                deviations[idx, :] += abs_deviation * 1000

    # Average deviations over all considered events
    deviations /= (num_events // 2)

    return deviations, distances

def plot_heatmap(deviations, distances, frame_indices, sf):
    # Ensure frame_indices are within the bounds of the deviations array
    frame_indices = [idx for idx in frame_indices if idx < deviations.shape[0]]
    
    # Convert frame indices to time in microseconds
    time_indices = [idx / sf * 1e6 for idx in frame_indices]  # Convert to microseconds
    
    # Bin distances and calculate mean deviation per bin
    bin_size = 100  # micrometers
    max_distance = np.max(distances)
    bins = np.arange(0, max_distance + bin_size, bin_size)
    binned_deviations = np.zeros((len(frame_indices), len(bins) - 1))

    for i in range(len(bins) - 1):
        bin_mask = (distances >= bins[i]) & (distances < bins[i + 1])
        if np.any(bin_mask):
            for idx, frame in enumerate(frame_indices):
                if frame < deviations.shape[0]:
                    binned_deviations[idx, i] = np.mean(deviations[frame, bin_mask], axis=0)

    # Plot heatmap with logarithmic color scale
    plt.figure(figsize=(10, 6))
    plt.imshow(
        binned_deviations, 
        aspect='auto', 
        origin='lower', 
        extent=[0, max_distance, min(time_indices), max(time_indices)], 
        norm=LogNorm(),
        cmap='viridis'  # Ensure a colormap is set, just in case
    )
    plt.colorbar(label='Deviation (log scale)')
    plt.xlabel('Distance (micrometers)')
    plt.ylabel('Time (microseconds)')  # Update ylabel to reflect time in microseconds
    plt.title('Heatmap of Deviation Over Time')
    plt.show()

def plot_error_bars(deviations, distances, frame_indices, sf):
    # Remove the stimulation electrode with 0 distance
    non_zero_distances = distances[distances > 0]
    non_zero_deviations = deviations[:, distances > 0]

    # Bin distances in logspace and calculate mean deviation and std deviation per bin
    bins = np.logspace(np.log10(np.min(non_zero_distances)), np.log10(np.max(non_zero_distances)), num=21)
    mean_deviations = np.zeros((len(frame_indices), len(bins) - 1))
    std_deviations = np.zeros((len(frame_indices), len(bins) - 1))

    for i in range(len(bins) - 1):
        bin_mask = (non_zero_distances >= bins[i]) & (non_zero_distances < bins[i + 1])
        if np.any(bin_mask):
            for idx, frame in enumerate(frame_indices):
                mean_deviations[idx, i] = np.mean(non_zero_deviations[frame, bin_mask], axis=0)
                std_deviations[idx, i] = np.std(non_zero_deviations[frame, bin_mask], axis=0)

    # Create a colormap normalization object
    norm = Normalize(vmin=0, vmax=len(frame_indices) - 1)
    colormap = viridis

    # Plot error bars for each specified frame on a log-log plot
    plt.figure(figsize=(10, 6))
    for idx, frame in enumerate(frame_indices):
        color = colormap(norm(idx))
        plt.errorbar(bins[:-1] + np.diff(bins) / 2, mean_deviations[idx, :], yerr=std_deviations[idx, :], fmt='-o', label=f'Frame {frame}', color=color)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Distance (micrometers)')
    plt.ylabel('Voltage Deviation')
    plt.title('Voltage Deviation Over Distance')
    plt.legend(loc='upper right')
    plt.show()

if __name__ == "__main__":
    #filename = '/Users/danielrebbin/Documents/Academia/UvA/Internship/Wes_Files/Data/Stimulation/240725/ephaptic_stimulation_caspfiguration_100hz.raw.h5'
    #filename = '/Users/danielrebbin/Documents/Academia/UvA/Internship/Wes_Files/Data/Stimulation/240801/ephaptic_stimulation_caspfiguration_100hz_sub_100mV.raw.h5'
    filename = '/Users/danielrebbin/Documents/Academia/UvA/Internship/Wes_Files/Data/Stimulation/Artefact/stimulation_artifact_test_fix.raw.h5'



    well_no = 0
    recording_no = 0
    start_time = 100
    time_length = 100
    stimulation_electrode_id = 13310
    sf = 20000
    frame_indices = list(range(0, 20, 1))  # Example list of frames [0, 10, 20, ..., 90]
    start_frame = int(start_time*sf)
    frame_length = int(time_length*sf)

    X, t, sf, event_data, layout = load_raw(filename, well_no, recording_no, start_frame, frame_length)
    deviations, distances = calculate_deviation(X, sf, event_data, start_frame, stimulation_electrode_id, frame_indices)

    plot_raw(X, t, event_data)
    plot_heatmap(deviations, distances, frame_indices, sf)
    plot_error_bars(deviations, distances, frame_indices, sf)
