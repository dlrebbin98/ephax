import h5py
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import chi2

def load_spikes_raw(filename, well_no, recording_no=0):
    # max_allowed_block_size = 40000
    # assert(block_size<=max_allowed_block_size)
    h5_file = h5py.File(filename, 'r')

    # Get the stimulated electrode number
    if 'electrodes' in h5_file['assay']['inputs']:
        stimulus_electrode = h5_file['assay']['inputs']['electrodes'][0]
        if isinstance(stimulus_electrode, np.bytes_):
            stimulus_electrode = stimulus_electrode.decode('utf-8')
        stimulus_electrode = int(re.search(r'"stim_must_include":"(\d+)"', stimulus_electrode).group(1))
    else:
        stimulus_electrode = 0
        print("stimulus_electrode does not exist, assigning default value 0.")

    h5_object = h5_file['wells']['well{0:0>3}'.format(well_no)]['rec{0:0>4}'.format(recording_no)]
    sf = h5_object['settings']['sampling'][0]

    # Load spikes data
    frameno = np.array(h5_object['spikes']['frameno'])
    channel = np.array(h5_object['spikes']['channel'])
    amplitude = np.array(h5_object['spikes']['amplitude'])
    

    # Correct for arbitrary frame numbers
    first_frame = min(frameno)
    time = (frameno - first_frame)/sf

    # Extract relevant columns
    spikes_data = {
        'time': time[:],
        'channel': channel[:],
        'amplitude': amplitude[:]
    }
    
        # Extract electrode mapping
    mapping = h5_object['settings']['mapping']
    channel_map = np.array(mapping['channel'])
    electrode_map = np.array(mapping['electrode'])

    layout = {
        'channel': channel_map[:],
        'electrode': electrode_map[:],
        'x': np.array(mapping['x'])[:],
        'y': np.array(mapping['y'])[:]
    }

    # Create a channel to electrode mapping
    channel_to_electrode = {ch: el for ch, el in zip(channel_map, electrode_map)}

    # Add electrode information to spikes_data
    spikes_data['electrode'] = np.array([channel_to_electrode.get(ch, None) for ch in spikes_data['channel']])
    valid_indices = np.where(spikes_data['electrode'] != None)[0]
    spikes_data = {
        key: np.array(value)[valid_indices] for key, value in spikes_data.items()
    }
    print(np.unique(spikes_data['electrode']), np.unique(['channel']))

    # Load event data
    events = h5_object['events']

    frameno = np.array(events['frameno'])
    eventtype = np.array(events['eventtype'])
    eventid = np.array(events['eventid'])
    eventmessage = np.array(events['eventmessage'])
    time = (frameno - first_frame)/sf

    event_data = {
        'time': time[:],
        'eventtype': eventtype[:],
        'eventid': eventid[:],
        'eventmessage': eventmessage[:]
    }


    return spikes_data, event_data, layout, sf, stimulus_electrode

def load_raw(filename, well_no, recording_no, start_frame, block_size):
    # The maximum allowed block size can be increased if needed,
    # However, if the block size is too large, handling of the data in Python gets too slow.
    max_allowed_block_size = 4000000
    assert(block_size <= max_allowed_block_size)
    h5_file = h5py.File(filename, 'r')
    h5_object = h5_file['wells']['well{0:0>3}'.format(well_no)]['rec{0:0>4}'.format(recording_no)]

    # Load settings from file
    lsb = h5_object['settings']['lsb'][0]
    sf = h5_object['settings']['sampling'][0]

    # compute time vector
    time = np.arange(start_frame, start_frame + block_size) / sf

    # Load raw data from file
    groups = h5_object['groups']
    group0 = groups[next(iter(groups))]
    first_frame = min(np.array(group0['frame_nos']))

    # Get the total number of frames
    total_frames = group0['raw'].shape[1]
    print(f"Total frames available: {total_frames}")

    # Adjust block size if it exceeds the total frames
    if start_frame + block_size > total_frames:
        block_size = total_frames - start_frame
        print(f"Adjusted block size: {block_size}")

    # events are stored in even wells 
    event_object = h5_file['wells']['well{0:0>3}'.format(well_no - well_no % 2)]['rec{0:0>4}'.format(recording_no)]
    
    events = event_object['events']
    frameno = np.array(events['frameno'])
    frameno -= first_frame
    eventtype = np.array(events['eventtype'])
    eventid = np.array(events['eventid'])
    eventmessage = np.array(events['eventmessage'])
    eventtime = frameno / sf

    # Filter events within the specific time range (600s to 601s)
    time_range_start = int(start_frame/sf)
    time_range_end = int((start_frame + block_size)/sf)
    event_mask = (eventtime >= time_range_start) & (eventtime <= time_range_end)

    event_data = {
        'time': eventtime[event_mask],
        'eventtype': eventtype[event_mask],
        'eventid': eventid[event_mask],
        'eventmessage': eventmessage[event_mask], 
        'frameno': frameno[event_mask]
    }

    # Extract electrode mapping
    mapping = h5_object['settings']['mapping']
    channel = np.array(mapping['channel'])
    electrode = np.array(mapping['electrode'])
    x = np.array(mapping['x'])
    y = np.array(mapping['y'])

    layout = {
        'channel': channel[:],
        'electrode': electrode[:],
        'x': x[:],
        'y': y[:]
    }

    # Create a mask for valid channels based on the layout
    valid_channels_mask = np.isin(np.arange(1024), layout['channel'])

    # Select only the valid channels from the raw data
    X = group0['raw'][valid_channels_mask, start_frame:start_frame + block_size].T * lsb

    return X, time, sf, event_data, layout


def load_spikes(filename, well_no, min_amp=10):
    # Open the new HDF5 file
    with h5py.File(filename, 'r') as f:
        # Access the specified well
        well_key = f'well{well_no:0>3}'
        if well_key not in f['wells']:
            raise ValueError(f'Well {well_no} not found in the file.')
        
        well_group = f['wells'][well_key]
        
        # Load spikes_data
        spikes_data = {}
        spikes_group = well_group['spikes']
        for key in spikes_group.keys():
            spikes_data[key] = spikes_group[key][:]
        
        # Filter spikes by minimum amplitude
        amplitudes = np.abs(spikes_data['amplitude'])
        mask = amplitudes >= min_amp
        # Apply mask to each field in spikes_data
        for field in list(spikes_data.keys()):
            spikes_data[field] = spikes_data[field][mask]
        
        # Load event_data
        event_data = {}
        events_group = well_group['events']
        for key in events_group.keys():
            event_data[key] = events_group[key][:]
        
        # Load layout
        layout = {}
        layout_group = well_group['layout']
        for key in layout_group.keys():
            layout[key] = layout_group[key][:]
        
        # Load sf and stimulus_electrode
        sf = well_group['sf'][()]
        stimulus_electrode = f['stimulus_electrode'][()]
        first_frame = well_group['first_frame'][()]
        
    return spikes_data, event_data, layout, sf, stimulus_electrode

def load_spikes_data(file_info, min_amp=0):
    """
    Load spikes data and layout for each file.
    
    """
    spikes_data_list = []
    layout_list = []
    start_times = []
    end_times = []

    for folder, filename, start_time, end_time, well in file_info:
        directory = f'/Users/danielrebbin/Documents/Academia/UvA/Internship/Wes_Files/Data/Stimulation/{folder}/'
        path = directory + filename
        spikes_data, _, layout, sf, _ = load_spikes(path, well, min_amp=min_amp)
        spikes_data_list.append(spikes_data)
        layout_list.append(layout)
        start_times.append(start_time)
        end_times.append(end_time)

    return sf, spikes_data_list, layout_list, start_times, end_times

def load_spikes_npz(file_info, min_amp=0):
    """
    Load spikes data and layout for each file based on file_info.
    
    Parameters:
        file_info (list): List of tuples containing (div, well, start_time, end_time).
    
    Returns:
        sf (float): Sampling frequency.
        spikes_data_list (list): List of spikes data dictionaries for each well.
        layout_list (list): List of layout dictionaries for each well.
        start_times (list): List of start times for each well.
        end_times (list): List of end times for each well.
    """
    wd = "/Users/danielrebbin/Documents/Academia/UvA/Internship/Wes_Files/Data/Wave Training Analysis/ProcessedData/"
    spikes_data_list = []
    layout_list = []
    start_times = []
    end_times = []

    for div, start_time, end_time, well in file_info:
        # Construct the file path
        filename = f"DIV{div}_stim_removal_well_{well}_exp_data.npz"
        path = f"{wd}{filename}"
        
        # Load the data
        data = np.load(path)
        sf = data['samp_rate']
        
        # Extract layout
        layout = {
            'channel': data['channelmap'][:, 0],
            'electrode': data['channelmap'][:, 2],
            'x': data['channelmap'][:, 3],
            'y': data['channelmap'][:, 4]
        }
        
        # Create a channel-to-electrode mapping
        channel_to_electrode = {ch: el for ch, el in zip(layout['channel'], layout['electrode'])}
        
        # Extract spikes data
        times = data['spike_data']['frameno'] / sf
        channels = data['spike_data']['channel']
        amplitudes = np.abs(data['spike_data']['amplitude'])
        electrodes = [channel_to_electrode.get(ch, None) for ch in channels]

        # Build masks: amplitude threshold and valid electrode mapping
        amp_mask = amplitudes >= min_amp
        valid_elec_mask = np.array([e is not None for e in electrodes])
        mask = amp_mask & valid_elec_mask

        # Apply mask and ensure numeric dtype for electrodes
        spikes_data = {
            'time': np.asarray(times)[mask],
            'channel': np.asarray(channels)[mask],
            'amplitude': np.asarray(amplitudes)[mask],
            'electrode': np.asarray([int(e) for e in np.asarray(electrodes)[mask]])
        }
        
        # Append to lists
        spikes_data_list.append(spikes_data)
        layout_list.append(layout)
        start_times.append(start_time)
        end_times.append(end_time)

    return sf, spikes_data_list, layout_list, start_times, end_times

def get_activity_sorted_electrodes(spikes_data_list, start=0, stop=None, start_time=0, end_time=np.Inf):
    # Initialize a dictionary to store spike counts across all recordings
    spike_counts = {}

    # Iterate through each spikes_data in the provided list
    for spikes_data in spikes_data_list:
        spikes_df = pd.DataFrame(spikes_data)

        # Filter spikes within the specified time window
        spikes_data_during = spikes_df[(spikes_df['time'] > start_time) & (spikes_df['time'] < end_time)]

        # Count the number of spikes per electrode
        for electrode in spikes_data_during['electrode']:
            if electrode in spike_counts:
                spike_counts[electrode] += 1
            else:
                spike_counts[electrode] = 1

    # Sort electrodes by spike count
    sorted_electrodes = sorted(spike_counts.items(), key=lambda item: item[1], reverse=True)

    # Set stop to a valid number
    if stop is None or stop > len(sorted_electrodes):
        stop = len(sorted_electrodes)

    # Return the top N electrodes based on start and stop indices
    most_active_electrodes = [electrode for electrode, count in sorted_electrodes[start:stop]]
    return most_active_electrodes

def assign_r_distance(spikes_df, layout_df, ref_electrode):
    if ref_electrode not in layout_df['electrode'].values:
        raise ValueError(f"No data found for electrode: {ref_electrode}")
    coords = layout_df.loc[layout_df['electrode'] == ref_electrode, ['x', 'y']].iloc[0]

    layout_df['distance'] = np.sqrt((layout_df['x'] - coords['x'])**2 + (layout_df['y'] - coords['y'])**2)

    channel_to_distance = layout_df.set_index('channel')['distance'].to_dict()
    channel_to_electrode = layout_df.set_index('channel')['electrode'].to_dict()

    spikes_df['distance'] = spikes_df['channel'].map(channel_to_distance)
    spikes_df['electrode'] = spikes_df['channel'].map(channel_to_electrode)
    spikes_df = spikes_df.dropna(subset=['distance', 'electrode'])

    return spikes_df, layout_df

def assign_r_distance_all(spikes_df, layout_df, ref_electrodes):
    valid_ref_electrodes = [e for e in ref_electrodes if e in layout_df['electrode'].values]
    if not valid_ref_electrodes:
        raise ValueError("None of the ref_electrodes are found in layout_df")
    
    # Get coordinates of electrodes
    electrode_coords = layout_df[['electrode', 'x', 'y']]
    # Get coordinates of valid ref_electrodes
    ref_coords = layout_df[layout_df['electrode'].isin(valid_ref_electrodes)][['electrode', 'x', 'y']]
    ref_coords = ref_coords.rename(columns={'electrode': 'ref_electrode', 'x': 'ref_x', 'y': 'ref_y'})
    
    # Create all combinations of electrodes and ref_electrodes
    electrode_coords['key'] = 1
    ref_coords['key'] = 1
    distances_df = pd.merge(electrode_coords, ref_coords, on='key').drop('key', axis=1)
    distances_df = distances_df[distances_df['electrode'] != distances_df['ref_electrode']]
    
    # Compute distances
    distances_df['distance'] = np.sqrt((distances_df['x'] - distances_df['ref_x'])**2 + (distances_df['y'] - distances_df['ref_y'])**2)
    
    # Map channels to electrodes
    channel_to_electrode = layout_df.set_index('channel')['electrode'].to_dict()
    spikes_df['electrode'] = spikes_df['channel'].map(channel_to_electrode)
    
    # Remove spikes with electrodes not in layout_df
    spikes_df = spikes_df.dropna(subset=['electrode'])
    
    return spikes_df, distances_df

def assign_r_theta_distance(spikes_df, layout_df, ref_electrode):
    spikes_df, layout_df = assign_r_distance(spikes_df, layout_df, ref_electrode)
    
    coords = layout_df.loc[layout_df['electrode'] == ref_electrode, ['x', 'y']].iloc[0]

    layout_df['theta'] = np.arctan2(layout_df['y'] - coords['y'], layout_df['x'] - coords['x'])
    channel_to_theta = layout_df.set_index('channel')['theta'].to_dict()
    spikes_df['theta'] = spikes_df['channel'].map(channel_to_theta)

    return spikes_df, layout_df

def log_likelihood(residuals, n_params):
    # Calculate the log-likelihood assuming Gaussian residuals
    sigma2 = np.var(residuals)
    n = len(residuals)
    log_likelihood = -0.5 * n * np.log(2 * np.pi * sigma2) - np.sum(residuals**2) / (2 * sigma2)
    return log_likelihood

def likelihood_ratio_test(logL_full, logL_reduced, df):
    # Compute the LRT statistic
    print(f"LRT(FULL): {logL_full}, LRT(REDUCED): {logL_reduced}")
    LRT_stat = -2 * (logL_reduced - logL_full)
    # Compute the p-value using chi-square distribution
    print(f"DF: {df}")
    p_value = chi2.sf(LRT_stat, df)
    return LRT_stat, p_value

# Visualization helpers
def truncate_colormap(cmap, minval: float = 0.0, maxval: float = 1.0, n: int = 100):
    """Return a truncated copy of a colormap between [minval, maxval]."""
    return LinearSegmentedColormap.from_list(
        f"truncated({getattr(cmap, 'name', 'cmap')},{minval:.2f},{maxval:.2f})",
        cmap(np.linspace(minval, maxval, n)),
    )
