import h5py
import numpy as np
import re

def extract_and_save_all_wells(input_filename, output_filename):
    # Open the original HDF5 file
    with h5py.File(input_filename, 'r') as h5_file:
        # Get the list of wells
        wells_group = h5_file['wells']
        well_keys = list(wells_group.keys())
        
        # Get the stimulated electrode number (assuming it's the same for all wells)
        if 'electrodes' in h5_file['assay']['inputs']:
            stimulus_electrode = h5_file['assay']['inputs']['electrodes'][0]
            if isinstance(stimulus_electrode, np.bytes_):
                stimulus_electrode = stimulus_electrode.decode('utf-8')
            stimulus_electrode = int(re.search(r'"stim_must_include":"(\d+)"', stimulus_electrode).group(1))
        else:
            stimulus_electrode = 0
            print("stimulus_electrode does not exist, assigning default value 0.")

        # Open the output HDF5 file
        with h5py.File(output_filename, 'w') as out_file:
            # Save the stimulus_electrode at the root level
            out_file.create_dataset('stimulus_electrode', data=stimulus_electrode)
            
            for well_key in well_keys:
                well_no = int(well_key[-3:])
                print(f"Processing {well_key}...")
                
                # Navigate to the specific recording (assuming recording_no=0)
                h5_object = h5_file['wells'][well_key]['rec{0:0>4}'.format(0)]
                sf = h5_object['settings']['sampling'][0]
                
                # Load spikes data
                frameno = np.array(h5_object['spikes']['frameno'])
                channel = np.array(h5_object['spikes']['channel'])
                amplitude = np.array(h5_object['spikes']['amplitude'])
                
                # Correct for arbitrary frame numbers
                first_frame = np.min(frameno)
                time = (frameno - first_frame) / sf
                
                # Extract relevant columns
                spikes_data = {
                    'time': time,
                    'channel': channel,
                    'amplitude': amplitude
                }
                
                # Extract electrode mapping
                mapping = h5_object['settings']['mapping']
                channel_map = np.array(mapping['channel'])
                electrode_map = np.array(mapping['electrode'])
                x_map = np.array(mapping['x'])
                y_map = np.array(mapping['y'])
                
                layout = {
                    'channel': channel_map,
                    'electrode': electrode_map,
                    'x': x_map,
                    'y': y_map
                }
                
                # Create a channel to electrode mapping
                channel_to_electrode = {ch: el for ch, el in zip(channel_map, electrode_map)}
                
                # Add electrode information to spikes_data
                spikes_data['electrode'] = np.array([channel_to_electrode.get(ch, -1) for ch in spikes_data['channel']])
                valid_indices = np.where(spikes_data['electrode'] != -1)[0]
                spikes_data = {key: value[valid_indices] for key, value in spikes_data.items()}
                
                # Load event data
                events = h5_object['events']
                frameno_events = np.array(events['frameno'])
                eventtype = np.array(events['eventtype'])
                eventid = np.array(events['eventid'])
                # Handle eventmessage (string data)
                eventmessage = np.array(events['eventmessage'], dtype='S')  # Convert to fixed-length bytes
                time_events = (frameno_events - first_frame) / sf
                
                event_data = {
                    'time': time_events,
                    'eventtype': eventtype,
                    'eventid': eventid,
                    'eventmessage': eventmessage
                }
                
                # Save data to the output HDF5 file under the well group
                well_group = out_file.create_group(f'wells/{well_key}')
                # Save spikes_data
                spikes_group = well_group.create_group('spikes')
                for key, value in spikes_data.items():
                    spikes_group.create_dataset(key, data=value, compression='gzip')
                # Save event_data
                events_group = well_group.create_group('events')
                for key, value in event_data.items():
                    # Handle string data types
                    if key == 'eventmessage':
                        dt = h5py.string_dtype(encoding='utf-8')
                        events_group.create_dataset(key, data=value.astype(dt), compression='gzip')
                    else:
                        events_group.create_dataset(key, data=value, compression='gzip')
                # Save layout
                layout_group = well_group.create_group('layout')
                for key, value in layout.items():
                    layout_group.create_dataset(key, data=value, compression='gzip')
                # Save sampling frequency and first_frame
                well_group.create_dataset('sf', data=sf)
                well_group.create_dataset('first_frame', data=first_frame)

if __name__ == "__main__":
    folder = 2407
    new_filename = 'control_1.raw.h5'
    old_filename = 'ephaptic_stimulation_caspfiguration_' + new_filename
    directory = f'/Users/danielrebbin/Documents/Academia/UvA/Internship/Wes_Files/Data/Stimulation/{folder}/'
    old_path = directory + old_filename
    new_path = directory + new_filename
    extract_and_save_all_wells(old_path, new_path)