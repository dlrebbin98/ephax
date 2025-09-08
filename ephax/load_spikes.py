import h5py

def load_spikes(path, well_no):

    """
    Load spike data and associated information from an HDF5 file for a specified well.

    This function opens an HDF5 file at the given path and loads spike data, event data,
    layout information, sampling frequency, and stimulus electrode information for the specified well.

    Parameters:
        path (str): The file path to the HDF5 file containing the data.
        well_no (int): The well number to access in the file. Should be an integer corresponding to the well index.

    Returns:
        tuple: A tuple containing the following elements:
            - spikes_data (dict): A dictionary where keys are time, channel and amplitude
            - event_data (dict): A dictionary where keys are time eventtype, eventid and eventmessage
            - layout (dict): A dictionary where keys are channel (enumerated id), electrode (location-specific id), x and y (both np.arrays)
            - sf (float): The sampling frequency.
            - stimulus_electrode (int): The electrode id(s) used for stimulation. 0 if none were stimulated or error in storing it.

    Raises:
        ValueError: If the specified well number is not found in the HDF5 file.

    Example:
        >>> spikes_data, event_data, layout, sf, stimulus_electrode = load_spikes('data/control_0.raw.h5', 0)
    """
    
    # Open the new HDF5 file
    with h5py.File(path, 'r') as f:
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
        sf = well_group['sf'][()] # sampling frequency
        stimulus_electrode = f['stimulus_electrode'][()]
        
    return spikes_data, event_data, layout, sf, stimulus_electrode

if __name__ == "__main__":

    # Example usage
    well = 0 # 0 or 1 for the control recording
    path = '.../control_0.raw.h5'

    spikes_data, event_data, layout, sf, stimulus_electrode = load_spikes(path, well)