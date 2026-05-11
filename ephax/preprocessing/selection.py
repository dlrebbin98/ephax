from __future__ import annotations

import numpy as np
import pandas as pd


def get_activity_sorted_electrodes(spikes_data_list, start=0, stop=None, start_time=0, end_time=np.inf):
    spike_counts = {}

    for spikes_data in spikes_data_list:
        spikes_df = pd.DataFrame(spikes_data)
        spikes_data_during = spikes_df[(spikes_df["time"] > start_time) & (spikes_df["time"] < end_time)]

        for electrode in spikes_data_during["electrode"]:
            spike_counts[electrode] = spike_counts.get(electrode, 0) + 1

    sorted_electrodes = sorted(spike_counts.items(), key=lambda item: item[1], reverse=True)

    if stop is None or stop > len(sorted_electrodes):
        stop = len(sorted_electrodes)

    return [electrode for electrode, _count in sorted_electrodes[start:stop]]
