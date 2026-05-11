from __future__ import annotations

import numpy as np
import pandas as pd


def assign_r_distance(spikes_df: pd.DataFrame, layout_df: pd.DataFrame, ref_electrode):
    if ref_electrode not in layout_df["electrode"].values:
        raise ValueError(f"No data found for electrode: {ref_electrode}")
    coords = layout_df.loc[layout_df["electrode"] == ref_electrode, ["x", "y"]].iloc[0]

    layout_df["distance"] = np.sqrt((layout_df["x"] - coords["x"]) ** 2 + (layout_df["y"] - coords["y"]) ** 2)

    channel_to_distance = layout_df.set_index("channel")["distance"].to_dict()
    channel_to_electrode = layout_df.set_index("channel")["electrode"].to_dict()

    spikes_df["distance"] = spikes_df["channel"].map(channel_to_distance)
    spikes_df["electrode"] = spikes_df["channel"].map(channel_to_electrode)
    spikes_df = spikes_df.dropna(subset=["distance", "electrode"])

    return spikes_df, layout_df


def assign_r_distance_all(spikes_df: pd.DataFrame, layout_df: pd.DataFrame, ref_electrodes):
    valid_ref_electrodes = [e for e in ref_electrodes if e in layout_df["electrode"].values]
    if not valid_ref_electrodes:
        raise ValueError("None of the ref_electrodes are found in layout_df")

    electrode_coords = layout_df[["electrode", "x", "y"]].copy()
    ref_coords = layout_df[layout_df["electrode"].isin(valid_ref_electrodes)][["electrode", "x", "y"]].copy()
    ref_coords = ref_coords.rename(columns={"electrode": "ref_electrode", "x": "ref_x", "y": "ref_y"})

    electrode_coords["key"] = 1
    ref_coords["key"] = 1
    distances_df = pd.merge(electrode_coords, ref_coords, on="key").drop("key", axis=1)
    distances_df = distances_df[distances_df["electrode"] != distances_df["ref_electrode"]]

    distances_df["distance"] = np.sqrt(
        (distances_df["x"] - distances_df["ref_x"]) ** 2
        + (distances_df["y"] - distances_df["ref_y"]) ** 2
    )

    channel_to_electrode = layout_df.set_index("channel")["electrode"].to_dict()
    spikes_df["electrode"] = spikes_df["channel"].map(channel_to_electrode)
    spikes_df = spikes_df.dropna(subset=["electrode"])

    return spikes_df, distances_df


def assign_r_theta_distance(spikes_df: pd.DataFrame, layout_df: pd.DataFrame, ref_electrode):
    spikes_df, layout_df = assign_r_distance(spikes_df, layout_df, ref_electrode)

    coords = layout_df.loc[layout_df["electrode"] == ref_electrode, ["x", "y"]].iloc[0]
    layout_df["theta"] = np.arctan2(layout_df["y"] - coords["y"], layout_df["x"] - coords["x"])
    channel_to_theta = layout_df.set_index("channel")["theta"].to_dict()
    spikes_df["theta"] = spikes_df["channel"].map(channel_to_theta)

    return spikes_df, layout_df
