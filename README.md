# ephax: Analyzer-first electrophysiology toolkit

This repository provides a small set of analyzers to explore resting activity
and stimulation experiments on large MEA recordings, specifically testing
ephaptic-axonal (ephax) interaction effects. The design is
analyzer-first: you load a dataset once, select reference electrodes via a
PrepConfig, and then use analyzer classes to compute and visualize results.

## Quick Start

```python
from ephax import RestingActivityDataset, PrepConfig
from ephax import IFRAnalyzer, IFRConfig
from ephax import CofiringTemporalAnalyzer, CofiringTemporalConfig
from ephax.analyzers.firing_distance import FiringDistanceAnalyzer

# Build dataset from your file_info (see helper_functions for loaders)
ds = RestingActivityDataset.from_file_info(file_info, source='h5', min_amp=0)

# Choose reference electrodes per recording
refs_cfg = PrepConfig(mode='top', top_start=10, top_stop=110, top_use_recording_window=True, verbose=True)

# IFR: histogram with GMM overlay + per-recording time-series heatmaps
ifr_cfg = IFRConfig(log_scale=True, hist_bins=100, overlay_gmm=True, time_grid_hz=100.0, max_time_points=5000)
ifr = IFRAnalyzer.from_dataset(ds, config=ifr_cfg, selection_prep_config=refs_cfg)
ifr.plot_histogram(show=True)
ifr.plot_timeseries()  # uses the same selection

# Co-firing: averaged heatmap and GIFs
cft_cfg = CofiringTemporalConfig(start_ms=-20, stop_ms=300, step_ms=10, normalize=False)
cft = CofiringTemporalAnalyzer(ds, cft_cfg, selection_prep_config=refs_cfg)
cft.plot_avg_cofiring_heatmap()
# cft.create_theta_gif('theta.gif'); cft.create_grid_gif('grid.gif')

# Distance analyses
fd = FiringDistanceAnalyzer(ds, selection_prep_config=refs_cfg)
cof = fd.cofiring_avg_vs_distance(plusminus_ms=1.0)
fd.plot_cofiring_with_synergy(cof)
fr = fd.avg_rate_vs_distance()
fd.plot_rate_with_synergy(fr)
dist_vals, weights = fd.distance_histogram(finite_size_correction=True)
fd.plot_distance_hist_with_synergy(dist_vals, weights)
```

## Configs

- `PrepConfig`: how to select references per recording
  - `mode`: `"threshold"` | `"top"` | `"selected"`
  - `top_start`, `top_stop`, `top_use_recording_window`, etc.

- `IFRConfig`: IFR analysis/plot options
  - `log_scale`: plot and fit in log10(Hz)
  - `hist_bins`: histogram bins for aggregate IFR
  - `overlay_gmm`: show GMM components and their sum
  - `time_grid_hz`: resampling rate for IFR heatmaps (decoupled from sf)
  - `max_time_points`: cap columns in heatmaps to bound memory

- `CofiringTemporalConfig`: co-firing temporal analysis options
  - `start_ms`, `stop_ms`, `step_ms`: delay axis
  - `normalize`: optional t0 normalization on plots

## Notes

- Parallelism uses threads (`joblib` with `prefer="threads"`) to avoid
  copying large data structures. If needed, expose an `n_jobs` knob.
- All analyzers respect per-recording windows from your dataset; we do not
  pool outside the defined ranges.
- The codebase prefers analyzer entry points; legacy wrappers in
  `resting_activity.py` have been removed.

