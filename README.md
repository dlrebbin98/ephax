# ephax: functional electrophysiology toolkit

This repository contains composable helpers for MEA resting-activity,
stimulation, spike-wave, LFP-wave, and ephaptic-axonal analyses. The codebase is
organized by responsibility:

- `ephax.data_io`: file loading and validation
- `ephax.preprocessing`: recording containers, electrode selection, and geometry helpers
- `ephax.metrics`: pure computations
- `ephax.modeling`: theoretical and simulation models
- `ephax.plotting`: rendering functions that consume precomputed objects
- `ephax.artifacts`: checkpoint and figure export helpers

## Installation

```bash
pip install -e ".[dev,notebook]"
pytest
```

## Reproducible Workflows

YAML workflow configs live in `configs/`. They avoid hard-coded local data
paths by using explicit `dataset.data_root` values or the `EPHAX_DATA_ROOT`
environment variable.

```bash
ephax configs/stim_removal_null.yaml
```

Generated figures, GIFs, and analysis outputs should go under `outputs/`.
Large recordings should stay outside the Python package; keep only small sample
fixtures in the repo.

## Quick Start

```python
from ephax import PrepConfig, RestingActivityDataset
from ephax.metrics.ifr import IFRConfig, ifr_peaks, prepare_ifr_timeseries_panels
from ephax.metrics.firing_distance import avg_rate_vs_distance, cofiring_avg_vs_distance
from ephax.metrics.layout_grid import compute_grid_avghz_per_recording
from ephax.modeling.gmm import fit_ifr_gmm
from ephax.plotting.ifr import plot_ifr_histogram, plot_ifr_timeseries_panels
from ephax.plotting.layout_grid import draw_grid_avghz_panel

ds = RestingActivityDataset.from_file_info(file_info, source="h5", min_amp=0)
prep = PrepConfig(mode="top", top_start=10, top_stop=110, top_use_recording_window=True, verbose=False)
refs = ds.select_ref_electrodes(prep)
spikes_list = [rec.spikes for rec in ds.recordings]
start_times = [rec.start_time for rec in ds.recordings]
end_times = [rec.end_time for rec in ds.recordings]

ifr_cfg = IFRConfig(log_scale=True, overlay_gmm=True, time_grid_hz=100.0, max_time_points=5000)
peaks = ifr_peaks(spikes_list, start_times, end_times, log_scale=ifr_cfg.log_scale, selected_refs_per_recording=refs)
fit = fit_ifr_gmm(peaks.values, log_scale=ifr_cfg.log_scale) if ifr_cfg.overlay_gmm else None
fig_hist, ax_hist = plot_ifr_histogram(peaks, ifr_cfg, fit=fit, hist_bins=100)

panels = prepare_ifr_timeseries_panels(
    spikes_list,
    start_times,
    end_times,
    refs,
    log_scale=ifr_cfg.log_scale,
    time_grid_hz=ifr_cfg.time_grid_hz,
    max_time_points=ifr_cfg.max_time_points,
)
figures = plot_ifr_timeseries_panels(panels, ifr_cfg)

grids = compute_grid_avghz_per_recording(ds.recordings, grid_size=50.0, interpolate=False)
rate_by_distance = avg_rate_vs_distance(ds.recordings, refs)
cofiring_by_distance = cofiring_avg_vs_distance(ds.recordings, refs, plusminus_ms=1.0)
```

The package intentionally no longer exposes analyzer classes. Build analyses by
combining metric functions, plotting functions, and artifact writers explicitly.
