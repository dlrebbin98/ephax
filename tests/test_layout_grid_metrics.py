import numpy as np
import pytest

from ephax import Recording
from ephax.metrics.layout_grid import (
    DEFAULT_ARRAY_X_MAX_UM,
    DEFAULT_ARRAY_Y_MAX_UM,
    compute_grid_avghz_per_recording,
    compute_grid_avghz_pooled,
)


def _recording(times=(0.0, 0.1, 0.2, 0.3), electrodes=(101, 101, 102, 103), end_time=1.0):
    spikes = {
        "time": np.asarray(times, dtype=float),
        "channel": np.asarray([1, 1, 2, 3][: len(times)], dtype=int),
        "amplitude": np.ones(len(times)),
        "electrode": np.asarray(electrodes, dtype=int),
    }
    layout = {
        "channel": np.array([1, 2, 3]),
        "electrode": np.array([101, 102, 103]),
        "x": np.array([25.0, 75.0, 125.0]),
        "y": np.array([25.0, 25.0, 75.0]),
    }
    return Recording(spikes=spikes, layout=layout, start_time=0.0, end_time=float(end_time), sf=1000.0)


def test_compute_grid_avghz_pooled_full_array_extents():
    result = compute_grid_avghz_pooled([_recording()], grid_size=50.0, interpolate=False)

    assert result.grid.shape == (77, 42)
    assert result.x_min == 0.0
    assert result.x_max == DEFAULT_ARRAY_X_MAX_UM
    assert result.y_min == 0.0
    assert result.y_max == DEFAULT_ARRAY_Y_MAX_UM
    assert result.grid[0, 0] == 2.0
    assert result.grid[1, 0] == 1.0
    assert result.grid[2, 1] == 1.0


def test_compute_grid_avghz_per_recording_returns_one_grid_each():
    results = compute_grid_avghz_per_recording([_recording(), _recording(end_time=2.0)], grid_size=50.0)

    assert len(results) == 2
    assert results[0].grid.shape == results[1].grid.shape
    assert results[0].grid[0, 0] == 2.0
    assert results[1].grid[0, 0] == 1.0


def test_compute_grid_interpolation_fills_empty_cells_when_possible():
    result = compute_grid_avghz_pooled([_recording()], grid_size=50.0, interpolate=True)

    assert np.isfinite(result.grid[1, 1])


def test_compute_grid_avghz_pooled_rejects_silent_recordings():
    silent = _recording(times=(), electrodes=())

    with pytest.raises(ValueError, match="No firing data"):
        compute_grid_avghz_pooled([silent])
