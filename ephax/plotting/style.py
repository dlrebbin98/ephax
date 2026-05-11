from __future__ import annotations

import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def truncate_colormap(cmap, minval: float = 0.0, maxval: float = 1.0, n: int = 100):
    """Return a truncated copy of a colormap between [minval, maxval]."""
    return LinearSegmentedColormap.from_list(
        f"truncated({getattr(cmap, 'name', 'cmap')},{minval:.2f},{maxval:.2f})",
        cmap(np.linspace(minval, maxval, n)),
    )
