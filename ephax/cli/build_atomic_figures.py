from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..analyzers.ifr import IFRConfig
from ..artifacts import load_ifr_timeseries_panels, save_figure
from ..plotting.ifr import plot_ifr_timeseries_panel


def build_ifr_timeseries_figures(
    run_dir: str | Path,
    *,
    prefix: str = "ifr_timeseries",
    formats: tuple[str, ...] = ("png",),
    ts_bins: int = 50,
) -> list[Path]:
    """Build IFR time-series atomic figures from CSV checkpoints."""
    run_dir = Path(run_dir)
    panels = load_ifr_timeseries_panels(run_dir / "figure_data", prefix=prefix)
    saved: list[Path] = []
    for panel in panels:
        cfg = IFRConfig(log_scale=panel.log_scale, ts_bins=ts_bins)
        fig, _axes = plot_ifr_timeseries_panel(panel, cfg, recording_label=f"recording_{panel.recording_index}")
        saved.extend(
            save_figure(
                fig,
                run_dir / "atomic_figures" / f"{prefix}_recording_{panel.recording_index}",
                formats=formats,
            )
        )
        plt.close(fig)
    return saved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build atomic ephax figures from CSV checkpoints.")
    parser.add_argument("run_dir", help="Run directory containing figure_data/ checkpoints.")
    parser.add_argument("--prefix", default="ifr_timeseries", help="Checkpoint filename prefix.")
    parser.add_argument("--formats", nargs="+", default=["png"], help="Figure formats to write.")
    parser.add_argument("--ts-bins", type=int, default=50, help="Histogram bins for IFR time-series panels.")
    args = parser.parse_args(argv)

    saved = build_ifr_timeseries_figures(
        args.run_dir,
        prefix=args.prefix,
        formats=tuple(args.formats),
        ts_bins=args.ts_bins,
    )
    for path in saved:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

