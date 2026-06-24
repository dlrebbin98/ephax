from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py


REQUIRED_WAVEFRONT_GROUPS = frozenset(
    ("wavefront_calibration", "wavefront_events", "wavefront_local")
)


@dataclass(frozen=True)
class WavefrontCacheConfig:
    source: str
    profile: str
    div: int
    band_low_hz: float
    band_high_hz: float
    fs_hz: float
    lambda_min_mm: float
    lambda_max_mm: float
    radial: bool
    interval_mode: str = "burst"

    def dataset_for_well(self, well: int) -> str:
        return f"data{int(well):04d}"

    @staticmethod
    def _num_token(value: float) -> str:
        return f"{float(value):g}"

    @property
    def band_token(self) -> str:
        return f"{self._num_token(self.band_low_hz)}-{self._num_token(self.band_high_hz)}Hz"

    @property
    def fs_token(self) -> str:
        return f"fs{self._num_token(self.fs_hz)}"

    @property
    def lambda_token(self) -> str:
        return f"lambda{self._num_token(self.lambda_min_mm)}-{self._num_token(self.lambda_max_mm)}"

    def wavefront_filename(self, well: int) -> str:
        well = int(well)
        dataset = self.dataset_for_well(well)
        return (
            f"wavefront_{self.source}_{self.profile}_well{well}_DIV{int(self.div)}_"
            f"{dataset}_{self.interval_mode}_{self.band_token}_{self.fs_token}_"
            f"{self.lambda_token}_radial{int(bool(self.radial))}.h5"
        )

    def wavefront_path(self, cache_dir: str | Path, well: int) -> Path:
        return Path(cache_dir) / self.wavefront_filename(well)

    def phasor_feature_filename(self, well: int) -> str:
        well = int(well)
        dataset = self.dataset_for_well(well)
        return (
            f"phasor_features_{self.source}_{self.profile}_well{well}_DIV{int(self.div)}_"
            f"{dataset}_{self.band_token}_{self.fs_token}.h5"
        )

    def phasor_feature_path(self, cache_dir: str | Path, well: int) -> Path:
        return Path(cache_dir) / self.phasor_feature_filename(well)


def wavefront_cache_status(path: str | Path) -> tuple[bool, str]:
    path = Path(path)
    if not path.exists():
        return False, "missing"
    try:
        with h5py.File(path, "r") as h5:
            missing = REQUIRED_WAVEFRONT_GROUPS - set(h5.keys())
            if missing:
                return False, f"missing groups {sorted(missing)}"
            if "complete" in h5.attrs and not bool(h5.attrs["complete"]):
                return False, "marked incomplete"
    except OSError as exc:
        return False, str(exc)
    return True, "ok"


def require_wavefront_caches(
    config: WavefrontCacheConfig,
    cache_dir: str | Path,
    wells: list[int] | tuple[int, ...],
    *,
    require_all: bool = True,
) -> dict[int, Path]:
    paths: dict[int, Path] = {}
    diagnostics: list[str] = []
    for well in wells:
        well = int(well)
        path = config.wavefront_path(cache_dir, well)
        ok, reason = wavefront_cache_status(path)
        if ok:
            paths[well] = path
        else:
            diagnostics.append(f"well {well}: {path} [{reason}]")
    if (require_all and len(paths) != len(wells)) or not paths:
        raise FileNotFoundError(
            "Missing compatible canonical wavefront caches:\n" + "\n".join(diagnostics)
        )
    return paths
