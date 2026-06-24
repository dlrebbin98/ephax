import h5py
import pytest

from ephax.lfp_wavefront_cache import (
    WavefrontCacheConfig,
    require_wavefront_caches,
    wavefront_cache_status,
)


def test_wavefront_cache_config_canonical_names():
    config = WavefrontCacheConfig(
        source="stim_removal_null_lfp",
        profile="confirm",
        div=23,
        band_low_hz=30.0,
        band_high_hz=50.0,
        fs_hz=500.0,
        lambda_min_mm=0.5,
        lambda_max_mm=10.0,
        radial=True,
    )

    assert (
        config.wavefront_filename(4)
        == "wavefront_stim_removal_null_lfp_confirm_well4_DIV23_data0004_burst_30-50Hz_fs500_lambda0.5-10_radial1.h5"
    )
    assert (
        config.phasor_feature_filename(4)
        == "phasor_features_stim_removal_null_lfp_confirm_well4_DIV23_data0004_30-50Hz_fs500.h5"
    )


def test_require_wavefront_caches_rejects_noncanonical_or_incomplete(tmp_path):
    config = WavefrontCacheConfig(
        source="stim_removal_null_lfp",
        profile="confirm",
        div=23,
        band_low_hz=30.0,
        band_high_hz=50.0,
        fs_hz=500.0,
        lambda_min_mm=0.5,
        lambda_max_mm=10.0,
        radial=True,
    )
    canonical = config.wavefront_path(tmp_path, 0)

    assert wavefront_cache_status(canonical) == (False, "missing")
    with pytest.raises(FileNotFoundError):
        require_wavefront_caches(config, tmp_path, [0], require_all=True)

    with h5py.File(canonical, "w") as h5:
        h5.create_group("wavefront_calibration")
        h5.create_group("wavefront_events")
        h5.create_group("wavefront_local")
        h5.attrs["complete"] = True

    assert wavefront_cache_status(canonical) == (True, "ok")
    assert require_wavefront_caches(config, tmp_path, [0], require_all=True) == {0: canonical}
