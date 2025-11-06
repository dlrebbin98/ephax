import os
from pathlib import Path
# Set MKL/OpenMP env vars BEFORE importing any library that may load MKL (numpy/scipy/matplotlib)
os.environ.setdefault("KMP_WARNINGS", "0")  # suppress OpenMP warnings
os.environ.setdefault("MKL_VERBOSE", "0")   # ensure MKL doesn’t chat
os.environ.setdefault("MKL_DEBUG_CPU_TYPE", "5")  # bypass CPU feature deprecation chatter

import matplotlib.pyplot as plt

from ephax.models import CofiringHeatmap as _CofiringHeatmap
from ephax.analyzers import IFRAnalyzer
from ephax import RestingActivityDataset, PrepConfig, Recording, StabilityAnalyzer, StabilityConfig


if __name__ == "__main__":

    start_s = 2400
    end_s = 3000


    # all divs for one well
    well = 4
    file_info = [(div, start_s, end_s, well) for div in [30, 33, 34, 35, 36, 37, 38, 40, 41]]

    # all divs for all wells
    file_info = [(div, start_s, end_s, well) for div, well in zip([30, 33, 34, 35, 36, 37, 38, 40, 41], range(5))]




    file_info = [
        ('2407', 'control_0.raw.h5', 0, 1800, 0),
        ('2407', 'control_1.raw.h5', 0, 1800, 0),
        ('2407', 'control_2.raw.h5', 0, 1800, 0),
        ('2407', 'control_0.raw.h5', 0, 600, 1),
        ('2407', 'control_1.raw.h5', 0, 1800, 1),
        ('2407', 'control_2.raw.h5', 0, 1800, 1),
        ('240725', '50hz.raw.h5', 0, 600, 2),
        ('240726', '50hz.raw.h5', 0, 600, 2),
        ('2407', '50hz_0.raw.h5', 0, 600, 2),
        ('2407', '50hz_1.raw.h5', 0, 600, 2),
        ('240725', '50hz.raw.h5', 0, 600, 3), #13260 instead of 13360
        ('240726', '50hz.raw.h5', 0, 600, 3),
        ('240725', '100hz.raw.h5', 0, 600, 4),
        ('240726', '100hz.raw.h5', 0, 600, 4),
        ('2407', '100hz_0.raw.h5', 0, 600, 4),
        ('2407', '100hz_1.raw.h5', 0, 600, 4),
        ('2407', '100hz_2.raw.h5', 0, 600, 4),
        ('2407', '100hz_1.raw.h5', 0, 600, 5),
        ('2407', '100hz_0.raw.h5', 0, 600, 5),
        ('2407', '100hz_2.raw.h5', 0, 1800, 5),
    ]

    file_info = [
        ('2407', 'control_0.raw.h5', 0, 600, 0),
        #('2407', 'control_1.raw.h5', 0, 1800, 0),
        #('2407', 'control_2.raw.h5', 0, 1800, 0),
        #('2407', 'control_0.raw.h5', 0, 600, 1),
        #('2407', 'control_1.raw.h5', 0, 1800, 1),
        #('2407', 'control_2.raw.h5', 0, 600, 1),
        #('240725', '50hz.raw.h5', 0, 600, 2),
        #('240726', '50hz.raw.h5', 0, 600, 2),
        #('2407', '50hz_0.raw.h5', 0, 600, 2),
        #('2407', '50hz_1.raw.h5', 0, 600, 2),
        #('240725', '50hz.raw.h5', 0, 600, 3), #13260 instead of 13360
        #('240726', '50hz.raw.h5', 0, 600, 3),
        #('240725', '100hz.raw.h5', 0, 600, 4),
        #('240726', '100hz.raw.h5', 0, 600, 4),
        #('2407', '100hz_0.raw.h5', 0, 600, 4),
        #('2407', '100hz_1.raw.h5', 0, 600, 4),
        #('2407', '100hz_2.raw.h5', 0, 600, 4),
        #('2407', '100hz_1.raw.h5', 0, 600, 5),
        #('2407', '100hz_0.raw.h5', 0, 600, 5),
        #('2407', '100hz_2.raw.h5', 0, 1800, 5),
    ]

    file_info = [
        ('2407', 'control_0.raw.h5', 0, 600, 0),
        #('2407', 'control_1.raw.h5', 0, 1800, 0),
        #('2407', 'control_2.raw.h5', 0, 1800, 0),
        #('2407', 'control_0.raw.h5', 0, 600, 1),
        #('2407', 'control_1.raw.h5', 0, 1800, 1),
        ('2407', 'control_2.raw.h5', 0, 600, 1),
        ('240725', '50hz.raw.h5', 0, 600, 2),
        #('240726', '50hz.raw.h5', 0, 600, 2),
        #('2407', '50hz_0.raw.h5', 0, 600, 2),
        #('2407', '50hz_1.raw.h5', 0, 600, 2),
        ('240725', '50hz.raw.h5', 0, 600, 3), #13260 instead of 13360
        #('240726', '50hz.raw.h5', 0, 600, 3),
        ('240725', '100hz.raw.h5', 0, 600, 4),
        #('240726', '100hz.raw.h5', 0, 600, 4),
        #('2407', '100hz_0.raw.h5', 0, 600, 4),
        #('2407', '100hz_1.raw.h5', 0, 600, 4),
        #('2407', '100hz_2.raw.h5', 0, 600, 4),
        ('2407', '100hz_1.raw.h5', 0, 600, 5),
        #('2407', '100hz_0.raw.h5', 0, 600, 5),
        #('2407', '100hz_2.raw.h5', 0, 1800, 5),
    ]

    file_info = [
        ('2407', 'control_0.raw.h5', 0, 600, 0),
        ('2407', 'control_2.raw.h5', 0, 600, 1),
        ('240725', '50hz.raw.h5', 0, 600, 2),
        ('240725', '50hz.raw.h5', 0, 600, 3), 
        ('240725', '100hz.raw.h5', 0, 600, 4),
        ('2407', '100hz_1.raw.h5', 0, 600, 5),
    ]



    # Load recordings; keep all spikes intact (no upfront filtering)
    data_root = Path(os.environ.get('EPHAX_DATA_ROOT', '/Users/danielrebbin/Documents/Academia/UvA/Internship/Wes_Files/Data/Stimulation'))
    ds = RestingActivityDataset.from_file_info(file_info, source='h5', min_amp=0, base_dir=data_root)

    shuffle_layout_coordinates = True  # Also randomize physical coordinates across electrodes
    permute_for_null = True
    num_permutations = 2
    perm_seed = 12345
    perm_datasets: list[RestingActivityDataset] = []
    if permute_for_null:
        for perm_idx in range(num_permutations):
            permuted_recs = []
            for rec_idx, rec in enumerate(ds.recordings):
                if perm_seed is None:
                    seed = None
                else:
                    seed = perm_seed + perm_idx * 1000 + rec_idx
                permuted_recs.append(
                    rec.randomize_electrode_mapping(
                        seed=seed,
                        inplace=False,
                        shuffle_layout_coordinates=shuffle_layout_coordinates,
                    )
                )
            perm_datasets.append(RestingActivityDataset(recordings=permuted_recs, sf=ds.sf))

    # Select references internally in analyzers (no external refs plumbing)
    cfg_refs = PrepConfig(mode='top', top_start=0, top_stop=1000, top_use_recording_window=True, verbose=True)
    cfg_refs = PrepConfig(mode='threshold', active_threshold=0.1, verbose=True)
    #cfg_refs = PrepConfig(mode='selected', selected_refs=[13260])

    # Layout grid plots (avg Hz) with optional interpolation
    from ephax import LayoutGridPlotter
    lg = LayoutGridPlotter(ds)
    lg.plot_grid_avghz_pooled(grid_size=50.0, interpolate=True)
    lg.plot_grid_avghz_panel(grid_size=50.0, ncols=3, interpolate=True)


    from ephax import FiringDistanceAnalyzer
    fd_perm_dataset = perm_datasets[0] if perm_datasets else None
    fd_an = FiringDistanceAnalyzer(
        ds,
        dataset_perm=fd_perm_dataset,
        selection_prep_config=cfg_refs,
        v_eph=0.1,
        v_ax=0.45,
        std=0.15,
        lambda_eph=100000.0,
    )
    corr_curve = fd_an.correlation_curve()
    lyap_corr_curve = corr_curve



    # Lyapunov stability analysis via class-based analyzer
    stab = StabilityAnalyzer(ds, StabilityConfig(), dataset_perm=perm_datasets)
    all_stability_data = stab.prepare(plot_heatmap=False)
    stability_metrics, bins = stab.bin_and_analyze(all_stability_data, max_distance=3500, max_pairs_per_bin=300)
    stability_metrics_perm_list: list = []
    for perm_ds in perm_datasets:
        all_stability_data_perm = stab.prepare(plot_heatmap=False, dataset=perm_ds)
        metrics_perm, _ = stab.bin_and_analyze(
            all_stability_data_perm,
            max_pairs_per_bin=200,
            bin_edges=bins,
        )
        stability_metrics_perm_list.append(metrics_perm)
    StabilityAnalyzer.plot_stability_distributions(
        stability_metrics,
        bins,
        stability_metrics_perm=stability_metrics_perm_list,
    )
    StabilityAnalyzer.plot_ci_with_significance(
        stability_metrics,
        bins,
        stability_metrics_perm=stability_metrics_perm_list,
        correlation_curve=lyap_corr_curve,
    )
    plt.show()

    # Co-firing vs distance with synergy overlay
    cof_res = fd_an.cofiring_avg_vs_distance(plusminus_ms=2.0, log=False)
    fd_an.plot_cofiring_with_synergy(cof_res)
    # Firing rate vs distance with synergy overlay
    fr_res = fd_an.avg_rate_vs_distance(log=False)
    fd_an.plot_rate_with_synergy(fr_res)
    # Pairwise active-electrode distance histogram with synergy shading
    dist_vals, dist_w = fd_an.distance_histogram(finite_size_correction=True)
    fd_an.plot_distance_hist_with_synergy(dist_vals, dist_w)

    plt.show()

    # IFR aggregate histogram + per-recording IFR time series via analyzer
    from ephax import IFRAnalyzer, IFRConfig
    ifr_cfg = IFRConfig(
        log_scale=True,
        hist_bins=100,
        overlay_gmm=True,
        show_kde=False,
        show_peaks=False,
        ts_bins=50,
        time_grid_hz=1000.0,
        max_time_points=50000,
    )

    ifr_an = IFRAnalyzer.from_dataset(ds, config=ifr_cfg, selection_prep_config=cfg_refs)
    ifr_an.plot_histogram(show=True)
    #ifr_an.plot_timeseries()  # uses refs from selection_prep_config

    # Temporal co-firing analyses (averaged heatmap + GIFs)
    from ephax import CofiringTemporalAnalyzer, CofiringTemporalConfig
    cfcfg = CofiringTemporalConfig(start_ms=-20, stop_ms=20, step_ms=10, normalize=False)
    # Pass cfg_refs so the analyzer uses the intended reference set (if provided)
    cfa = CofiringTemporalAnalyzer(ds, cfcfg, selection_prep_config=cfg_refs)
    #cfa.plot_avg_cofiring_heatmap() # takes about 15 minutes to run on all files
    #cfa.create_theta_gif(output_filename='cofiring_theta_all.gif')
    cfa.create_grid_gif(output_filename='cofiring_grid_all.gif')

    # DCT analysis on interpolated grids
    from ephax import DCTAnalyzer
    dct_an = DCTAnalyzer(ds)
    interp_grids = dct_an.compute_interpolated_grids(grid_size=50.0)
    # Average normalized DCT across recordings
    avg_dct = DCTAnalyzer.average_dct([g.grid for g in interp_grids])
    # Reconstruct from top components and visualize
    recon = DCTAnalyzer.reconstruct_from_top_components(avg_dct, stop_rank=3, start_rank=0, plot_distribution=True)
    DCTAnalyzer.plot_reconstructed_grid(reconstructed=recon, res=interp_grids[0])
    # Extract spatial frequencies along x
    DCTAnalyzer.extract_and_plot_spatial_frequencies_from_dct(avg_dct, n_components_stop=3, n_components_start=0, array_dims=(3800, 2100), axis='x')
    
    
