import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize, LogNorm, LinearSegmentedColormap
import matplotlib.ticker as ticker




# Define the exponential decay model for propagation speed
def v(x, v0=0.9, v_min=0.4, k=0.002):
    return v_min + (v0 - v_min) * np.exp(-k * x)

def plot_v_decay(max_um, v0=0.9, v_min=0.4, k=0.002):

    # Define the distance range (x-axis) in micrometers
    x = np.linspace(0, max_um, int(max_um/10))  # Distance in micrometers

    # Calculate propagation speed over the distance range
    speed = v(x, v0, v_min, k)

    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.plot(x, speed, label=f"v(x) = {v_min} + ({v0} - {v_min}) * exp(-{k} * x)")
    plt.xlabel("Distance (micrometers)")
    plt.ylabel("Propagation Speed (m/s)")
    plt.title("Exponential Decay of Propagation Speed in Unmyelinated Axon")
    plt.grid(True)
    plt.legend()
    plt.show()

def correlation_function(r_um, hz, v_eph, v_ax, lambda_eph):
    """
    Compute the correlation values over a range of distances.

    Parameters:
    - r_um: numpy array of distances in micrometers.
    - hz_list: list of frequencies in Hz.
    - v_eph: ephaptic velocity in micrometers per second.
    - v_ax: axonal velocity in micrometers per second.
    - lambda_eph: attenuation length in micrometers.

    Returns:
    - R_total: 1D numpy array of correlation values.
    """
    # Compute Delta_t in seconds
    Delta_t = r_um * (1 / v_eph - 1 / v_ax)  # Units: μm * (s/μm) = s

    # Initialize summed correlations
    R_total = np.zeros_like(r_um)

    R_total = (np.cos(2 * np.pi * hz * Delta_t) * np.exp((-r_um / lambda_eph))**2)/hz

    return R_total

### for r in r_um:
###        # Compute Delta_t in seconds
###        delta = r*(v_min + v_eph)/(v_min*v_eph) - 1/(k*v_min)*np.log((1+beta)/(1+beta*np.exp(-k*r)))
###        Delta_t = np.append(Delta_t, delta)

def correlation_function_w_sum(r_um, hz_list, v_eph, v_ax, lambda_eph):
    """
    Compute the correlation values over a range of distances.

    Parameters:
    - r_um: numpy array of distances in micrometers.
    - hz_list: list of frequencies in Hz.
    - v_eph: ephaptic velocity in micrometers per second.
    - v_ax: axonal velocity in micrometers per second.
    - lambda_eph: attenuation length in micrometers.

    Returns:
    - R_total: 1D numpy array of correlation values.
    """
    Delta_t = r_um * (1 / v_eph - 1 / v_ax)
    print(v(r_um))  # Units: μm * (s/μm) = s
    R_total = np.zeros_like(r_um)

    # Sum the cosine waves for each frequency
    for hz in hz_list:
        R_total += (np.cos(2 * np.pi * hz * Delta_t) * np.exp((-r_um / lambda_eph))**2)/hz

    return R_total


def get_summed_hz_distribution(f_list, sigma_list):
    summed_distribution = np.array([])

    for f, sigma in zip(f_list, sigma_list):
        mean_log = np.log(f)
        lognormal = np.random.lognormal(mean_log, sigma, 10000)
        summed_distribution = np.append(summed_distribution, lognormal, axis=0)
    
    return summed_distribution

def plot_sum_lognormal(f_list, v_eph, v_ax, max_um, sigma):
    # Define x-axis and log-transform frequency
    r_um = np.linspace(0, max_um, int(max_um/10))

    summed_distribution = get_summed_hz_distribution(f_list, sigma)
    
    summed_r = correlation_function_w_sum(r_um, summed_distribution, v_eph, v_ax, 100000)    

    # Construct final plot
    plt.figure(figsize=(8, 4))
    plt.plot(r_um, summed_r)
    plt.title(f'Cross-Correlation Function Summed Across {str(f_list)[1:-1]} Hz vs. Distance')
    plt.xlabel('Distance $r$ ($\mu$m)')
    plt.ylabel('Cross-Correlation')
    plt.grid(True)
    plt.legend()
    plt.show()    


import numpy as np
import matplotlib.pyplot as plt

def plot_r_sigma(f, v_eph, v_ax, max_um, sigma_list):
    # Define x-axis and log-transform frequency
    r_um = np.linspace(0, max_um, int(max_um/10))
    mean_log = np.log(f)
    print("Mean of log-normal distributions (mean_log):", mean_log)
    
    num_sigmas = len(sigma_list)
    
    # Determine if sigma is close to zero
    sigma_threshold = 0.01 
    include_histogram = any(sigma > sigma_threshold for sigma in sigma_list)
    
    if include_histogram:
        # Create figure with subplots: 1 row, 2 columns
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={'width_ratios': [1, 2]})
        ax_hist = axes[0]
        ax_corr = axes[1]
    else:
        # Only create a single plot for cross-correlation
        fig, ax_corr = plt.subplots(figsize=(8, 4))
        ax_hist = None  # No histogram axis

    # Define colors for consistency
    colors = plt.cm.viridis(np.linspace(0, 1, num_sigmas))
    
    # Histogram of log-normal distributions (only if sigma not close to zero)
    if include_histogram:
        # Define bins for histogram (log scale)
        min_value = f * np.exp(-3 * max(sigma_list))
        max_value = f * np.exp(3 * max(sigma_list))
        bins = np.logspace(np.log10(min_value), np.log10(max_value), 100)
        
        for idx, sigma in enumerate(sigma_list):
            if sigma > sigma_threshold:
                lognormal = np.random.lognormal(mean_log, sigma, 100000)
                ax_hist.hist(lognormal, bins=bins, color=colors[idx], alpha=0.5, label=f'$\sigma$ = {sigma}')
        
        ax_hist.set_xscale('log')
        ax_hist.set_xlabel('Frequency (Hz)')
        ax_hist.set_ylabel('Count')
        ax_hist.set_title('Frequency Distributions Across $\sigma$ Values')
        ax_hist.legend()
        
        # Set x-axis ticks at 10, 20, 30, ..., placed on log scale
        max_tick = int(max_value // 10 * 10 + 10)  # Round up to the next multiple of 10
        x_ticks = np.arange(10, max_tick, 10)
        ax_hist.set_xticks(x_ticks)
        ax_hist.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax_hist.tick_params(axis='x', which='major', labelsize=8)
        plt.setp(ax_hist.get_xticklabels(), rotation=45, ha='right')
    
    # Cross-correlation functions
    for idx, sigma in enumerate(sigma_list):
        lognormal = np.random.lognormal(mean_log, sigma, 1000000)
        corr = correlation_function_w_sum(r_um, lognormal, v_eph, v_ax, 1000000)
        ax_corr.plot(r_um, corr, color=colors[idx], label=f'$\sigma$ = {sigma}')
    ax_corr.set_title(f'Cross-Correlation Function vs. Distance at {f} Hz')
    ax_corr.set_xlabel('Distance $r$ ($\mu$m)')
    ax_corr.set_ylabel('Cross-Correlation')
    ax_corr.grid(True)
    ax_corr.legend()
    
    plt.tight_layout()
    plt.show()

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """
    Truncate a colormap to use only a portion of its range.

    Parameters:
    - cmap: The original colormap to truncate.
    - minval: The lower bound of the colormap to use (between 0 and 1).
    - maxval: The upper bound of the colormap to use (between 0 and 1).
    - n: The number of points in the colormap.

    Returns:
    - new_cmap: The truncated colormap.
    """
    new_cmap = LinearSegmentedColormap.from_list(
        f'truncated({cmap.name},{minval:.2f},{maxval:.2f})',
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def animate_sum_of_waves(f_list, v_eph, v_ax, max_um, lambda_eph):
    """
    Animate the summing up of individual contributions from multiple frequencies into one superposed wave.

    Parameters:
    - f_list: list of frequencies in Hz.
    - v_eph: ephaptic velocity in micrometers per second.
    - v_ax: axonal velocity in micrometers per second.
    - max_um: maximum distance in micrometers.
    - lambda_eph: attenuation length in micrometers.
    """
    # Prepare data
    r_um = np.linspace(0, max_um, int(max_um/10))
    hz_list = f_list  # Frequencies are fixed

    # Compute individual contributions
    contributions = []
    for hz in hz_list:
        R = correlation_function(r_um, hz, v_eph, v_ax, lambda_eph)
        contributions.append(R)

    # === New Code Starts Here ===

    # Create a truncated colormap (e.g., top half of viridis)
    original_cmap = plt.cm.viridis
    cmap = truncate_colormap(original_cmap, 0.3, 0.95)

    # Normalize frequencies for color mapping
    norm = plt.Normalize(min(hz_list), max(hz_list))

    # Map each frequency to a color
    colors = [cmap(norm(hz)) for hz in hz_list]

    # Initialize the figure and axis
    fig, ax = plt.subplots(figsize=(8, 4))

    # Create individual lines with assigned colors
    lines_individual = []
    for hz, color in zip(hz_list, colors):
        line, = ax.plot([], [], label=f'{hz} Hz', color=color)
        lines_individual.append(line)

    # Create the total line (black color)
    line_total, = ax.plot([], [], label='Total', color='black')

    # === New Code Ends Here ===

    ax.set_xlim(r_um[0], r_um[-1])
    # Determine y-limits from the data
    y_min = np.min([np.min(c) for c in contributions])
    y_max = np.max([np.max(c) for c in contributions])
    ax.set_ylim(y_min * 1.5, y_max * 1.5)
    ax.set_xlabel('Distance $r$ ($\mu$m)')
    ax.set_ylabel('Signal Synergy Coefficient')
    ax.set_title('Signal Summation Example')
    ax.grid(True)
    ax.legend(loc='upper right')

    # Number of frames
    N_frames = 300  # Total number of frames

    # Determine frame divisions for stages
    N_stages = len(contributions)
    N_per_stage = N_frames // (N_stages + 1)

    # Animation function
    def animate(frame):
        # Initialize multipliers
        multipliers = [0] * len(contributions)

        for i in range(len(contributions)):
            start_frame = N_per_stage * i
            end_frame = N_per_stage * (i + 1)

            if frame < start_frame:
                multipliers[i] = 0
            elif frame < end_frame:
                # Fade in the current frequency
                multipliers[i] = (frame - start_frame) / (end_frame - start_frame)
            else:
                multipliers[i] = 1

        # Compute total contribution
        total_R = np.zeros_like(r_um)
        for i, (multiplier, contrib) in enumerate(zip(multipliers, contributions)):
            contrib_scaled = multiplier * contrib
            total_R += contrib_scaled
            # Update individual line
            lines_individual[i].set_data(r_um, contrib_scaled)

        # Update total line
        line_total.set_data(r_um, total_R)
        return [line_total] + lines_individual

    # Create the animation
    ani = FuncAnimation(fig, animate, frames=N_frames, interval=50, blit=True)

    # Save the animation
    ani.save('signal_summation.gif', writer='imagemagick', fps=30)
    fig.savefig('signal_summation.png')

    plt.show()

    


if __name__ == "__main__":
    # Parameters
    f1 = 52     # frequency in Hz
    f2 = 201
    f3 = 615    
    v_eph = 0.1*1e6     # Velocity v1 in m/s (variable)
    v_ax = 0.45*1e6      # Velocity v2 in m/s (variable)
    max_um = 3500       # Maximum micrometer plotted on x-axis

    sigma_list = [0., 0.1, 0.2, 0.3]
    plot_r_sigma(f1, v_eph, v_ax, max_um, sigma_list)
    #plot_r_sigma(f2, v_eph, v_ax, max_um, sigma_list)
    sigma = 0.2
    #plot_sum_lognormal([f1], v_eph, v_ax, max_um, sigma_list)
    # Animate the summing up of individual contributions
    animate_sum_of_waves([f1, f2, f3], v_eph, v_ax, max_um, 100000)

    #plot_v_decay(max_um)

