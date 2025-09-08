import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Function to calculate the signal at a given point (x, y) and time t
def signal(x, y, t, speed):
    distance = np.sqrt(x**2 + y**2)
    return t - distance / speed

def correlation_function(r, lambda_eph, f, v_eph):
    return np.exp(-(r / lambda_eph)**3) * np.cos(4 * np.pi * f * r / v_eph)

# Function to update the plot for each frame of the animation
def update(frame):
    ax1.clear()
    
    # Set limits to ensure the plot remains within a circular view
    ax1.set_ylim(0, 10)  # Set this to match the maximum radial distance
    ax1.set_aspect('auto')
    
    ephaptic_speed = 0.1
    axonal_speed = 0.5
    ms_delay = 20
    
    for angle in np.linspace(0, 2 * np.pi, 200):
        ex = frame * np.cos(angle) * ephaptic_speed
        ey = frame * np.sin(angle) * ephaptic_speed 
        r = np.sqrt(ex**2 + ey**2)
        theta = angle  # angle already in radians
        ax1.plot(theta, r, 'bo', markersize=5 * signal(ex, ey, frame, ephaptic_speed))
    
    axonal_decay_rate = 0.1
    for angle in np.linspace(0, 2 * np.pi, 200):
        ax = frame * np.cos(angle) * axonal_speed
        ay = frame * np.sin(angle) * axonal_speed
        r = np.sqrt(ax**2 + ay**2)
        theta = angle  # angle already in radians
        intensity_axonal = signal(ax, ay, frame, axonal_speed)
        ax1.plot(theta, r, 'ro', markersize=5 * intensity_axonal)
    
    if frame > ms_delay:
        for angle in np.linspace(0, 2 * np.pi, 200):
            ex1 = (frame-ms_delay) * np.cos(angle) * ephaptic_speed
            ey1 = (frame-ms_delay) * np.sin(angle) * ephaptic_speed
            r = np.sqrt(ex1**2 + ey1**2)
            theta = angle
            intensity_ephaptic = signal(ex1, ey1, frame, ephaptic_speed)
            ax1.plot(theta, r, 'bo', markersize=5 * intensity_ephaptic)
        
        for angle in np.linspace(0, 2 * np.pi, 200):
            ax2 = (frame-ms_delay) * np.cos(angle) * axonal_speed
            ay2 = (frame-ms_delay) * np.sin(angle) * axonal_speed
            r = np.sqrt(ax2**2 + ay2**2)
            theta = angle
            ax1.plot(theta, r, 'ro', markersize=5 * signal(ax2, ay2, frame, axonal_speed))

# Create the figure and axes
fig, ax1 = plt.subplots(figsize=(8,8), subplot_kw={'projection': 'polar'})

# Create the animation
animation = FuncAnimation(fig, update, frames=np.arange(0, 40, 0.1), interval=50)

# Save the animation as a GIF
#animation.save('animation.gif', writer='imagemagick', fps=30)

plt.legend()
plt.show()
