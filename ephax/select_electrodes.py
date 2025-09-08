import matplotlib.pyplot as plt
import h5py
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# Set seed for reproducibility
np.random.seed(46)

# Define the electrode array coordinates
nr_electrodes_x = np.tile(np.arange(220), 120)
nr_electrodes_y = np.repeat(np.arange(120), 220)

layout = {
    'x': 17.5 * nr_electrodes_x[:],
    'y': 17.5 * nr_electrodes_y[:],
    'x_index': nr_electrodes_x[:],
    'y_index': nr_electrodes_y[:],
    'electrode': nr_electrodes_x[:] + 220 * nr_electrodes_y[:]
}

# Returns numpy array of 1020 electrodes randomly chosen for stimulation
# with probabilities proportional to distance from stimulation electrode
def assign_electrodes(layout, stimulation_electrode):
    
    stim_electrode_idx = stimulation_electrode
    stim_x = layout['x'][stim_electrode_idx]
    stim_y = layout['y'][stim_electrode_idx]

    distances = np.sqrt((layout['x'] - stim_x)**2 + (layout['y'] - stim_y)**2)

    epsilon = 1e-10 # Avoid div by 0
    probabilities = 1 / (distances + epsilon)
    # probabilities = np.exp(-distances)

    num_electrodes_to_assign = 1020 
    probabilities /= probabilities.sum()
    probabilities *= num_electrodes_to_assign

    assigned_electrodes = np.random.choice(layout['electrode'], size=num_electrodes_to_assign, p=probabilities / probabilities.sum(), replace=False)

    # print("Assigned electrodes:", assigned_electrodes)
    print("Length assigned electrodes:", len(assigned_electrodes))

    return assigned_electrodes


def assign_n_proximate_electrodes(layout, stimulation_electrode, n):
    # Get the index of the stimulation electrode in the layout
    stim_electrode_idx = np.where(layout['electrode'] == stimulation_electrode)[0][0]
    stim_x = layout['x'][stim_electrode_idx]
    stim_y = layout['y'][stim_electrode_idx]

    # Calculate the distances from the stimulation electrode to all other electrodes
    distances = np.sqrt((layout['x'] - stim_x)**2 + (layout['y'] - stim_y)**2)

    # Get the indices of the n closest electrodes, excluding the stimulation electrode itself
    closest_indices = np.argsort(distances)[1:n+1]

    # Combine the stimulation electrode with the closest electrodes
    assigned_electrodes = [stimulation_electrode] + layout['electrode'][closest_indices].tolist()

    return assigned_electrodes


def assign_electrodes_by_circles(layout, stimulation_electrode, max_electrodes=1024):
    stim_x = layout['x'][stimulation_electrode]
    stim_y = layout['y'][stimulation_electrode]
    
    radial_step = 2.366
    max_radius = 3850
    min_x = 0
    max_x = 3850
    min_y = 0
    max_y = 2100
    angle_shift = np.pi / 2
    
    assigned_electrodes = []
    current_radius = radial_step
    circle_number = 1
    
    while current_radius <= max_radius and len(assigned_electrodes) < max_electrodes:
        num_electrodes = 2**((circle_number - 1).bit_length())
        
        angles = np.linspace(0, 2 * np.pi, num_electrodes, endpoint=False) + angle_shift
        if circle_number%2 != 0: 
            angles += angle_shift
        ideal_coords = [(stim_x + current_radius * np.cos(angle), stim_y + current_radius * np.sin(angle)) for angle in angles]
        
        for x, y in ideal_coords:
            if min_x <= x <= max_x and min_y <= y <= max_y:
                closest_electrode_id, closest_x, closest_y = find_closest_electrode(layout, x, y)
                if closest_electrode_id not in assigned_electrodes:
                    assigned_electrodes.append(closest_electrode_id)
                    if len(assigned_electrodes) >= max_electrodes:
                        break
        
        if circle_number > 1 and (circle_number & (circle_number - 1)) == 0:  # Check if circle_number is a power of 2
            radial_step *= 2  # Double the radial step when the number of points doubles
        

        current_radius += radial_step
        circle_number += 1
        angle_shift = np.pi / num_electrodes if num_electrodes > 1 else 0
    
    return assigned_electrodes

def assign_electrodes_fibonacci(layout, stimulation_electrode, max_electrodes=1024):
    stim_x = layout['x'][stimulation_electrode]
    stim_y = layout['y'][stimulation_electrode]
    
    min_x = 0
    max_x = 3850
    min_y = 0
    max_y = 2100
    
    assigned_electrodes = []
    golden_angle = np.pi * (3 - np.sqrt(5))  # Golden angle in radians
    
    ideal_coords = []
    for n in range(1, max_electrodes * 5):  # Generate more points to ensure we have enough valid ones
        radius = np.sqrt(n) * 50
        angle = n * golden_angle
        
        x = stim_x + radius * np.cos(angle)
        y = stim_y + radius * np.sin(angle)
        
        if min_x <= x <= max_x and min_y <= y <= max_y:
            ideal_coords.append((x, y))
            if len(ideal_coords) >= max_electrodes:
                break
    
    for x, y in ideal_coords:
        closest_electrode_id, closest_x, closest_y = find_closest_electrode(layout, x, y)
        if closest_electrode_id not in assigned_electrodes:
            assigned_electrodes.append(closest_electrode_id)
        if len(assigned_electrodes) >= max_electrodes:
            break
    
    return assigned_electrodes

def assign_electrodes_fibonacci_circles(layout, stimulation_electrode, max_electrodes=1024):
    stim_x = layout['x'][stimulation_electrode]
    stim_y = layout['y'][stimulation_electrode]
    
    min_x = 0
    max_x = 3850
    min_y = 0
    max_y = 2100
    
    assigned_electrodes = []
    golden_angle = np.pi * (3 - np.sqrt(5))  # Golden angle in radians
    
    ideal_coords = []
    for n in range(1, 100):  # Generate more points to ensure we have enough valid ones
        radius = 40 * n  # Increase the radius linearly with circle number
        angle = n * golden_angle
        
        num_electrodes_on_circle = 2**((n - 1).bit_length())
        angles = np.linspace(0, 2 * np.pi, num_electrodes_on_circle, endpoint=False) + angle
        
        for a in angles:
            x = stim_x + radius * np.cos(a)
            y = stim_y + radius * np.sin(a)
            
            if min_x <= x <= max_x and min_y <= y <= max_y:
                ideal_coords.append((x, y))
                if len(ideal_coords) >= max_electrodes:
                    break
        if len(ideal_coords) >= max_electrodes:
            break
    
    for x, y in ideal_coords:
        closest_electrode_id, closest_x, closest_y = find_closest_electrode(layout, x, y)
        if closest_electrode_id not in assigned_electrodes:
            assigned_electrodes.append(closest_electrode_id)
        if len(assigned_electrodes) >= max_electrodes:
            break
    
    return assigned_electrodes

# Layout with electrode id labels to see where which electrode is
def plot_layout(layout):
    plt.figure(figsize=(20, 12))
    plt.scatter(layout['x'], layout['y'], s=5, c='b', marker='o', alpha=0.5)
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.title('Spatial Layout of Well')
    plt.grid(True)
    plt.show()

# Plots layout of the electrodes chosen for measurement as well as stimulation electrode
def plot_assigned_layout(assignment, layout=layout, stimulation_electrode=None):

    electrode_indices = np.isin(layout['electrode'], assignment)

    plt.figure(figsize=(10, 6))
    plt.scatter(layout['x'][electrode_indices], layout['y'][electrode_indices], s=10, c='b', marker='o', alpha=0.5)
    if stimulation_electrode != None:
        index = np.where(layout['electrode'] == stimulation_electrode)
        plt.scatter(layout['x'][index], layout['y'][index], s=100, c='none', edgecolors='red', label='Stimulated Electrode')
    plt.legend(loc='upper right')
    plt.xlabel('X-coordinate ($\mu m$)')
    plt.ylabel('Y-coordinate ($\mu m$)')
    plt.title('Spatial Layout of Top 100 Most Active Recording Sites')
    plt.gca().set_aspect('equal')
    plt.grid(True)
    plt.show()

def get_electrode_id_at_coordinates(layout, x_coord, y_coord):
    # Find the index of the electrode at the specified coordinates
    index = np.where((layout['x'] == x_coord) & (layout['y'] == y_coord))[0]
    if index.size > 0:
        electrode_id = layout['electrode'][index[0]]
        return electrode_id
    else:
        return None

def find_closest_electrode(layout, x_coord, y_coord):
    # Calculate the Euclidean distance from the given coordinates to all electrodes
    distances = np.sqrt((layout['x'] - x_coord)**2 + (layout['y'] - y_coord)**2)
    
    # Find the index of the minimum distance
    min_index = np.argmin(distances)
    
    # Get the electrode id at the minimum distance
    closest_electrode_id = layout['electrode'][min_index]
    
    # Get the coordinates of the closest electrode
    closest_x = layout['x'][min_index]
    closest_y = layout['y'][min_index]
    
    return closest_electrode_id, closest_x, closest_y



if __name__ == "__main__":

    x_coord = 2905
    y_coord = 1136

    # Find closest electrode
    closest_electrode_id, closest_x, closest_y = find_closest_electrode(layout, x_coord, y_coord)
    print(f"Closest electrode ID to ({closest_x}, {closest_y}) at coordinates ({x_coord}, {y_coord}): {closest_electrode_id}")

    # Get the electrode ID
    #electrode_id = get_electrode_id_at_coordinates(layout, x_coord, y_coord)
    #print(f"Electrode ID at coordinates ({x_coord}, {y_coord}): {electrode_id}")

    stimulation_electrode = 11284
    #assignment = assign_electrodes_by_circles(layout, stimulation_electrode)
    #assignment = assign_n_proximate_electrodes(layout, stimulation_electrode, 20)
    #plot_layout(layout)
    #plot_assigned_layout(layout, assignment, stimulation_electrode)
    #print(f"Assigned {len(assignment)} electrodes in total)")
    #np.save('caspfiguration_center.npy', assignment)
    #np.savetxt('caspfiguration_10138.txt', assignment, fmt='%d')