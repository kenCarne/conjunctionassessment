import os
import numpy as np
from sgp4.api import Satrec, jday
from scipy.stats import multivariate_normal
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count
import csv
import plotly.graph_objs as go
import plotly.express as px
import plotly.colors as pc

# Constants
EARTH_RADIUS = 6371  # Earth's radius in kilometers
COLLISION_PROB_THRESHOLD = 1e-6  # Threshold for plotting conjunction satellites

def propagate_tle(tle_line1, tle_line2, time):
    satellite = Satrec.twoline2rv(tle_line1, tle_line2)
    jd, fr = jday(time.year, time.month, time.day, time.hour, time.minute, time.second)
    e, r, v = satellite.sgp4(jd, fr)
    if e != 0:
        # Log the problematic TLE
        print(f"Error in propagating TLE: {tle_line1}, {tle_line2} at time {time}")
        return None, None  # Return None to indicate failure
    return np.array(r), np.array(v)

def calculate_collision_probability(r1, r2, cov1, cov2):
    r1 = np.asarray(r1, dtype=np.float64)
    r2 = np.asarray(r2, dtype=np.float64)
    cov1 = np.asarray(cov1, dtype=np.float64)
    cov2 = np.asarray(cov2, dtype=np.float64)
    
    relative_position = r1 - r2
    relative_covariance = cov1 + cov2
    distribution = multivariate_normal(mean=[0, 0, 0], cov=relative_covariance)
    probability_of_collision = distribution.pdf(relative_position)
    return probability_of_collision

def find_closest_approach_time(tle1, tle2, start_time, end_time, time_step_seconds=60):
    min_distance = float('inf')
    closest_time = start_time
    best_r1 = best_r2 = None
    
    current_time = start_time
    while current_time <= end_time:
        r1, _ = propagate_tle(tle1[0], tle1[1], current_time)
        r2, _ = propagate_tle(tle2[0], tle2[1], current_time)
        
        if r1 is None or r2 is None:
            # Skip calculation if propagation failed
            return None, None, None
        
        distance = np.linalg.norm(r1 - r2)
        
        if distance < min_distance:
            min_distance = distance
            closest_time = current_time
            best_r1, best_r2 = r1, r2
        
        current_time += timedelta(seconds=time_step_seconds)
    
    return closest_time, best_r1, best_r2

def calculate_collision_probability_for_pair(args):
    tle1, tle2, start_time, end_time, sat_num_i, sat_num_j = args
    collision_epoch, r1, r2 = find_closest_approach_time(tle1, tle2, start_time, end_time)
    
    if r1 is None or r2 is None:
        return sat_num_i, sat_num_j, 0, collision_epoch, [0, 0, 0]  # Return 0 probability if propagation failed
    
    cov1 = np.diag([1e-6, 1e-6, 1e-6])
    cov2 = np.diag([1e-6, 1e-6, 1e-6])
    prob = calculate_collision_probability(r1, r2, cov1, cov2)
    collision_point = (r1 + r2) / 2  # Midpoint as the likely collision point
    return sat_num_i, sat_num_j, prob, collision_epoch, collision_point

def calculate_batch_collision_probabilities(tle_list, satellite_numbers, start_time, end_time, satlist=None):
    if satlist:
        filtered_tle_list, filtered_satellite_numbers = filter_tle_list(tle_list, satellite_numbers, satlist)
    else:
        filtered_tle_list, filtered_satellite_numbers = tle_list, satellite_numbers

    args_list = []
    for i, tle1 in enumerate(filtered_tle_list):
        for j, tle2 in enumerate(tle_list):
            args_list.append((tle1, tle2, start_time, end_time, filtered_satellite_numbers[i], satellite_numbers[j]))

    with Pool(processes=cpu_count()) as pool:
        collision_probabilities = pool.map(calculate_collision_probability_for_pair, args_list)
    
    return collision_probabilities

def read_tle_file(file_path):
    tle_list = []
    satellite_numbers = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for i in range(0, len(lines), 3):
            if len(lines[i+1].strip()) < 2 or len(lines[i+2].strip()) < 2:
                continue  # Skip any empty or invalid lines
            try:
                tle_line1 = lines[i+1].strip()
                tle_line2 = lines[i+2].strip()
                
                # Extract satellite number from the second line
                satellite_number = tle_line2.split()[1][:5]  # First 5 characters after the '1' in the second line
                
                tle_list.append((tle_line1, tle_line2))
                satellite_numbers.append(satellite_number)
            except IndexError:
                print(f"Skipping malformed TLE at lines {i+1} to {i+3}")
                continue
    return tle_list, satellite_numbers

def filter_tle_list(tle_list, satellite_numbers, satlist):
    filtered_tle_list = []
    filtered_satellite_numbers = []

    for i, sat_number in enumerate(satellite_numbers):
        if sat_number in satlist:
            filtered_tle_list.append(tle_list[i])
            filtered_satellite_numbers.append(sat_number)

    return filtered_tle_list, filtered_satellite_numbers

def write_results_to_file(collision_probabilities, filename):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['Parent_Satellite', 'Tested_Satellite', 'Collision_Probability', 'Collision_Epoch']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for sat_num_i, sat_num_j, prob, epoch, _ in collision_probabilities:
            writer.writerow({
                'Parent_Satellite': sat_num_i,
                'Tested_Satellite': sat_num_j,
                'Collision_Probability': f"{prob:.6e}",
                'Collision_Epoch': epoch.strftime("%Y-%m-%d %H:%M:%S") if epoch else "N/A"
            })

def create_earth():
    # Create a sphere representing Earth
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = EARTH_RADIUS * np.outer(np.cos(u), np.sin(v))
    y = EARTH_RADIUS * np.outer(np.sin(u), np.sin(v))
    z = EARTH_RADIUS * np.outer(np.ones(np.size(u)), np.cos(v))

    return x, y, z

def visualize_collision_hotspots(collision_probabilities, tle_list, satellite_numbers, start_time, end_time, satlist):
    # Create the Earth
    x, y, z = create_earth()
    colors = pc.qualitative.Alphabet

    # Create a 3D figure
    fig = go.Figure()

    # Add the Earth to the plot
    fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale='Blues', opacity=0.5, showscale=False))

    # Plot satellites from satlist and any conjunction satellites with a probability > 1e-6
    plotted_satellites = set()
    collision_points = []

    for prob_entry in collision_probabilities:
        parent_sat, tested_sat, prob, epoch, pos = prob_entry

        if parent_sat in satlist and parent_sat not in plotted_satellites:
            # Plot the satlist satellite
            plot_orbit(fig, tle_list, satellite_numbers, parent_sat, start_time, end_time, colors)
            plotted_satellites.add(parent_sat)
        
        if prob > COLLISION_PROB_THRESHOLD:
            collision_points.append(pos)
            if tested_sat not in plotted_satellites:
                # Plot the conjunction satellite
                plot_orbit(fig, tle_list, satellite_numbers, tested_sat, start_time, end_time, colors)
                plotted_satellites.add(tested_sat)

    # Add heatmap of collision points
    if collision_points:
        collision_points = np.array(collision_points)
        fig.add_trace(go.Scatter3d(
            x=collision_points[:, 0], y=collision_points[:, 1], z=collision_points[:, 2],
            mode='markers',
            marker=dict(size=5, color='red', opacity=0.6),
            name='Collision Hotspots'
        ))

    # Set aspect ratio to ensure Earth is spherical
    fig.update_layout(scene=dict(
        xaxis=dict(nticks=4, range=[-EARTH_RADIUS*1.5, EARTH_RADIUS*1.5], title='X (km)'),
        yaxis=dict(nticks=4, range=[-EARTH_RADIUS*1.5, EARTH_RADIUS*1.5], title='Y (km)'),
        zaxis=dict(nticks=4, range=[-EARTH_RADIUS*1.5, EARTH_RADIUS*1.5], title='Z (km)'),
        aspectmode='data'  # Ensures equal scaling for x, y, z axes
    ))

    # Move the legend to the left side
    fig.update_layout(legend=dict(
        x=0,
        y=0.5,
        traceorder="normal",
        font=dict(
            family="sans-serif",
            size=12,
            color="black"
        ),
        bgcolor="LightSteelBlue",
        bordercolor="Black",
        borderwidth=2
    ))

    fig.show()

def plot_orbit(fig, tle_list, satellite_numbers, sat_num, start_time, end_time, colors):
    idx = satellite_numbers.index(sat_num)
    tle = tle_list[idx]
    orbital_positions = []
    current_time = start_time

    while current_time <= end_time:
        r, _ = propagate_tle(tle[0], tle[1], current_time)
        if r is None:
            continue  # Skip this satellite if propagation failed
        orbital_positions.append(r)
        current_time += timedelta(minutes=10)  # Sample the orbit every 10 minutes

    if orbital_positions:
        orbital_positions = np.array(orbital_positions)
        fig.add_trace(go.Scatter3d(
            x=orbital_positions[:, 0], y=orbital_positions[:, 1], z=orbital_positions[:, 2],
            mode='lines',
            line=dict(color=colors[idx % len(colors)], width=2),
            name=f'Orbit of {sat_num}'
        ))

if __name__ == "__main__":
    root = 'C:\\Users\\extre\\Documents\\GitHub\\6_dof_sim\\'
    tle_file_path = os.path.join(root,"tle_data2.txt")
    start_time = datetime.utcnow()
    end_time = start_time + timedelta(days=1)
    print('Runnning...')
    tle_list, satellite_numbers = read_tle_file(tle_file_path)
    satlist = ["02872"] 
    #satlist = ["43812","44499","47971","49469","49470","49772","49773","49949","49950","52196","52197","55982", "55983"]
    collision_probabilities = calculate_batch_collision_probabilities(tle_list, satellite_numbers, start_time, end_time, satlist=satlist)


    output_filename = os.path.join(root,"collision_probabilities.csv")
    write_results_to_file(collision_probabilities, output_filename)
    print(f"Results written to {output_filename}")
    visualize_collision_hotspots(collision_probabilities, tle_list, satellite_numbers, start_time, end_time, satlist)

    print('Complete')
    """   
    Implement more robust error logging to capture and report any issues
    """
