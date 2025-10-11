import datetime
import math
import os
import pickle
import random
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from geopy.distance import geodesic

# Define perturbation parameters
DISTANCE_LEVEL = 3  # Position perturbation level (grid units)
GRID_SIZE = 0.0008  # Lat/lon grid unit approximately equals 200 meters
OUTLIER_RATIO = 0.1  # Ratio of trajectories that need outliers added
POINT_PROB = 0.1  # Ratio of points to add outliers in each trajectory (α = 0.1)

def perturb_geo_segment(points, level):
    """Apply continuous perturbation to lat/lon point sequence, maintaining consistent perturbation direction,
    but prioritizing directions perpendicular to the original trajectory direction"""
    if len(points) < 2:
        # If too few points, cannot determine direction, use random perturbation
        directions = [
            (GRID_SIZE, 0), (0, GRID_SIZE), (-GRID_SIZE, 0), (0, -GRID_SIZE),
            (GRID_SIZE, GRID_SIZE), (-GRID_SIZE, -GRID_SIZE), (-GRID_SIZE, GRID_SIZE), (GRID_SIZE, -GRID_SIZE)
        ]
        lon_offset, lat_offset = random.choice(directions)
    else:
        # Calculate main direction of trajectory segment
        start_lon, start_lat = points[0]
        end_lon, end_lat = points[-1]
        # Calculate direction difference
        lon_diff = end_lon - start_lon
        lat_diff = end_lat - start_lat
        # Determine main direction of trajectory segment: north-south or east-west
        is_north_south = abs(lat_diff) > abs(lon_diff)
        # Choose perpendicular perturbation direction based on main direction
        if is_north_south:
            # If trajectory is mainly north-south, choose east-west perturbation
            directions = [(GRID_SIZE, 0), (-GRID_SIZE, 0)]
            # Add some slight diagonal directions, but still mainly east-west
            directions += [(GRID_SIZE, GRID_SIZE * 0.3), (GRID_SIZE, -GRID_SIZE * 0.3),
                           (-GRID_SIZE, GRID_SIZE * 0.3), (-GRID_SIZE, -GRID_SIZE * 0.3)]
        else:
            # If trajectory is mainly east-west, choose north-south perturbation
            directions = [(0, GRID_SIZE), (0, -GRID_SIZE)]
            # Add some slight diagonal directions, but still mainly north-south
            directions += [(GRID_SIZE * 0.3, GRID_SIZE), (-GRID_SIZE * 0.3, GRID_SIZE),
                           (GRID_SIZE * 0.3, -GRID_SIZE), (-GRID_SIZE * 0.3, -GRID_SIZE)]
        # Randomly select one from suitable directions
        lon_offset, lat_offset = random.choice(directions)
    # Apply offset to all points
    new_points = []
    for lon, lat in points:
        new_lon = lon + lon_offset * level
        new_lat = lat + lat_offset * level
        new_points.append((new_lon, new_lat))
    return new_points

def calculate_path_distance(points):
    """Calculate total path distance (meters)"""
    total_distance = 0
    for i in range(1, len(points)):
        total_distance += geodesic(points[i - 1][::-1], points[i][::-1]).meters
    return total_distance

def perturb_time(timestamps, st_loc, end_loc, original_points, perturbed_points):
    """Adjust timestamps based on path length changes"""
    new_timestamps = timestamps.copy()
    # Calculate distances of original and perturbed paths
    original_segment = original_points[st_loc:end_loc]
    perturbed_segment = perturbed_points[st_loc:end_loc]
    original_distance = calculate_path_distance(original_segment)
    perturbed_distance = calculate_path_distance(perturbed_segment)
    # Assuming constant speed, calculate time offset factor
    if original_distance < 0.1:  # Avoid division by zero
        time_factor = 1.0
    else:
        time_factor = perturbed_distance / original_distance
    # Adjust time intervals within the anomalous segment
    for i in range(st_loc + 1, end_loc):
        original_interval = timestamps[i] - timestamps[i - 1]
        new_interval = original_interval * time_factor
        new_timestamps[i] = new_timestamps[i - 1] + new_interval
    # Adjust all timestamps after the anomalous segment
    if end_loc < len(timestamps):
        total_shift = new_timestamps[end_loc - 1] - timestamps[end_loc - 1]
        for i in range(end_loc, len(timestamps)):
            new_timestamps[i] = timestamps[i] + total_shift
    return new_timestamps

def generate_outliers(trajectory_list, ratio=OUTLIER_RATIO, level=DISTANCE_LEVEL, point_prob=POINT_PROB):
    """Generate outlier trajectories by applying consistent perturbation to continuous trajectory segments"""
    traj_num = len(trajectory_list)
    # Randomly select trajectory indices that need outliers added
    selected_idx = np.random.choice(traj_num, size=int(traj_num * ratio), replace=False)
    # New trajectory list
    new_trajectory_dict = {}
    for idx, trajectory in tqdm(enumerate(trajectory_list), total=traj_num, desc="Processing trajectories"):
        # Create deep copy of trajectory to avoid modifying original data
        new_traj = deepcopy(trajectory)
        traj_id = trajectory['devid']
        # If current trajectory index is in selected indices, add outliers
        if idx in selected_idx:
            geo_points = new_traj['o_geo']
            timestamps = new_traj['tms']
            # Determine length and starting position of trajectory segment to add outliers
            traj_len = len(geo_points)
            anomaly_len = max(2, int(traj_len * point_prob))
            anomaly_st_loc = np.random.randint(1, max(2, traj_len - anomaly_len - 1))
            anomaly_ed_loc = min(traj_len, anomaly_st_loc + anomaly_len)
            # Save original geo points for distance calculation
            original_points = geo_points.copy()
            # Apply same direction perturbation to continuous segment
            segment_to_perturb = geo_points[anomaly_st_loc:anomaly_ed_loc]
            perturbed_segment = perturb_geo_segment(segment_to_perturb, level)
            # Update trajectory
            for i, new_point in enumerate(perturbed_segment):
                geo_points[anomaly_st_loc + i] = new_point
            # Adjust timestamps to reflect path changes
            new_traj['tmp'] = perturb_time(
                timestamps,
                anomaly_st_loc,
                anomaly_ed_loc,
                original_points,
                geo_points
            )
            # Remove potentially inconsistent road segment ID information
            if 'opath' in new_traj:
                del new_traj['opath']
            new_trajectory_dict[traj_id] = new_traj
    return new_trajectory_dict, selected_idx

def main():
    # Load your trajectory data
    with open('../../chengdu_demo.pkl', 'rb') as f:
        trajectory_list = pickle.load(f)
    # Get all filenames in the folder
    # Truncate o_geo list in each trajectory, keeping only first 100 elements
    for trajectory in trajectory_list:
        if len(trajectory['o_geo']) > 150:
            trajectory['o_geo'] = trajectory['o_geo'][:150]
            trajectory['opath'] = trajectory['opath'][:150]
            trajectory['tmp'] = trajectory['tms'][:150]
    # Set random seed for reproducibility
    np.random.seed(1234)
    random.seed(1234)
    print(f"Parameter settings: Perturbation level d = {DISTANCE_LEVEL}, Outlier trajectory ratio ρ = {OUTLIER_RATIO}, Outlier trajectory segment ratio α = {POINT_PROB}")
    print(f"Grid unit set to approximately 200 meters (lat/lon increment: {GRID_SIZE})")
    # Filter trajectories with length >= 150
    long_trajectories = [traj for traj in trajectory_list if len(traj['o_geo']) >= 150]
    print(f"Found {len(long_trajectories)} trajectories with length >= 150")
    # Generate outlier trajectories
    print("Generating outlier trajectories...")
    outlier_trajs, outlier_idx = generate_outliers(
        long_trajectories[0:200],
        ratio=OUTLIER_RATIO,
        level=DISTANCE_LEVEL,
        point_prob=POINT_PROB
    )
    # Save results
    with open('outlier_trajectories.pkl', 'wb') as f:
        pickle.dump(outlier_trajs, f)
    with open('outlier_indices.pkl', 'wb') as f:
        pickle.dump(outlier_idx, f)
    print(f"Outlier trajectories generated! Total processed {len(trajectory_list)} trajectories, "
          f"of which {len(outlier_idx)} had outliers added.")
    print(f"Results saved to 'outlier_trajectories.pkl' and 'outlier_indices.pkl'")

if __name__ == '__main__':
    main()