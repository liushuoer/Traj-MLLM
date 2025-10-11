import os
import pickle
import matplotlib.patches as patches
import geopandas
import csv

import geopandas as gpd
from shapely.errors import GEOSException
from shapely.wkt import loads
from sklearn.neighbors import BallTree
from tqdm import tqdm
from segmentation import *
from utils import *
from datetime import datetime, timedelta
import pickle
from collections import Counter
# Load fmm_traj data
import geopandas as gpd
from collections import Counter
import numpy as np
from tqdm import tqdm
from tools import *


# Read .geo file and build mapping relationship between geo_id and highway
def load_highway_mapping(geo_path):
    highway_dict = {}
    with open(geo_path, 'r') as f:
        # Use csv reader to handle fields with quotes
        reader = csv.reader(f)
        header = next(reader)  # Skip header row
        # Print header row to confirm maxspeed position
        print(f"File header: {header}")

        for row in reader:
            try:
                geo_id = int(row[0])  # geo_id
                highway = int(row[-1])  # highway field
                highway_dict[geo_id] = highway
            except Exception as e:
                print(f"Error parsing ID {row[0]}: {e}")
                continue

    print(f"Loaded highway information for {len(highway_dict)} roads")
    if highway_dict:
        print("Example of first road's highway mapping:")
        print(list(highway_dict.items())[0])

    return highway_dict


def load_maxspeed_mapping(geo_path):
    maxspeed_dict = {}
    with open(geo_path, 'r') as f:
        # Use csv reader to handle fields with quotes
        reader = csv.reader(f)
        header = next(reader)  # Skip header row

        for row in reader:
            try:
                geo_id = int(row[0])  # geo_id
                maxspeed = int(row[-1])  # maxspeed field (assumed to be last field)
                maxspeed_dict[geo_id] = maxspeed
            except Exception as e:
                print(f"Error parsing ID {row[0]}: {e}")
                continue

    print(f"Loaded maxspeed information for {len(maxspeed_dict)} roads")
    if maxspeed_dict:
        print("Example of first road's maxspeed mapping:")
        print(list(maxspeed_dict.items())[0])

    return maxspeed_dict


def print_direction_and_speed_trend(trajectory, trajectory_id):
    coords = trajectory.get('o_geo', [])
    if 'tmp' in trajectory:
        timestamps = trajectory.get('tmp', [])  # Assume trajectory has timestamps in seconds
    else:
        timestamps = trajectory.get('tms', [])  # Assume trajectory has timestamps in seconds

    # Calculate direction of last five points
    start = coords[-5]
    end = coords[-1]
    bearing = calculate_bearing(start[0], start[1], end[0], end[1])
    direction = bearing_to_direction(bearing)

    # Calculate speed sequence of last 10 points
    speeds = []
    for i in range(-5, -1):
        speed = calculate_speed(coords[i], coords[i+1], timestamps[i], timestamps[i+1])
        speeds.append(speed)

    # Calculate speed trend (simply use final speed minus initial speed for judgment)
    trend_value = speeds[-1] - speeds[0]
    if trend_value > 0.1:  # You can adjust this threshold
        speed_trend = "Speed up"
    elif trend_value < -0.1:
        speed_trend = "Slow down"
    else:
        speed_trend = "Steady speed"

    # Generate output text
    output_text = f"""
**Trajectory End-Point Information:**
* **Visuals:**
    * `last_trajectory_view` (A zoomed-in image of the last few points, marked with red dots, with the green line representing the trajectory)
* **Stats:**
    * Overall direction of the last five points: {direction} (Azimuth: {bearing:.2f}°)
    * Movement trend of the last ten points: {speed_trend}
    """

    return output_text

def convert_timestamp_to_datetime(timestamp):
    """Convert float timestamp to datetime object"""
    return datetime.fromtimestamp(timestamp)

def convert_timestamp_to_datetime_1(timestamp):
    """Convert float timestamp to datetime object"""
    return datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')



def calculate_segment_distance(segment):
    """
    Calculate total distance of trajectory segment (kilometers)

    Parameters:
    segment - trajectory segment

    Returns:
    total distance (kilometers)
    """
    if 'points' not in segment or len(segment['points']) < 2:
        return 0

    coords = segment['points']
    total_distance = 0

    for i in range(len(coords) - 1):
        lon1, lat1 = coords[i]
        lon2, lat2 = coords[i + 1]

        dist_m = haversine_distance(lat1, lon1, lat2, lon2)
        total_distance += dist_m

    return total_distance


# Get maximum speed of road segment based on maximum speed mapping
def calculate_segment_maxspeed(segment_road_ids, maxspeed_dict):
    """
    Get maximum speed value for a single trajectory segment

    Parameters:
    segment_road_ids - list of road IDs contained in this trajectory segment
    maxspeed_dict - mapping dictionary from geo_id to maxspeed

    Returns:
    maximum speed value in this segment, returns None if no valid value
    """
    # Get maximum speed for each road ID
    maxspeeds = []

    # Collect all valid maximum speeds
    for road_id in segment_road_ids:
        if road_id in maxspeed_dict and maxspeed_dict[road_id] > 0:
            maxspeeds.append(maxspeed_dict[road_id])

    # If no valid maximum speed information, return None
    if not maxspeeds:
        return None

    # Return maximum speed value
    return max(maxspeeds)


def generate_llm_prompt(current_trajectory_for_prompt, devid, segments, edges_dict=None, maxspeed_dict=None, results_dict=None):
    """
    Generate prompt for LLM navigation planning and arrival time prediction

    Parameters:
    devid - device ID, used to find detailed information for specific trajectory
    segments - list of trajectory segments
    """
    segment_info = []
    index_start = 0
    index_end = 0

    # Extract information for each trajectory segment
    for i, segment in enumerate(segments):
        segment_details = {"segment_number": i + 1}

        # Only save start time for first segment
        if i == 0 and 'times' in segment:
            segment_details["Start time"] = segment['times'][0]

        # Calculate trajectory segment movement distance
        segment_distance = calculate_segment_distance(segment)
        segment_details["Distance (m)"] = round(segment_distance, 4)

        # # Calculate trajectory segment average speed
        # segment_speed = calculate_segment_speed(segment)
        # segment_details["Average speed (m/s)"] = round(segment_speed, 4)

        # Get main road type
        # Get corresponding fmm_traj
        fmm_traj = current_trajectory_for_prompt['opath']
        # Calculate corresponding fmm_traj index range for this segment
        segment_road_ids = fmm_traj[index_start:index_end + 1]
        index_start = index_end + 1
        index_end += len(segment['points'])

        # # Get corresponding road types
        # highway_types = [edges_dict.get(road_id) for road_id in segment_road_ids if road_id in edges_dict]
        #
        # # Filter out None values (cases where road_id is not in edges_dict)
        # # and only keep non-"unknown" road types
        # known_highway_types = [h for h in highway_types if h is not None and h != "unknown"]
        #
        # if known_highway_types:
        #     highway_counter = Counter(known_highway_types)
        #     most_common = highway_counter.most_common(2)
        #
        #     if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
        #         segment_details["Main road types"] = [most_common[0][0], most_common[1][0]]
        #     else:
        #         segment_details["Main road types"] = most_common[0][0]
        # else:
        #     # Only set to "Unknown" if there are no known road types
        #     segment_details["Main road types"] = "Unknown"

        # Add the following code snippet in generate_llm_prompt function
        # Calculate maximum speed
        max_speed = calculate_segment_maxspeed(segment_road_ids, maxspeed_dict)
        # if max_speed is not None:
        #     segment_details["Max speed"] = max_speed
        # else:
        #     segment_details["Max speed"] = "Unknown"

        segment_info.append(segment_details)

    # Generate LLM navigation planning prompt
    prompt = generate_navigation_prompt(segment_info)

    prompt += print_direction_and_speed_trend(current_trajectory_for_prompt, devid)

    prompt += """
    ---

**Overall Trajectory Visuals:**
* **Visuals:**
    * `poi_full_image` (Comprehensive view of the entire trajectory with POIs and signalized intersections)
    * `road_structure_full_image` (Detailed illustration of the full trajectory's road class, lanes, and network structure)
    """
    return prompt


def calculate_segment_speed_new(segment):
    """
    Calculate speed information for a single trajectory segment

    Parameters:
    segment - trajectory segment

    Returns:
    dictionary containing speed information, including start time, end time, duration, total distance, average speed, maximum speed, minimum speed and maximum acceleration
    """
    from geopy import distance

    # Calculate distance between two points (in meters)
    def calculate_distance(lat1, lon1, lat2, lon2):
        return distance.distance((lat1, lon1), (lat2, lon2)).meters

    # Calculate speed (meters/second)
    def calculate_speed(dist, time_diff_seconds):
        if time_diff_seconds > 0:
            return dist / time_diff_seconds
        return 0

    # Calculate acceleration (meters/second²)
    def calculate_acceleration(speed1, speed2, time_diff_seconds):
        if time_diff_seconds > 0:
            return abs(speed2 - speed1) / time_diff_seconds
        return 0

    if 'points' not in segment or 'times' not in segment or len(segment['points']) < 2:
        return {
            "avg_speed": 0,
            "start_time": None,
            "end_time": None,
            "duration_minutes": 0,
            "total_distance": 0,
            "max_speed": 0,
            "min_speed": 0,
            "max_acceleration": 0
        }

    coords = segment['points']
    times = [convert_timestamp_to_datetime_1(t) for t in segment['times']]

    # Record start time and end time
    start_time = times[0]
    end_time = times[-1]
    duration_minutes = (end_time - start_time).total_seconds() / 60

    total_distance = 0
    total_time = 0

    # Calculate speed between each point
    speeds = []
    accelerations = []
    prev_speed = None

    for i in range(len(coords) - 1):
        lon1, lat1 = coords[i]
        lon2, lat2 = coords[i + 1]
        time1 = times[i]
        time2 = times[i + 1]

        time_diff = (time2 - time1).total_seconds()
        if time_diff <= 0:
            continue

        # Use geopy to calculate distance
        dist_m = calculate_distance(lat1, lon1, lat2, lon2)

        # Calculate current speed
        current_speed = calculate_speed(dist_m, time_diff)

        # Add to speed list
        speeds.append(current_speed)

        # Calculate acceleration
        if prev_speed is not None:
            acceleration = calculate_acceleration(prev_speed, current_speed, time_diff)
            accelerations.append(acceleration)

        prev_speed = current_speed
        total_distance += dist_m
        total_time += time_diff

    # Calculate average speed
    avg_speed = 0
    if total_time > 0:
        avg_speed = (total_distance / total_time)

    # Calculate maximum speed and minimum speed
    max_speed = max(speeds) if speeds else 0
    min_speed = min(speeds) if speeds else 0

    # Calculate maximum acceleration
    max_acceleration = max(accelerations) if accelerations else 0

    # Return dictionary containing all information
    return {
        "start_time": start_time.strftime('%Y-%m-%d %H:%M:%S'),
        "end_time": end_time.strftime('%Y-%m-%d %H:%M:%S'),
        "duration_minutes": round(duration_minutes, 2),
        "total_distance": total_distance,
        "avg_speed": avg_speed,
        "max_speed": max_speed,
        "min_speed": min_speed,
        "max_acceleration": max_acceleration
    }

def generate_llm_prompt_new(current_trajectory_for_prompt, devid, segments):
    """
    Generate prompt for LLM navigation planning and arrival time prediction

    Parameters:
    devid - device ID, used to find detailed information for specific trajectory
    segments - list of trajectory segments
    """
    segment_info = []
    index_start = 0
    index_end = 0

    # Extract information for each trajectory segment
    for i, segment in enumerate(segments):
        segment_details = {"segment_number": i + 1}

        # Only save start time for first segment
        if i == 0 and 'times' in segment:
            segment_details["Start time"] = segment['times'][0]

        # Calculate trajectory segment speed information
        speed_info = calculate_segment_speed_new(segment)
        segment_details["Start time"] = speed_info["start_time"]
        segment_details["End time"] = speed_info["end_time"]
        segment_details["Duration (minutes)"] = speed_info["duration_minutes"]
        segment_details["Total Distance (meters)"] = speed_info["total_distance"]
        segment_details["Average speed (m/s)"] = speed_info["avg_speed"]
        segment_details["Maximum speed (m/s)"] = speed_info["max_speed"]
        segment_details["Minimum speed (m/s)"] = speed_info["min_speed"]
        segment_details["Maximum acceleration (m/s²)"] = speed_info["max_acceleration"]

        # Get main road type
        index_start = index_end + 1
        index_end += len(segment['points'])

        segment_info.append(segment_details)

    # Generate LLM navigation planning prompt
    prompt = generate_navigation_prompt(segment_info)
    prompt += """
            ---

        **Overall Trajectory Visuals:**
        * **Visuals:**
            * `poi_full_image` (Comprehensive view of the entire trajectory with POIs and signalized intersections)
            * `road_structure_full_image` (Detailed illustration of the full trajectory's road class, lanes, and network structure)
        """
    return prompt


def generate_navigation_prompt(segment_info):
    """
    Generate complete LLM navigation planning prompt based on trajectory segment information

    Parameters:
    segment_info - list of trajectory segment information

    Returns:
    complete LLM prompt string
    """
    # Get start time of first segment as start time of entire trajectory
    start_time = None
    if len(segment_info) > 0 and 'start_time' in segment_info[0]:
        start_time = segment_info[0]['start_time']

    prompt = "**The trajectory is segmented based on approximate uniformity in traffic semantics. **\n"

    # Add trajectory start time (if available)
    if start_time:
        prompt += f"**Trajectory Start Time: {start_time} **\n\n"

    prompt += f"** Below are details for the {len(segment_info)} segments of this trajectory, each with statistical information and corresponding visual data: **\n\n"
    # Add detailed information for each trajectory segment
    for info in segment_info:
        prompt += f"**Segment {info['segment_number']}:**\n"
        prompt += f"*   **Visuals:**\n"
        prompt += f"    *   `image{(info['segment_number'] - 1) * 2 + 1}` (Shows segment path, POIs, signalized intersections)\n"
        prompt += f"    *   `image{(info['segment_number'] - 1) * 2 + 2}` (Illustrates road class, lanes, network structure)\n"
        prompt += f"*   **Segment Stats:**\n"

        # Iterate through all key-value pairs and display in specific order
        for key, value in info.items():
            if key == "Start time":
                prompt += f"    *   **Start Time: {value} **\n"
            elif key == "End time":
                prompt += f"    *   **End Time: {value} **\n"
            elif key == "Duration (minutes)":
                prompt += f"    *   **Duration (minutes): {value} **\n"
            elif key == "Distance (m)":
                prompt += f"    *   **Distance (meters): {value} meters **\n"
            elif key == "Total Distance (meters)":
                prompt += f"    *   **Total Distance (meters): {value} **\n"
            elif key == "Average speed (m/s)":
                prompt += f"    *   **Average Speed (m/s): {value} **\n"
            elif key == "Maximum speed (m/s)":
                prompt += f"    *   **Maximum Speed (m/s): {value} **\n"
            elif key == "Minimum speed (m/s)":
                prompt += f"    *   **Minimum Speed (m/s): {value} **\n"
            elif key == "Maximum acceleration (m/s²)":
                prompt += f"    *   **Maximum Acceleration (m/s²): {value} **\n"
            elif key == "Max speed":
                prompt += f"    *   **Max speed: {value} m/s **\n"
            elif key == "Main road types":
                prompt += f"    *   **Main road types: {value} **\n"
            # Skip "segment_number" key as it's already used in segment title
            elif key != "segment_number":
                prompt += f"    *   **{key}: {value} **\n"

        prompt += "\n"

    return prompt

def calculate_segment_speed(segment):
    """
    Calculate average speed of a single trajectory segment

    Parameters:
    segment - trajectory segment

    Returns:
    average speed (m/s)
    """
    if 'points' not in segment or 'times' not in segment or len(segment['points']) < 2:
        return 0

    coords = segment['points']
    times = [convert_timestamp_to_datetime_1(t) for t in segment['times']]

    total_distance = 0
    total_time = 0

    for i in range(len(coords) - 1):
        lon1, lat1 = coords[i]
        lon2, lat2 = coords[i + 1]
        time1 = times[i]
        time2 = times[i + 1]

        time_diff = (time2 - time1).total_seconds()
        if time_diff <= 0:
            continue

        dist_m = haversine_distance(lat1, lon1, lat2, lon2)

        total_distance += dist_m
        total_time += time_diff

    if total_time > 0:
        avg_speed = (total_distance / total_time) # Convert to km/hour
        return avg_speed
    else:
        return 0


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate Haversine distance between two points (in meters)
    """
    # Convert angles to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371000  # Earth average radius (meters)
    return c * r

# processed_trajectories = []
#
# with open('xian/final_data_test.pkl', 'rb') as f:
#     trajectory_list = pickle.load(f)
#
# with open('./xian/fmm_results.pkl', 'rb') as f:
#     results_dict = pickle.load(f)
#
# # Load road segment information
# edges = gpd.read_file('./xian/xian_network/edges.shp')  # Please ensure path is correct
# edges_dict = {row['fid']: row['highway'] for _, row in edges.iterrows()}
# total_segments_count = 0
# index = 0
# for trajectory in tqdm(trajectory_list):
#     devid = trajectory['devid']  # Assume trajectory has devid field
#     trajectory['opath'] = trajectory['opath'].split(',')  # Each element is str type
#     segments = segment_trajectory(trajectory, angle_threshold=30, merge_threshold=5)
#     # Get segment count for single trajectory and add to total segment count
#     prompt = generate_llm_prompt(devid, segments, edges_dict)
#     index += 1
# print(f"All prompts have been saved to ./xian/prompt_results/ directory")
