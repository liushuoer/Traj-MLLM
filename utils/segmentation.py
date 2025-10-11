import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math
from datetime import datetime
import random
from tqdm import tqdm



def calculate_angle(line1_start, line1_end, line2_start, line2_end):
    """
    Calculate the angle between two line segments (in degrees)
    line1_start, line1_end: start and end points of the first line segment [lon, lat]
    line2_start, line2_end: start and end points of the second line segment [lon, lat]
    """
    # Calculate vectors of line segments
    vector1 = [line1_end[0] - line1_start[0], line1_end[1] - line1_start[1]]
    vector2 = [line2_end[0] - line2_start[0], line2_end[1] - line2_start[1]]

    # Calculate dot product of vectors
    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]

    # Calculate magnitude of vectors
    magnitude1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
    magnitude2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)

    # Calculate angle (in radians)
    try:
        cos_angle = dot_product / (magnitude1 * magnitude2)
        # Ensure cos_angle is between -1 and 1 to avoid numerical errors
        cos_angle = max(-1, min(1, cos_angle))
        angle_rad = math.acos(cos_angle)
        # Convert to degrees
        angle_deg = math.degrees(angle_rad)
        return angle_deg
    except:
        # If line segment length is 0 or other problems occur, return 180 degrees
        return 180.0


def segment_trajectory_equal_parts(line, trajectory_id, num_segments=4):
    """
    Simply divide trajectory into equal parts

    Parameters:
    line: dictionary containing o_geo, opath, devid, tmp
    trajectory_id: trajectory ID
    num_segments: number of segments to divide into, default is 4

    Returns:
    segments: list of segmented trajectory parts, each part is a dictionary containing point and time information
    """
    o_geo = line['o_geo']  # Format: [[lon, lat], [lon, lat], ...]
    devid = trajectory_id  # Device ID
    if 'tmp' in line:
        tmp = line['tmp']  # Timestamp array
    else:
        tmp = line['tms']
    import pytz

    # porto_tz = pytz.timezone('Europe/Lisbon')
    porto_tz = pytz.timezone('Asia/Shanghai')

    # Convert timestamps to readable format
    time_readable = [datetime.fromtimestamp(t, tz=porto_tz).strftime('%Y-%m-%d %H:%M:%S') for t in tmp]

    # Check number of trajectory points
    total_points = len(o_geo)
    if total_points < num_segments:
        # If points are fewer than segments needed, each point becomes a segment
        segments = []
        for i in range(total_points):
            segment = {
                'points': [o_geo[i]],
                'times': [time_readable[i]],
                'devid': devid
            }
            segments.append(segment)
        return segments

    # Calculate number of points per segment
    points_per_segment = total_points // num_segments
    remainder = total_points % num_segments

    segments = []
    start_idx = 0

    for i in range(num_segments):
        # Calculate end index of current segment
        # If there's remainder, first remainder segments each get one extra point
        end_idx = start_idx + points_per_segment + (1 if i < remainder else 0)

        # Create current segment
        segment = {
            'points': o_geo[start_idx:end_idx],
            'times': time_readable[start_idx:end_idx],
            'devid': devid
        }

        segments.append(segment)
        start_idx = end_idx

    return segments



def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate distance between two points using Haversine formula (in meters)
    """
    R = 6371000  # Earth radius (meters)

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    a = (math.sin(delta_lat / 2) ** 2 +
         math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def calculate_speed_factor(trajectory, a, b):
    """
    Calculate speed factor: variance of speeds within segment (Motion Homogeneity)
    Lower variance indicates more consistent motion
    """
    o_geo = trajectory['o_geo']
    timestamps = trajectory.get('tmp', trajectory.get('tms', []))

    if b - a < 2:  # Need at least 2 points to calculate speed
        return 0.0

    speeds = []
    for i in range(a, min(b, len(o_geo) - 1)):
        if i + 1 < len(o_geo) and i + 1 < len(timestamps):
            # Calculate distance and time difference between two points
            lat1, lon1 = o_geo[i][1], o_geo[i][0]
            lat2, lon2 = o_geo[i + 1][1], o_geo[i + 1][0]

            distance = haversine_distance(lat1, lon1, lat2, lon2)
            time_diff = timestamps[i + 1] - timestamps[i]

            if time_diff > 0:
                speed = distance / time_diff  # meters/second
                speeds.append(speed)

    if len(speeds) < 2:
        return 0.0

    # Return speed variance
    return float(np.var(speeds))


def calculate_road_factor(trajectory, a, b):
    """
    Calculate road factor: number of road type changes (Route Homogeneity)
    """
    road_ids = trajectory.get('fmm_traj', trajectory.get('opath', []))

    if b - a < 2 or len(road_ids) == 0:
        return 0.0

    changes = 0
    for i in range(a, min(b, len(road_ids) - 1)):
        if i + 1 < len(road_ids) and road_ids[i] != road_ids[i + 1]:
            changes += 1

    return float(changes)


def calculate_length_factor(trajectory, a, b):
    """
    Calculate length factor: reciprocal of number of points, encourages longer segments (Segment Length Regularization)
    """
    segment_length = b - a + 1
    return 1.0 / segment_length if segment_length > 0 else float('inf')


def calculate_angle_factor(trajectory, a, b):
    """
    Calculate angle factor: accumulation of angle changes within segment
    """
    o_geo = trajectory['o_geo']

    if b - a < 3:  # Need at least 3 points to calculate turning angle
        return 0.0

    total_angle_change = 0.0

    for i in range(a, min(b - 1, len(o_geo) - 2)):
        if i + 2 < len(o_geo):
            # Calculate turning angle of three consecutive points
            p1 = o_geo[i]
            p2 = o_geo[i + 1]
            p3 = o_geo[i + 2]

            # Calculate turning angle
            angle = calculate_turning_angle(p1, p2, p3)
            total_angle_change += abs(180 - angle)  # Degree of deviation from straight line

    return total_angle_change


def calculate_turning_angle(p1, p2, p3):
    """
    Calculate turning angle between three points
    """
    # Calculate vectors
    v1 = [p2[0] - p1[0], p2[1] - p1[1]]
    v2 = [p3[0] - p2[0], p3[1] - p2[1]]

    # Calculate angle
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

    if mag1 == 0 or mag2 == 0:
        return 0.0

    cos_angle = dot_product / (mag1 * mag2)
    cos_angle = max(-1, min(1, cos_angle))  # Ensure within [-1,1] range

    angle = math.acos(cos_angle)
    return math.degrees(angle)


def normalize_factor(value, factor_type, normalization_params):
    """
    Normalize factors using min-max normalization to [0,1] range
    """
    if factor_type not in normalization_params:
        return value

    min_val = normalization_params[factor_type]['min']
    max_val = normalization_params[factor_type]['max']

    if max_val == min_val:
        return 0.0

    normalized = (value - min_val) / (max_val - min_val)
    return max(0.0, min(1.0, normalized))  # Ensure within [0,1] range


def cost_function(trajectory, a, b, factor_weights=None, normalization_params=None):
    """
    Calculate cost of sub-trajectory from position a to position b

    Parameters:
    trajectory: dictionary containing trajectory information
    a, b: start and end indices of sub-trajectory (includes a, excludes b)
    factor_weights: dictionary of factor weights
    normalization_params: dictionary of normalization parameters

    Returns:
    total_cost: total cost
    """
    if factor_weights is None:
        factor_weights = {'speed': 1.0, 'road': 1.0, 'length': 1.0, 'angle': 1.0}

    # Calculate each factor
    f_speed = calculate_speed_factor(trajectory, a, b)
    f_road = calculate_road_factor(trajectory, a, b)
    f_len = calculate_length_factor(trajectory, a, b)
    f_angle = calculate_angle_factor(trajectory, a, b)

    # Normalize (as described in paper)
    if normalization_params:
        f_speed = normalize_factor(f_speed, 'speed', normalization_params)
        f_road = normalize_factor(f_road, 'road', normalization_params)
        f_len = normalize_factor(f_len, 'length', normalization_params)
        f_angle = normalize_factor(f_angle, 'angle', normalization_params)

    # Calculate total cost
    total_cost = (factor_weights['speed'] * f_speed +
                  factor_weights['road'] * f_road +
                  factor_weights['length'] * f_len +
                  factor_weights['angle'] * f_angle)

    return total_cost


def segment_trajectory_dp(line, trajectory_id, factor_weights=None, normalization_params=None, min_segment_length=10):
    """
    Parameters:
    line: dictionary containing trajectory information
    trajectory_id: trajectory ID
    factor_weights: weights for each factor
    normalization_params: normalization parameters
    min_segment_length: minimum segment length
    Returns:
    segments: list of segmented trajectory parts
    """
    o_geo = line['o_geo']
    n = len(o_geo)

    if n < 2:
        return []

    # Set default parameters
    if factor_weights is None:
        factor_weights = {'speed': 1.0, 'road': 1.0, 'length': 0.1, 'angle': 0.5}

    if normalization_params is None:
        # Temporary normalization parameters, should be obtained from dataset statistics in practice
        normalization_params = {
            'speed': {'min': 0.0, 'max': 50.0},  # Speed variance range
            'road': {'min': 0.0, 'max': 20.0},  # Road change count range
            'length': {'min': 0.0, 'max': 1.0},  # Length factor range
            'angle': {'min': 0.0, 'max': 1000.0}  # Angle accumulation range
        }

    # Initialize DP table: DP[h] = minimum cost of segmenting first h points
    dp = [float('inf')] * (n + 1)
    dp[0] = 0.0
    parent = [-1] * (n + 1)  # For backtracking segmentation points

    # Dynamic programming solution (implementing formula 2 from paper)
    for h in range(min_segment_length, n + 1):  # h is current endpoint
        for d in range(0, h - min_segment_length + 1):  # d is candidate previous segmentation point
            if dp[d] != float('inf'):
                cost = cost_function(line, d, h, factor_weights, normalization_params)
                if dp[d] + cost < dp[h]:
                    dp[h] = dp[d] + cost
                    parent[h] = d

    # Backtrack to find optimal segmentation points
    segmentation_points = []
    current = n
    while current > 0:
        segmentation_points.append(current)
        current = parent[current]

    segmentation_points.reverse()

    # Build segmentation results
    segments = build_segments_from_points(line, trajectory_id, segmentation_points)

    return segments


def build_segments_from_points(line, trajectory_id, segmentation_points):
    """
    Build trajectory segments from segmentation points
    """
    o_geo = line['o_geo']
    road_ids = line.get('fmm_traj', line.get('opath', []))
    timestamps = line.get('tmp', line.get('tms', []))

    import pytz
    porto_tz = pytz.timezone('Asia/Shanghai')
    time_readable = [datetime.fromtimestamp(t, tz=porto_tz).strftime('%Y-%m-%d %H:%M:%S')
                     for t in timestamps] if timestamps else []

    segments = []
    start = 0

    for end in segmentation_points:
        if end > start:
            segment = {
                'points': o_geo[start:end],
                'times': time_readable[start:end] if time_readable else [],
                'road_ids': road_ids[start:end] if road_ids else [],
                'devid': trajectory_id
            }
            segments.append(segment)
            start = end

    return segments


# Keep original function for backward compatibility
def calculate_angle(line1_start, line1_end, line2_start, line2_end):
    """
    Calculate the angle between two line segments (in degrees)
    line1_start, line1_end: start and end points of the first line segment [lon, lat]
    line2_start, line2_end: start and end points of the second line segment [lon, lat]
    """
    # Calculate vectors of line segments
    vector1 = [line1_end[0] - line1_start[0], line1_end[1] - line1_start[1]]
    vector2 = [line2_end[0] - line2_start[0], line2_end[1] - line2_start[1]]

    # Calculate dot product of vectors
    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]

    # Calculate magnitude of vectors
    magnitude1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
    magnitude2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)

    # Calculate angle (in radians)
    try:
        cos_angle = dot_product / (magnitude1 * magnitude2)
        # Ensure cos_angle is between -1 and 1 to avoid numerical errors
        cos_angle = max(-1, min(1, cos_angle))
        angle_rad = math.acos(cos_angle)
        # Convert to degrees
        angle_deg = math.degrees(angle_rad)
        return angle_deg
    except:
        # If line segment length is 0 or other problems occur, return 180 degrees
        return 180.0


def segment_trajectory(line, trajectory_id, angle_threshold=60, merge_threshold=5):
    """
    Parameters:
    line: dictionary containing o_geo, opath, devid, tmp

    Returns:
    segments: list of segmented trajectory parts, each part is a dictionary containing point and time information
    """
    o_geo = line['o_geo']  # Format: [[lon, lat], [lon, lat], ...]
    if 'fmm_traj' in line:
        opath = line['fmm_traj']  # Format: [id, id, ...]
    else:
        opath = line['opath']
    devid = trajectory_id  # Device ID
    if 'tmp' in line:
        tmp = line['tmp']  # Timestamp array
    else:
        tmp = line['tms']
    import pytz

    # porto_tz = pytz.timezone('Europe/Lisbon')
    porto_tz = pytz.timezone('Asia/Shanghai')
    # Convert timestamps to readable format
    time_readable = [datetime.fromtimestamp(t, tz=porto_tz).strftime('%Y-%m-%d %H:%M:%S') for t in tmp]
    # Initialize segmentation
    segments = []
    current_segment = {
        'points': [],
        'times': [],
        'road_ids': [],
        'devid': devid
    }

    # Starting road ID
    current_road_id = opath[0] if len(opath) > 0 else None

    # Add first point to current segment
    if len(o_geo) > 0:
        current_segment['points'].append(o_geo[0])
        current_segment['times'].append(time_readable[0])
        current_segment['road_ids'].append(current_road_id)

    for i in range(1, len(o_geo)):
        if opath[i] == current_road_id:
            # If road ID is same, add to current segment
            current_segment['points'].append(o_geo[i])
            current_segment['times'].append(time_readable[i])
            current_segment['road_ids'].append(opath[i])
        else:
            # Road ID is different, need to determine whether to merge
            merge = False

            # Check if current segment and next segment have sufficient length
            if len(current_segment['points']) >= merge_threshold and i + merge_threshold < len(o_geo):
                # Count back merge_threshold points from end of current segment
                last_index = len(current_segment['points']) - 1
                prev_index = max(0, last_index - merge_threshold)

                # Count forward merge_threshold points from start of next segment
                next_index = i + merge_threshold

                # Calculate line segments
                line1_start = current_segment['points'][prev_index]
                line1_end = current_segment['points'][last_index]

                line2_start = o_geo[i]
                line2_end = o_geo[next_index]

                # Calculate angle
                angle = calculate_angle(line1_start, line1_end, line2_start, line2_end)

                # If angle is less than 60 degrees, merge
                if angle < angle_threshold:
                    merge = True

            if merge:
                # Merge into current segment
                current_segment['points'].append(o_geo[i])
                current_segment['times'].append(time_readable[i])
                current_segment['road_ids'].append(opath[i])
                current_road_id = opath[i]  # Update current road ID
            else:
                # Save current segment and start new segment
                segments.append(current_segment)
                current_road_id = opath[i]
                current_segment = {
                    'points': [o_geo[i]],
                    'times': [time_readable[i]],
                    'road_ids': [current_road_id],
                    'devid': devid
                }

    # Add last segment
    if current_segment['points']:
        segments.append(current_segment)
    merged_segments = merge_small_segments(segments)
    merged_segments = fix_segments_continuity(merged_segments)
    # visualize_trajectory_segments(merged_segments, trajectory_id)
    return merged_segments

def fix_segments_continuity_new(segments):
    """
    Ensure continuity between segments: last point of each segment matches first point of next segment.

    Parameters:
    segments: list of trajectory segments

    Returns:
    fixed_segments: list of segments with fixed continuity
    """
    if not segments or len(segments) < 2:
        return segments  # No need to fix if less than 2 segments

    fixed_segments = []

    for i in range(len(segments) - 1):
        # Current segment and next segment
        current_segment = segments[i]
        next_segment = segments[i + 1]

        # Get last point of current segment
        last_point = current_segment['points'][-1]
        last_time = current_segment['times'][-1]

        # Insert last point at beginning of next segment (if not already overlapping)
        if last_point != next_segment['points'][0]:  # Exclude cases that are already overlapping
            next_segment['points'].insert(0, last_point)
            next_segment['times'].insert(0, last_time)

        # Save current segment
        fixed_segments.append(current_segment)

    # Don't forget to add the last segment
    fixed_segments.append(segments[-1])

    return fixed_segments


def fix_segments_continuity(segments):
    """
    Ensure continuity between segments: last point of each segment matches first point of next segment.

    Parameters:
    segments: list of trajectory segments

    Returns:
    fixed_segments: list of segments with fixed continuity
    """
    if not segments or len(segments) < 2:
        return segments  # No need to fix if less than 2 segments

    fixed_segments = []

    for i in range(len(segments) - 1):
        # Current segment and next segment
        current_segment = segments[i]
        next_segment = segments[i + 1]

        # Get last point of current segment
        last_point = current_segment['points'][-1]
        last_time = current_segment['times'][-1]
        last_road_id = current_segment['road_ids'][-1]

        # Insert last point at beginning of next segment (if not already overlapping)
        if last_point != next_segment['points'][0]:  # Exclude cases that are already overlapping
            next_segment['points'].insert(0, last_point)
            next_segment['times'].insert(0, last_time)
            next_segment['road_ids'].insert(0, last_road_id)

        # Save current segment
        fixed_segments.append(current_segment)

    # Don't forget to add the last segment
    fixed_segments.append(segments[-1])

    return fixed_segments


def merge_small_segments(segments, min_points=30):
    """
    Merge small events (events with fewer than min_points points)

    Strategy:
    1. If two adjacent events are both smaller than min_points, merge them
    2. If a small event is adjacent to a large event, merge with the large event

    Parameters:
    segments: list of initially segmented events
    min_points: threshold for small events, default is 10

    Returns:
    merged_segments: list of events after merging small events
    """
    if len(segments) <= 1:
        return segments

    # Mark whether each event is a small event
    is_small = [len(segment['points']) < min_points for segment in segments]

    # Create merged event list
    merged_segments = []
    i = 0

    while i < len(segments):
        current_segment = segments[i].copy()

        # If current event is small
        if is_small[i]:
            # Check if there's a next event
            if i + 1 < len(segments):
                # If next event is also small, or current is small but next is large
                if is_small[i + 1] or (is_small[i] and not is_small[i + 1]):
                    # Merge current event and next event
                    next_segment = segments[i + 1]

                    # Merge points, times and road IDs
                    current_segment['points'].extend(next_segment['points'])
                    current_segment['times'].extend(next_segment['times'])
                    current_segment['road_ids'].extend(next_segment['road_ids'])

                    # Skip next event (because it's already merged)
                    i += 1

        # If previous event is small and current is large, and previous event hasn't been merged with others
        elif i > 0 and is_small[i - 1] and len(merged_segments) > 0 and len(merged_segments[-1]['points']) < min_points:
            # Merge current event (large) into previous event (small)
            merged_segments[-1]['points'].extend(current_segment['points'])
            merged_segments[-1]['times'].extend(current_segment['times'])
            merged_segments[-1]['road_ids'].extend(current_segment['road_ids'])

            # Continue processing next event
            i += 1
            continue

        # Add merged event to result list
        merged_segments.append(current_segment)
        i += 1

    # Check if another merge is needed - handle cases where adjacent small events might still be small after merging
    for _ in range(4):  # Maximum 2 iterations to avoid infinite loop
        if len(merged_segments) <= 1:
            break

        tmp_segments = []
        i = 0

        while i < len(merged_segments):
            current = merged_segments[i].copy()

            # If current event is small and has next event
            if len(current['points']) < min_points and i + 1 < len(merged_segments):
                next_seg = merged_segments[i + 1]

                # If next event is also small or large
                if len(next_seg['points']) < min_points or len(next_seg['points']) >= min_points:
                    # Merge events
                    current['points'].extend(next_seg['points'])
                    current['times'].extend(next_seg['times'])
                    current['road_ids'].extend(next_seg['road_ids'])
                    i += 1  # Skip next event

            tmp_segments.append(current)
            i += 1

        # Check if all small events have been merged
        if all(len(seg['points']) >= min_points for seg in tmp_segments):
            merged_segments = tmp_segments
            break

        merged_segments = tmp_segments
    # Force merge all small events
    final_segments = []
    buffer_segment = None  # Used to store unmerged small events

    for segment in merged_segments:
        if len(segment['points']) < min_points:  # If it's a small event
            if buffer_segment is None:
                buffer_segment = segment
            else:
                # Merge two small events
                buffer_segment['points'].extend(segment['points'])
                buffer_segment['times'].extend(segment['times'])
                buffer_segment['road_ids'].extend(segment['road_ids'])

            # If still smaller than min_points after merging, continue waiting for merge
            if len(buffer_segment['points']) >= min_points:
                final_segments.append(buffer_segment)
                buffer_segment = None
        else:
            # If not a small event, check if there's a small event in buffer to merge
            if buffer_segment is not None:
                segment['points'].extend(buffer_segment['points'])
                segment['times'].extend(buffer_segment['times'])
                segment['road_ids'].extend(buffer_segment['road_ids'])
                buffer_segment = None
            final_segments.append(segment)
    if len(final_segments)!=0:
        # If there's still an unprocessed small event at the end
        if buffer_segment is not None:
            final_segments[-1]['points'].extend(buffer_segment['points'])
            final_segments[-1]['times'].extend(buffer_segment['times'])
            final_segments[-1]['road_ids'].extend(buffer_segment['road_ids'])

    return final_segments

def visualize_trajectory_segments(segments, index):
    """
    Visualize trajectory segments with different colors for different events

    Parameters:
    segments: list of segmented trajectory parts
    """
    plt.figure(figsize=(12, 8))

    # Get different colors
    colors = list(mcolors.TABLEAU_COLORS.values())
    if len(segments) > len(colors):
        # If segments exceed predefined colors, generate random colors
        random.seed(42)  # Set random seed for consistency
        colors = [tuple(random.random() for _ in range(3)) for _ in range(len(segments))]

    for i, segment in enumerate(segments):
        points = segment['points']
        # Extract longitude and latitude points
        lons = [point[0] for point in points]
        lats = [point[1] for point in points]

        # Draw trajectory line
        plt.plot(lons, lats, '-o', color=colors[i % len(colors)],
                 markersize=3, linewidth=2, label=f'segment {i + 1}')

        # Mark start and end points
        plt.scatter(lons[0], lats[0], marker='^', s=100, color=colors[i % len(colors)], edgecolor='black',
                    label=f'start {i + 1}')
        plt.scatter(lons[-1], lats[-1], marker='s', s=100, color=colors[i % len(colors)], edgecolor='black',
                    label=f'end {i + 1}')

    plt.title(f'{segments[0]["devid"]} trajectory segmentation')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)

    # Add legend but don't show duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='best')

    plt.tight_layout()
    # plt.show()

    # for i, segment in enumerate(segments):
    #     print(f"Segment {i + 1}:")
    #     print(f"  Points: {len(segment['points'])}")
    #     unique_road_ids = set(segment['road_ids'])
    #     print(f"  Road IDs: {unique_road_ids}")
    #     print(f"  Start time: {segment['times'][0]}")
    #     print(f"  End time: {segment['times'][-1]}")
    #     print()
