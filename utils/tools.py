import json
import math

from tqdm import tqdm
import numpy as np
earthR = 6378137.0

from tqdm import tqdm
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
import folium
from folium.features import DivIcon
import matplotlib.pyplot as plt
import networkx as nx
import math
from shapely.geometry import Point, LineString

# Load road network data
def load_road_network(node_file, edge_file):
    """Load road network data and build topology relationships"""
    nodes = gpd.read_file(node_file)
    edges = gpd.read_file(edge_file)
    print(f"Loaded {len(nodes)} nodes and {len(edges)} edges")

    # Build road network graph
    G = nx.DiGraph()

    # Add nodes
    for idx, node in nodes.iterrows():
        G.add_node(node['osmid'],
                   geometry=node['geometry'],
                   x=node['geometry'].x,
                   y=node['geometry'].y)

    # Add edges
    for idx, edge in tqdm(edges.iterrows()):
        G.add_edge(edge['u'], edge['v'],
                   edge_id=edge['fid'],
                   geometry=edge['geometry'],
                   length=edge['length'] if 'length' in edge.index else 0)

    # Create mapping from road segment ID to edge
    edge_id_map = {}
    for idx, edge in edges.iterrows():
        # Primarily use fid as key
        if 'fid' in edge:
            edge_id_map[int(edge['fid'])] = edge  # Ensure as integer
            edge_id_map[str(edge['fid'])] = edge  # Also add string form

        # Also add other ID mappings as backup
        if 'osmid' in edge:
            if isinstance(edge['osmid'], list):
                for osmid in edge['osmid']:
                    edge_id_map[str(osmid)] = edge
            else:
                edge_id_map[str(edge['osmid'])] = edge

        # Add index as backup
        edge_id_map[idx] = edge

    return G, nodes, edges, edge_id_map

# Calculate coordinates by extending from start point in specified direction for 100 meters
def extend_line(start_coords, direction, meters=100):
    """
    Extend from start point along specified direction for specified meters

    Parameters:
    start_coords: start point coordinates (lon, lat)
    direction: direction coordinates (lon, lat)
    meters: extension distance in meters

    Returns:
    end point coordinates after extension (lon, lat)
    """
    # Calculate direction vector
    vector = np.array([direction[0] - start_coords[0], direction[1] - start_coords[1]])

    # Normalize direction vector
    magnitude = np.linalg.norm(vector)
    if magnitude == 0:
        return start_coords  # If start and end points coincide, return start point

    unit_vector = vector / magnitude

    # Earth radius (meters)
    earth_radius = 6378137.0

    # Convert distance to radians
    angular_distance = meters / earth_radius

    # Convert latitude and longitude from degrees to radians
    lat1 = math.radians(start_coords[1])
    lon1 = math.radians(start_coords[0])

    # Calculate bearing angle
    bearing = math.atan2(unit_vector[0], unit_vector[1])

    # Use great circle formula to calculate new position
    lat2 = math.asin(math.sin(lat1) * math.cos(angular_distance) +
                     math.cos(lat1) * math.sin(angular_distance) * math.cos(bearing))

    lon2 = lon1 + math.atan2(math.sin(bearing) * math.sin(angular_distance) * math.cos(lat1),
                             math.cos(angular_distance) - math.sin(lat1) * math.sin(lat2))

    # Convert back to degrees
    lat2 = math.degrees(lat2)
    lon2 = math.degrees(lon2)

    return (lon2, lat2)

# Create arrow-shaped coordinate list
def create_arrow_points(end_point, bearing, size=20):
    """
    Create arrow-shaped coordinates based on end point and bearing angle

    Parameters:
    end_point: end point coordinates (lat, lon)
    bearing: bearing angle in degrees
    size: arrow size in meters

    Returns:
    arrow-shaped coordinate list [(lat, lon), ...]
    """
    # Convert to radians
    bearing_rad = math.radians(bearing)

    # Angle offset for left and right sides of arrow (120 degrees)
    left_angle = bearing_rad + math.radians(150)
    right_angle = bearing_rad - math.radians(150)

    # End point coordinates
    end_lat, end_lon = end_point

    # Convert distance to radians (arrow wing length)
    arrow_size_rad = size / 6378137.0  # Earth radius is 6378137.0 meters

    # Calculate left wing coordinates of arrow
    left_lat = math.asin(math.sin(math.radians(end_lat)) * math.cos(arrow_size_rad) +
                         math.cos(math.radians(end_lat)) * math.sin(arrow_size_rad) * math.cos(left_angle))
    left_lon = math.radians(end_lon) + math.atan2(math.sin(left_angle) * math.sin(arrow_size_rad) * math.cos(math.radians(end_lat)),
                                                  math.cos(arrow_size_rad) - math.sin(math.radians(end_lat)) * math.sin(left_lat))
    left_lat = math.degrees(left_lat)
    left_lon = math.degrees(left_lon)

    # Calculate right wing coordinates of arrow
    right_lat = math.asin(math.sin(math.radians(end_lat)) * math.cos(arrow_size_rad) +
                          math.cos(math.radians(end_lat)) * math.sin(arrow_size_rad) * math.cos(right_angle))
    right_lon = math.radians(end_lon) + math.atan2(math.sin(right_angle) * math.sin(arrow_size_rad) * math.cos(math.radians(end_lat)),
                                                   math.cos(arrow_size_rad) - math.sin(math.radians(end_lat)) * math.sin(right_lat))
    right_lat = math.degrees(right_lat)
    right_lon = math.degrees(right_lon)

    # Return three point coordinates of arrow
    return [(end_lat, end_lon), (left_lat, left_lon), (right_lat, right_lon)]

# Calculate bearing angle between two points
def calculate_bearing_1(lon1, lat1, lon2, lat2):
    """Calculate bearing angle from point 1 to point 2 (in degrees)"""
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1

    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)

    bearing = math.atan2(x, y)

    # Convert to degrees
    bearing = math.degrees(bearing)

    # Adjust to geographic bearing (North is 0, clockwise rotation)
    bearing = (bearing + 360) % 360

    return bearing


# Calculate turning angle between two edges
def calculate_turning_angle(edge1, edge2):
    """Calculate turning angle from edge1 to edge2"""
    # Get the last segment direction of edge1
    if isinstance(edge1['geometry'], LineString):
        coords1 = list(edge1['geometry'].coords)
        direction1 = np.array(coords1[-1]) - np.array(coords1[-2])
    else:
        return None  # Cannot calculate

    # Get the first segment direction of edge2
    if isinstance(edge2['geometry'], LineString):
        coords2 = list(edge2['geometry'].coords)
        direction2 = np.array(coords2[1]) - np.array(coords2[0])
    else:
        return None  # Cannot calculate

    # Calculate angle between two vectors
    dot_product = np.dot(direction1, direction2)
    norm1 = np.linalg.norm(direction1)
    norm2 = np.linalg.norm(direction2)

    if norm1 * norm2 == 0:
        return None

    cos_angle = dot_product / (norm1 * norm2)
    cos_angle = min(1.0, max(-1.0, cos_angle))  # Ensure within [-1, 1] range
    angle = np.arccos(cos_angle)

    # Determine left or right turn
    cross_product = np.cross(np.append(direction1, 0), np.append(direction2, 0))[2]
    if cross_product < 0:
        angle = -angle  # Right turn is negative angle

    return math.degrees(angle)

# Determine turn type
def get_turn_type(angle):
    """Determine turn type based on angle"""
    if angle is None:
        return "unknown"

    angle = abs(angle)
    if angle < 30:
        return "straight"
    elif angle < 120:
        if angle > 0:
            return "left_turn"
        else:
            return "right_turn"
    else:
        return "u_turn"

def get_connected_edges(edge_id, edges, G, edge_id_map):
    """Get all road segments connected to the given road segment and their turning relationships"""
    # Find target edge - try multiple forms of keys
    target_edge = None
    key_attempts = [edge_id, str(edge_id), int(float(edge_id)) if str(edge_id).replace('.', '', 1).isdigit() else None]

    for key in key_attempts:
        if key is not None and key in edge_id_map:
            target_edge = edge_id_map[key]
            break

    if target_edge is None:
        print(f"Cannot find road segment ID: {edge_id}, tried forms: {key_attempts}")
        print(f"Some keys in edge_id_map: {list(edge_id_map.keys())[:10]}")
        return None, []

    # Get end node of target edge
    to_node = target_edge['v']

    # Find all edges starting from this node
    connected_edges = []
    for neighbor in G.neighbors(to_node):
        # Get edge data
        edge_data = G.get_edge_data(to_node, neighbor)

        # Check edge data type and handle appropriately
        if isinstance(edge_data, dict):
            # If it's a dictionary, there might be multiple edges
            for key, data in edge_data.items():
                if isinstance(data, dict):
                    connected_edge_id = data.get('edge_id')
                else:
                    # If data is not a dictionary, it might be a direct value
                    connected_edge_id = data if key == 'edge_id' else None
        else:
            # If not a dictionary, it might be a direct value
            connected_edge_id = edge_data

        # If edge_id is not found, try to find directly in graph
        if connected_edge_id is None:
            try:
                connected_edge_id = G[to_node][neighbor].get('edge_id')
            except:
                continue

        # Try to find connected edge in edge_id_map
        connected_edge = None
        if connected_edge_id in edge_id_map:
            connected_edge = edge_id_map[connected_edge_id]
        elif str(connected_edge_id) in edge_id_map:
            connected_edge = edge_id_map[str(connected_edge_id)]
        elif isinstance(connected_edge_id, (int, float)):
            # Try to find matching fid
            for _, edge in edges.iterrows():
                if 'fid' in edge and edge['fid'] == connected_edge_id:
                    connected_edge = edge
                    break

        if connected_edge is not None:
            # Calculate turning angle
            angle = calculate_turning_angle(target_edge, connected_edge)
            turn_type = get_turn_type(angle)

            # Get ID for display
            display_id = connected_edge.get('fid', connected_edge_id) if hasattr(connected_edge, 'get') else connected_edge_id

            connected_edges.append({
                'edge_id': connected_edge_id,
                'display_id': display_id,
                'geometry': connected_edge['geometry'],
                'angle': angle,
                'turn_type': turn_type
            })

    return target_edge, connected_edges

# Visualize road segments and turning relationships
def visualize_trajectory_and_connections(trajectory, G, nodes, edges, edge_id_map):
    """Visualize last few road segments of trajectory and their turning relationships"""
    # Get last few road segments of trajectory
    if 'fmm_traj' in trajectory:
        opath = trajectory['fmm_traj']  # Format: [id, id, ...]
    else:
        opath = trajectory['opath']
    last_segments = opath[-5:]  # Last 5 road segments
    prediction_segment = last_segments[-1]  # Last road segment as prediction target

    print(f"Last 5 segments of trajectory: {last_segments}")
    print(f"Prediction target segment: {prediction_segment}")

    # Get road segments connected to prediction target segment
    target_edge, connected_edges = get_connected_edges(prediction_segment, edges, G, edge_id_map)

    if target_edge is None:
        print(f"Cannot find target segment ID: {prediction_segment}")
        return

    # Print turning relationships
    print("\nRoad segments connected to target segment and turning relationships:")
    for edge in connected_edges:
        print(f"Segment ID: {edge['edge_id']}, Turn type: {edge['turn_type']}, Angle: {edge['angle']:.2f} degrees")

    # Create map
    # Get center point of target segment as map center
    if isinstance(target_edge['geometry'], LineString):
        center_point = list(target_edge['geometry'].centroid.coords)[0]
        center = [center_point[1], center_point[0]]  # [lat, lon]
    else:
        # Find an available center
        for _, edge in edges.iterrows():
            if isinstance(edge['geometry'], LineString):
                center_point = list(edge['geometry'].centroid.coords)[0]
                center = [center_point[1], center_point[0]]
                break

    m = folium.Map(location=center, zoom_start=15, tiles='cartodbpositron')
    # Add trajectory original points
    if 'o_geo' in trajectory and trajectory['o_geo']:
        # Create coordinate list of trajectory points
        trajectory_points = [(point[1], point[0]) for point in trajectory['o_geo'] if point]
        print(trajectory_points[1])

        # Add trajectory line
        folium.PolyLine(
            trajectory_points[-20:-5],
            color='Lime',
            weight=4,
            opacity=0.7,
            tooltip="Trajectory path"
        ).add_to(m)

        # Add trajectory start and end markers
        if trajectory_points:
            # Start point
            folium.Marker(
                trajectory_points[-20],
                icon=folium.Icon(color='green', icon='play', prefix='fa'),
                tooltip="Start point"
            ).add_to(m)

            # End point
            folium.Marker(
                trajectory_points[-5],
                icon=folium.Icon(color='red', icon='stop', prefix='fa'),
                tooltip="End point"
            ).add_to(m)
            # Mark the 10th to 5th last points of trajectory
            if len(trajectory_points) >= 10:
                for i in range(6):
                    index = len(trajectory_points) - 10 + i  # 10th to 5th last points
                    folium.CircleMarker(
                        trajectory_points[index],
                        radius=6,
                        color='red',
                        fill=True,
                        fill_color='red',
                        fill_opacity=0.8,
                        tooltip=f"Trajectory point {10-i} from end"
                    ).add_to(m)


    # Add road segments connected to target segment
    turn_colors = {
        "straight": "green",
        "left_turn": "orange",
        "right_turn": "purple",
        "u_turn": "black",
        "unknown": "gray"
    }
    # Modify visualization code for connected road segments, replace original for loop
    for edge in connected_edges:
        if isinstance(edge['geometry'], LineString):
            # Get start point and second point coordinates of edge to determine direction
            coords = list(edge['geometry'].coords)
            start_point = coords[0]  # (lon, lat)
            if len(coords) > 1:
                direction_point = coords[1]  # Used to determine direction
            else:
                continue  # Skip if only one point

            # Calculate end point after extension
            end_point = extend_line(start_point, direction_point, meters=256)

            # Calculate bearing angle
            bearing = calculate_bearing_1(start_point[0], start_point[1], end_point[0], end_point[1])

            # Create new coordinate list with only start point and extended end point
            new_coords = [(start_point[1], start_point[0]), (end_point[1], end_point[0])]

            # Add road segment line
            color = turn_colors.get(edge['turn_type'], 'gray')
            line = folium.PolyLine(
                new_coords,
                color=color,
                weight=6,
                opacity=0.6,
                tooltip=f"Segment ID: {edge['edge_id']}, Turn: {edge['turn_type']}"
            ).add_to(m)

            # Create arrow coordinates
            arrow_points = create_arrow_points((end_point[1], end_point[0]), bearing, size=40)

            # Add arrow
            folium.Polygon(
                locations=arrow_points,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=1.0,
                weight=1
            ).add_to(m)



    # Add legend
    legend_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; width: 180px; height: 130px;
    background-color: white; border:2px solid grey; z-index:9999; font-size:12px;
    padding: 10px; border-radius: 5px;">
    <b>Turn Types</b><br>
    <i class="fa fa-minus" style="color:red;"></i> Target segment<br>
    <i class="fa fa-minus" style="color:blue;"></i> Trajectory segment<br>
    <i class="fa fa-minus" style="color:green;"></i> Straight<br>
    <i class="fa fa-minus" style="color:orange;"></i> Left turn<br>
    <i class="fa fa-minus" style="color:purple;"></i> Right turn<br>
    <i class="fa fa-minus" style="color:black;"></i> U-turn<br>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    # Save map
    map_file = 'trajectory_connections.html'
    m.save(map_file)
    print(f"Map saved to: {map_file}")

# Calculate trajectory bounding box
def calculate_trajectory_bounds(trajectory):
    """Calculate trajectory bounding box and extend by certain range"""
    # Extract latitude and longitude of trajectory points
    if isinstance(trajectory.get('o_geo', []), list) and len(trajectory.get('o_geo', [])) > 0:
        coords = trajectory['o_geo']
    else:
        print("Cannot find valid trajectory coordinates")
        return None

    # Initialize boundary values
    min_lat = min_lng = float('inf')
    max_lat = max_lng = float('-inf')

    # Traverse all points to update boundary values
    for point in coords:
        if len(point) >= 2:
            lng, lat = point[0], point[1]
            min_lng = min(min_lng, lng)
            max_lng = max(max_lng, lng)
            min_lat = min(min_lat, lat)
            max_lat = max(max_lat, lat)

    # Extend boundaries (add buffer)
    buffer = 0.01  # About 1 kilometer
    return {
        'min_lng': min_lng - buffer,
        'max_lng': max_lng + buffer,
        'min_lat': min_lat - buffer,
        'max_lat': max_lat + buffer
    }

# Check if point is within bounding box
def is_point_in_bounds(point, bounds):
    """Check if point is within bounding box"""
    lng, lat = point
    return (lng >= bounds['min_lng'] and lng <= bounds['max_lng'] and
            lat >= bounds['min_lat'] and lat <= bounds['max_lat'])

# Check if line segment intersects with bounding box
def is_line_in_bounds(line, bounds):
    """Check if line segment intersects with bounding box"""
    # Simple check: if any point is within bounding box, then line segment intersects with bounding box
    for point in line:
        if is_point_in_bounds(point, bounds):
            return True
    return False

# Calculate minimum distance from point to line segment
def distance_to_segment(p, v, w):
    """Calculate minimum distance from point to line segment"""
    # Square length of line segment v-w
    l2 = (v[0] - w[0])**2 + (v[1] - w[1])**2

    # If line segment is actually a point, return point-to-point distance
    if l2 == 0:
        return calculate_distance_1(p[1], p[0], v[1], v[0])

    # Consider line segment v-w as parameterized line segment: v + t (w - v)
    # Parameter t of projection point = ((p-v) . (w-v)) / |w-v|^2
    t = ((p[0] - v[0]) * (w[0] - v[0]) + (p[1] - v[1]) * (w[1] - v[1])) / l2

    if t < 0:
        return calculate_distance_1(p[1], p[0], v[1], v[0])  # Beyond v endpoint
    if t > 1:
        return calculate_distance_1(p[1], p[0], w[1], w[0])  # Beyond w endpoint

    # Projection falls on line segment, calculate projection point
    projection = [
        v[0] + t * (w[0] - v[0]),
        v[1] + t * (w[1] - v[1])
    ]

    return calculate_distance_1(p[1], p[0], projection[1], projection[0])

# Calculate minimum distance from point to trajectory
def min_distance_to_trajectory(point, trajectory_data):
    """Calculate minimum distance from point to trajectory"""
    min_dist = float('inf')

    # Calculate minimum distance from point to trajectory line segments
    for i in range(len(trajectory_data) - 1):
        seg_start = trajectory_data[i]
        seg_end = trajectory_data[i + 1]

        dist = distance_to_segment(
            point,
            (seg_start[0], seg_start[1]),
            (seg_end[0], seg_end[1])
        )

        min_dist = min(min_dist, dist)

    return min_dist

# Calculate minimum distance from trajectory points to line segment
def min_distance_from_trajectory_to_segment(trajectory_data, segment):
    """Calculate minimum distance from trajectory points to line segment"""
    min_dist = float('inf')

    # Calculate minimum distance from each point on trajectory to line segment
    for point in trajectory_data:
        dist = distance_to_segment(
            point,
            (segment[0][0], segment[0][1]),
            (segment[1][0], segment[1][1])
        )

        min_dist = min(min_dist, dist)

    return min_dist

# Calculate bearing angle between two points
def calculate_bearing(lon1, lat1, lon2, lat2):
    """Calculate bearing angle from point 1 to point 2 (in degrees)"""
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1

    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)

    bearing = math.atan2(x, y)

    # Convert to degrees
    bearing = math.degrees(bearing)

    # Adjust to geographic bearing (North is 0, clockwise rotation)
    bearing = (bearing + 360) % 360

    return bearing

# Calculate point position on line segment
def calculate_position_on_line(start, end, fraction):
    """Calculate specific position point on line segment"""
    return [
        start[1] + (end[1] - start[1]) * fraction,  # lat
        start[0] + (end[0] - start[0]) * fraction   # lng
    ]

# Check if position is too close to trajectory points
def is_too_close_to_trajectory(lat, lng, trajectory_points, min_distance=30):
    """Check if position is too close to trajectory points"""
    for point in trajectory_points:
        dist = calculate_distance_1(lat, lng, point[0], point[1])
        if dist < min_distance:
            return True
    return False

def bearing_to_direction(bearing):
    """Convert bearing angle to one of 8 directions"""
    directions = ['North', 'Northeast', 'East', 'Southeast', 'South', 'Southwest', 'West', 'Northwest']
    idx = int((bearing + 22.5) // 45) % 8
    return directions[idx]

def calculate_speed(point1, point2, time1, time2):
    """Calculate speed between two points in meters/second"""
    distance = calculate_distance_1(point1[1], point1[0], point2[1], point2[0])
    time_diff = time2 - time1
    if time_diff == 0:
        return 0
    return distance / time_diff

def print_direction_and_speed_trend(trajectory, trajectory_id):
    coords = trajectory.get('o_geo', [])
    if 'tmp' in trajectory:
        timestamps = trajectory.get('tmp', [])  # Assume trajectory has timestamps in seconds
    else:
        timestamps = trajectory.get('tms', [])  # Assume trajectory has timestamps in seconds
    if 'fmm_traj' in trajectory:
        opath = trajectory['fmm_traj']
    else:
        opath = trajectory['opath']
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

    # Calculate speed trend (simply use final speed minus initial speed to judge)
    trend_value = speeds[-1] - speeds[0]
    if trend_value > 0.1:  # you can adjust this threshold
        speed_trend = "accelerating"
    elif trend_value < -0.1:
        speed_trend = "decelerating"
    else:
        speed_trend = "stable speed"
    output_text = ""
    output_text += "Relevant trajectory information:\n"
    output_text += f"- Last 5 points direction: {direction} (bearing: {bearing:.2f}°)\n"
    output_text += f"- Last 10 points speed trend: {speed_trend}\n\n"
    output_text += f"- Road segment ID for the last points: {opath[-5]}\n"
    return output_text

# Use geometric information of line segment to calculate midpoint
def calculate_midpoint_on_line(line_geom):
    """Calculate midpoint on actual line segment"""
    # Get total length of line
    line_length = line_geom.length

    # Find midpoint on line (geometrically point at half length from start)
    mid_point = line_geom.interpolate(line_length / 2)

    return mid_point.y, mid_point.x  # Return lat, lng

# Visualize road segments and turning relationships
def visualize_trajectory_and_connections(trajectory, G, nodes, edges, edge_id_map, trajectory_id):
    """Visualize last few road segments of trajectory and their turning relationships"""
    # Get last few road segments of trajectory
    if 'fmm_traj' in trajectory:
        opath = trajectory['fmm_traj']  # Format: [id, id, ...]
    else:
        opath = trajectory['opath']
    last_segments = opath[-5:]  # Last 5 road segments
    prediction_segment = last_segments[-1]  # Last road segment as prediction target

    # Add trajectory original points
    if 'o_geo' in trajectory and trajectory['o_geo']:
        # Create coordinate list of trajectory points
        trajectory_points = [(point[1], point[0]) for point in trajectory['o_geo'] if point]
        print_direction_and_speed_trend(trajectory, trajectory_id)

        # Filter edges related to trajectory
        filtered_edges = filter_edges_by_trajectory(trajectory, edges)

        # Calculate trajectory center point
        center_lat = sum(point[1] for point in trajectory_points[-8:]) / len(trajectory_points[-8:])
        center_lng = sum(point[0] for point in trajectory_points[-8:]) / len(trajectory_points[-8:])
        m = folium.Map(location=[center_lng,center_lat], zoom_start=18, tiles='cartodbpositron')

        used_positions = {}
        min_distance = 0.0001  # Minimum distance threshold, about 10 meters

        # Add filtered roads
        # Modify code for adding edge ID and arrows
        # Add edge ID and direction arrows with adaptive position adjustment and road direction arrows
        for edge in filtered_edges:
            if isinstance(edge['geometry'], LineString):
                coords = list(edge['geometry'].coords)
                edge_id = edge.get('fid', 'NA')

                # Draw road line
                line_locations = [[coord[1], coord[0]] for coord in coords]
                line = folium.PolyLine(
                    locations=line_locations,
                    color='blue',
                    weight=3,
                    opacity=0.7
                )
                line.add_to(m)

                # Add road direction arrows (add 1-2 arrows on line segment)
                if len(coords) >= 2:
                    # If line segment is long enough, add arrows
                    for i in range(len(coords) - 1):
                        start_point = coords[i]
                        end_point = coords[i+1]

                        # Calculate line segment length
                        segment_length = calculate_distance_1(
                            start_point[1], start_point[0],  # lat1, lng1
                            end_point[1], end_point[0]       # lat2, lng2
                        )

                        # Skip arrow addition for this segment if segment length is less than 30 meters
                        if segment_length < 30:
                            continue

                        # Add arrows at certain intervals to avoid being too dense
                        if i % 2 == 0 or len(coords) <= 3:
                            # Calculate arrow position (midpoint of segment)
                            arrow_lat = (start_point[1] + end_point[1]) / 2
                            arrow_lng = (start_point[0] + end_point[0]) / 2

                            # Calculate direction angle
                            bearing = calculate_bearing(start_point[0], start_point[1], end_point[0], end_point[1])
                            bearing = (bearing - 90) % 360  # Rotate 90 degrees counterclockwise

                            # Add arrow marker
                            folium.Marker(
                                location=[arrow_lat, arrow_lng],
                                icon=DivIcon(
                                    icon_size=(16, 16),
                                    icon_anchor=(8, 8),
                                    html=f'''
                                    <div style="transform: rotate({bearing}deg);">
                                        <svg height="16" width="16" viewBox="0 0 100 100">
                                            <polygon points="0,0 100,50 0,100" fill="blue" />
                                        </svg>
                                    </div>
                                    '''
                                )
                            ).add_to(m)
                # Calculate road midpoint position for displaying ID
                if len(coords) >= 2:
                    start_point = coords[0]
                    end_point = coords[-1]

                    # Calculate distance between start and end points (in meters)
                    distance = calculate_distance_1(start_point[1], start_point[0], end_point[1], end_point[0])

                    # Only add marker when distance is greater than or equal to 20 meters
                    if distance >= 30:
                        # Create dictionary to store used positions
                        if 'used_positions' not in locals():
                            used_positions = {}

                        # Calculate line segment direction angle
                        bearing = calculate_bearing(start_point[0], start_point[1], end_point[0], end_point[1])

                        # Calculate midpoint and nearby positions
                        # Assume edge_geometry is the LineString geometry object of this edge
                        if isinstance(edge['geometry'], LineString):
                            mid_lat, mid_lng = calculate_midpoint_on_line(edge['geometry'])

                        # Add edge ID and direction arrow
                        folium.Marker(
                            location=[mid_lat, mid_lng],
                            icon=DivIcon(
                                icon_size=(46, 16),
                                icon_anchor=(23, 8),
                                html=f'''
                                <div style="
                                    display: flex;
                                    align-items: center;
                                    background-color: rgba(255, 255, 255, 0.85);
                                    border: 2px solid blue;
                                    border-radius: 4px;
                                    padding: 0 4px;
                                    font-size: 12px;
                                    font-weight: bold;
                                    white-space: nowrap;
                                    color: blue;
                                    box-shadow: 0 0 2px rgba(0, 0, 0, 0.2);
                                ">
                                    <span style="margin-right: 4px;">{edge_id}</span>
                                </div>
                                '''
                            ),
                            zIndexOffset=1000  # Set large offset value to ensure on top layer
                        ).add_to(m)


        # Add trajectory line
        folium.PolyLine(
            trajectory_points[-20:-4],
            color='Lime',
            weight=10,
            opacity=0.7,
            tooltip="Trajectory path"
        ).add_to(m)

        # Add trajectory start and end markers
        if trajectory_points:
            # Mark the 10th to 5th last points of trajectory
            if len(trajectory_points) >= 10:
                for i in range(6):
                    index = len(trajectory_points) - 10 + i  # 10th to 5th last points
                    folium.CircleMarker(
                        trajectory_points[index],
                        radius=6,
                        color='red',
                        fill=True,
                        fill_color='red',
                        fill_opacity=0.8,
                        tooltip=f"Trajectory point {10-i} from end"
                    ).add_to(m)

    return m

def filter_edges_by_trajectory(trajectory, edges, distance_threshold=0.001):
    """Filter edges related to trajectory"""
    # Calculate trajectory bounding box
    bounds = calculate_trajectory_bounds(trajectory)
    if bounds is None:
        return []

    trajectory_data = trajectory['o_geo'][-5:]

    # Calculate overall movement direction of trajectory
    if len(trajectory_data) >= 2:
        # Use last few points to calculate trajectory direction
        start_point = trajectory_data[0]
        end_point = trajectory_data[-1]
        trajectory_bearing = calculate_bearing(start_point[0], start_point[1], end_point[0], end_point[1])
    else:
        # If trajectory points are insufficient, cannot calculate direction, so don't filter by direction
        trajectory_bearing = None

    # First use bounding box for coarse filtering
    pre_filtered_edges = []
    for _, edge in edges.iterrows():
        if isinstance(edge['geometry'], LineString):
            coords = list(edge['geometry'].coords)
            if is_line_in_bounds(coords, bounds):
                pre_filtered_edges.append(edge)


    # Then perform precise distance filtering
    filtered_edges = []
    for edge in pre_filtered_edges:
        if isinstance(edge['geometry'], LineString):
            coords = list(edge['geometry'].coords)

            # For each line segment, calculate minimum distance from trajectory points to line segment
            for i in range(len(coords) - 1):
                segment = [coords[i], coords[i+1]]
                dist = min_distance_from_trajectory_to_segment(trajectory_data, segment)

                if dist < distance_threshold * 111000:  # Convert to meters (1 degree ≈ 111km)
                    # Calculate road segment direction
                    if trajectory_bearing is not None:
                        segment_bearing = calculate_bearing(
                            segment[0][0], segment[0][1],
                            segment[1][0], segment[1][1]
                        )

                        # Calculate direction difference (angle difference)
                        angle_diff = abs(trajectory_bearing - segment_bearing)
                        if angle_diff > 180:
                            angle_diff = 360 - angle_diff

                        # If direction difference is greater than 120 degrees (almost opposite), exclude this road segment
                        if angle_diff > 120:
                            continue  # Skip this road segment, don't add to result

                    filtered_edges.append(edge)
                    break

    return filtered_edges

# Calculate geographic distance between two points (in meters)
def calculate_distance_1(lat1, lng1, lat2, lng2):
    """Calculate geographic distance between two points (in meters)"""
    R = 6371000  # Earth radius in meters
    dLat = math.radians(lat2 - lat1)
    dLng = math.radians(lng2 - lng1)
    a = (math.sin(dLat/2) * math.sin(dLat/2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dLng/2) * math.sin(dLng/2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c
    return distance

# Calculate distance between two points (in meters)
def calculate_distance(point1, point2):
    R = 6371000  # Earth radius in meters
    lat1, lon1 = map(math.radians, [point1[1], point1[0]])
    lat2, lon2 = map(math.radians, [point2[1], point2[0]])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c

# Calculate midpoint of line segment
def calculate_midpoint(point1, point2):
    return ((point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2)

def load_co_f(s):
    p = s.split(',')
    p_ = [float(i) for i in p]
    return p_

def load_or_i(s):
    def f(l):
        return [int(i) for i in l]

    p = s.split('|')
    p_ = [f(i.split(',')) for i in p]
    return p_

# Calculate vector between two coordinate points
def calculate_vector(point1, point2):
    return (point2[0] - point1[0], point2[1] - point1[1])

# Helper function: format timestamp, add date and day of week
def format_timestamp(timestamp, include_date=False):
    from datetime import datetime
    dt = datetime.fromtimestamp(timestamp)

    if include_date:
        # English weekday representation
        weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        weekday = weekdays[dt.weekday()]
        return f"{dt.year}-{dt.month:02d}-{dt.day:02d} {weekday} {dt.strftime('%H:%M:%S')}"
    else:
        return dt.strftime('%H:%M:%S')

# Assume we have a function to read and process all trajectory data in pkl file
def preprocess_trajectory_data(pkl_file_path):
    """Preprocess trajectory data, return index structure for quick lookup"""
    import pickle

    # Read pkl file
    with open(pkl_file_path, 'rb') as f:
        data = pickle.load(f)

    # Build index from road segment ID to trajectories
    road_segment_to_trajectories = {}
    from shapely.wkt import loads
    for line in tqdm(data):
        o_geo = line['o_geo']
        opath = line['opath']
        devid = line['devid']
        if 'tmp' in line:
            tmp = line['tmp']  # Timestamp array
        else:
            tmp = line['tms']
        o_geo = list(loads(o_geo).coords)
        opath = load_co_i(opath)
        # Ensure time data and geographic data have consistent length
        if len(tmp) != len(o_geo) or len(opath) != len(o_geo):
            continue

        # Build index for each road segment ID
        for i, road_id in enumerate(opath):
            if i < len(tmp) - 1:  # Ensure there is next time point
                # Calculate speed from this point to next point
                if road_id not in road_segment_to_trajectories:
                    road_segment_to_trajectories[road_id] = []

                # Only calculate speed when there is next point and time difference between two points
                if i < len(o_geo) - 1 and tmp[i + 1] > tmp[i]:
                    distance = calculate_distance(o_geo[i], o_geo[i + 1])
                    time_diff = tmp[i + 1] - tmp[i]
                    if time_diff > 0:
                        speed = distance / time_diff
                        road_segment_to_trajectories[road_id].append({
                            'devid': devid,
                            'timestamp': tmp[i],
                            'speed': speed,
                            'distance': distance
                        })
    with open("road_segment_to_trajectories.pkl", 'wb') as f:
        pickle.dump(road_segment_to_trajectories, f)

    return road_segment_to_trajectories


# Calculate average speed of specified road segment within given time period
def get_average_speed_for_segment(road_segment_to_trajectories, road_id, time_point, time_threshold=2,
                                  min_trajectories=50):
    """Get average speed of specified road segment within given time period"""
    if road_id not in road_segment_to_trajectories:
        return None

    trajectories = road_segment_to_trajectories[road_id]

    # Initial threshold
    current_threshold = time_threshold
    max_threshold = 10  # Maximum threshold 5 minutes

    while current_threshold <= max_threshold:
        relevant_trajectories = []

        for traj in trajectories:
            # Check if within time threshold
            if abs(traj['timestamp'] - time_point) <= current_threshold:
                relevant_trajectories.append(traj)

        # If enough trajectories are found, calculate average speed
        if len(relevant_trajectories) >= min_trajectories:
            total_speed = sum(traj['speed'] for traj in relevant_trajectories)
            return total_speed / len(relevant_trajectories)

        # Otherwise increase time threshold and continue trying
        current_threshold += 2

    # If even expanding threshold cannot find enough trajectories, use all available trajectories
    if trajectories:
        total_speed = sum(traj['speed'] for traj in trajectories)
        return total_speed / len(trajectories)

    return None

# Helper function: format timestamp
def format_timestamp_1(timestamp):
    from datetime import datetime
    return datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')


# Helper function: format time duration
def format_time_duration(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)

    if hours > 0:
        return f"{hours}h{minutes}m{seconds}s"
    elif minutes > 0:
        return f"{minutes}m{seconds}s"
    else:
        return f"{seconds}s"

def outOfChina(lat, lng):
    return not (72.004 <= lng <= 137.8347 and 0.8293 <= lat <= 55.8271)

# Calculate perpendicular distance from point to line segment
def point_to_segment_distance(point, segment_start, segment_end):
    """Calculate perpendicular distance from point to line segment"""
    px, py = point
    sx1, sy1 = segment_start
    sx2, sy2 = segment_end

    if (sx1 == sx2) and (sy1 == sy2):
        # Start and end points are the same, degenerate to calculating point-to-point distance
        return math.sqrt((px - sx1) ** 2 + (py - sy1) ** 2)

    # Calculate square of line segment length
    line_length_sq = (sx2 - sx1) ** 2 + (sy2 - sy1) ** 2

    # Calculate proportion parameter t of projection point
    t = ((px - sx1) * (sx2 - sx1) + (py - sy1) * (sy2 - sy1)) / line_length_sq
    t = max(0, min(1, t))  # Limit t to [0, 1] range

    # Calculate coordinates of projection point
    proj_x = sx1 + t * (sx2 - sx1)
    proj_y = sy1 + t * (sy2 - sy1)

    # Calculate perpendicular distance
    return math.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)

def load_co_i(s):
    p = s.split(',')
    p_ = [int(i) for i in p]
    return p_

# Handle JSON serialization of NumPy arrays
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def swap_lat_lng(trajectory):
    swapped_coords = []
    for coord in trajectory:
        # Ensure coordinate is a list containing two elements
        if len(coord) == 2:
            # Swap longitude and latitude positions [lng, lat] -> [lat, lng] or [lat, lng] -> [lng, lat]
            swapped_coords.append([coord[1], coord[0]])
        else:
            # If coordinate is not in expected format, keep as is
            swapped_coords.append(coord)

    return swapped_coords

def transform(x, y):
    xy = x * y
    absX = math.sqrt(abs(x))
    xPi = x * math.pi
    yPi = y * math.pi
    d = 20.0*math.sin(6.0*xPi) + 20.0*math.sin(2.0*xPi)

    lat = d
    lng = d

    lat += 20.0*math.sin(yPi) + 40.0*math.sin(yPi/3.0)
    lng += 20.0*math.sin(xPi) + 40.0*math.sin(xPi/3.0)

    lat += 160.0*math.sin(yPi/12.0) + 320*math.sin(yPi/30.0)
    lng += 150.0*math.sin(xPi/12.0) + 300.0*math.sin(xPi/30.0)

    lat *= 2.0 / 3.0
    lng *= 2.0 / 3.0

    lat += -100.0 + 2.0*x + 3.0*y + 0.2*y*y + 0.1*xy + 0.2*absX
    lng += 300.0 + x + 2.0*y + 0.1*x*x + 0.1*xy + 0.1*absX

    return lat, lng


def delta(lat, lng):
    ee = 0.00669342162296594323
    dLat, dLng = transform(lng-105.0, lat-35.0)
    radLat = lat / 180.0 * math.pi
    magic = math.sin(radLat)
    magic = 1 - ee * magic * magic
    sqrtMagic = math.sqrt(magic)
    dLat = (dLat * 180.0) / ((earthR * (1 - ee)) / (magic * sqrtMagic) * math.pi)
    dLng = (dLng * 180.0) / (earthR / sqrtMagic * math.cos(radLat) * math.pi)
    return dLat, dLng


def wgs2gcj(wgsLat, wgsLng):
    if outOfChina(wgsLat, wgsLng):
        return wgsLat, wgsLng
    else:
        dlat, dlng = delta(wgsLat, wgsLng)
        return wgsLat + dlat, wgsLng + dlng

def gcj2wgs(gcjLat, gcjLng):
    if outOfChina(gcjLat, gcjLng):
        return gcjLat, gcjLng
    else:
        dlat, dlng = delta(gcjLat, gcjLng)
        return gcjLat - dlat, gcjLng - dlng

def traj2wgs(points):
    traj_wgs = []
    for p in points:
        traj_wgs.append(gcj2wgs(p[1],p[0]))
    return traj_wgs
