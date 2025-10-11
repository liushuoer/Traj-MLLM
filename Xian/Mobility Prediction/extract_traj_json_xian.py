'''
Input: Trajectory sequence, visualization type (road network structure, POI information)
Output: Complete trajectory and segmented json files, saved to specified folder
'''
import os
from utils.tools import *
from utils.segmentation import *

def extract_full_traj_json(trajectory, trajectory_id, vis_type="road_structure"):
    # Create trajectory output directory
    output_dir = '../data/road_structure_trajectory_jsons'
    if vis_type == "poi":
        output_dir = '../data/poi_trajectory_jsons'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process trajectory data
    o_geo = trajectory['o_geo']
    devid = trajectory_id
    segments = segment_trajectory(trajectory, trajectory_id, angle_threshold=30, merge_threshold=5)

    # print(segments)
    for i in range(len(segments)):
        print(len(segments[i]['points']))

    # Create new trajectory object, maintaining original format
    truncated_trajectory = {
        'devid': trajectory_id,
        'o_geo': o_geo,
        'times': trajectory['tms'],
        'road_ids': trajectory['opath']
    }

    # Save each segment to separate JSON file
    for seg_idx, segment in enumerate(segments):
        segment_file = os.path.join(output_dir, f'{devid}_segment_{seg_idx}.json')
        # Convert to trajectory object consistent with original format
        trajectory_segments = {
            'devid': devid,
            'o_geo': segment['points'],
            'times': segment['times'],
            'road_ids': segment['road_ids']
        }
        with open(segment_file, 'w') as f:
           json.dump(trajectory_segments, f, cls=NumpyEncoder, indent=2)

    output_file = os.path.join(output_dir, f'{devid}.json')
    with open(output_file, 'w') as f:
        json.dump(trajectory, f, cls=NumpyEncoder, indent=2)

    # print(f"Processed trajectory, Device ID: {devid}, Total segments: {len(segments)} segments")
    return True  # <--- Add this line to indicate success