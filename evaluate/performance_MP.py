import pickle
import os
import re
import ast

# Configuration
BASE_DIR = "../Chengdu/TTE/consolidated_data"
PREDICTION_FILE_NAME = "mp-o4-mini.txt"
GROUND_TRUTH_FILE = '../Chengdu/chengdu_demo.pkl'


def load_ground_truth_data(filepath):
    """Load trajectory data from pickle file"""
    try:
        with open(filepath, 'rb') as f:
            trajectories_data = pickle.load(f)
        print(f"Successfully loaded ground truth data: {filepath}")
        return trajectories_data
    except FileNotFoundError:
        print(f"Error: Ground truth file not found: {filepath}")
        return None
    except Exception as e:
        print(f"Error loading ground truth file: {e}")
        return None


def find_trajectory_by_id(trajectories_list, traj_id):
    """Find trajectory in list by ID"""
    for traj in trajectories_list:
        # Check different possible ID fields
        if isinstance(traj, dict):
            # Try common ID field names
            for id_field in ['id', 'traj_id', 'trajectory_id', 'devid', 'device_id']:
                if id_field in traj and str(traj[id_field]) == traj_id:
                    return traj
    return None


def parse_prediction_file(filepath):
    """
    Parse prediction file to extract Top 1 and Top 5 road_ids
    Returns (top1_road_id, list_of_top5_road_ids)
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        top1_road_id = None
        top5_road_ids = []

        # Try format 1: <<<road_id>>>
        top1_match = re.search(r'<<<([^>]+?)>>>', content)
        if top1_match:
            top1_road_id = top1_match.group(1).strip()

        # Try format 1: [[[road_id_1,road_id_2]]]
        top5_match = re.search(r'\[\[\[([^\]]+?)\]\]\]', content)
        if top5_match:
            ids_str = top5_match.group(1)
            top5_road_ids = [s.strip() for s in ids_str.split(',') if s.strip()]

        # If format 1 fails, try format 2
        if top1_road_id is None:
            top1_match2 = re.search(r'Top-1 prediction:\s*(\d+)', content)
            if top1_match2:
                top1_road_id = top1_match2.group(1).strip()

        if not top5_road_ids:
            top5_match2 = re.search(r'Top-5 candidates:\s*(\[\[.+?\]\])', content)
            if top5_match2:
                try:
                    top5_list_str = top5_match2.group(1)
                    parsed_list = ast.literal_eval(top5_list_str)
                    if isinstance(parsed_list, list) and len(parsed_list) > 0 and isinstance(parsed_list[0], list):
                        top5_road_ids = [str(road_id) for road_id in parsed_list[0]]
                    elif isinstance(parsed_list, list):
                        top5_road_ids = [str(road_id) for road_id in parsed_list]
                except (SyntaxError, ValueError) as e:
                    print(f"Warning: Error parsing Top-5 list: {e}")

        # Convert to strings for comparison
        if top1_road_id is not None:
            top1_road_id = str(top1_road_id)
        top5_road_ids = [str(s_id) for s_id in top5_road_ids]

        return top1_road_id, top5_road_ids

    except FileNotFoundError:
        print(f"Warning: Prediction file not found: {filepath}")
        return None, []
    except Exception as e:
        print(f"Warning: Error parsing prediction file {filepath}: {e}")
        return None, []


def main():
    ground_truth_trajectories = load_ground_truth_data(GROUND_TRUTH_FILE)
    if not ground_truth_trajectories:
        return

    # Initialize counters
    total_evaluated_samples = 0
    correct_top1_predictions = 0
    correct_top5_predictions = 0

    # Get trajectory directories
    traj_dirs = [d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))]
    if not traj_dirs:
        print(f"No trajectory folders found in '{BASE_DIR}'")
        return

    print(f"\nEvaluating {len(traj_dirs)} trajectories...")

    for folder_name in traj_dirs:
        prediction_file_path = os.path.join(BASE_DIR, folder_name, PREDICTION_FILE_NAME)

        # Check if prediction file exists
        if not os.path.exists(prediction_file_path):
            print(f"Warning: Prediction file not found for folder '{folder_name}'. Skipping...")
            continue

        # Extract trajectory ID by removing "consolidated_" prefix
        if folder_name.startswith('consolidated_'):
            traj_id = folder_name.replace('consolidated_', '')
        else:
            traj_id = folder_name

        # Find trajectory in ground truth data
        trajectory_data = find_trajectory_by_id(ground_truth_trajectories, traj_id)
        if trajectory_data is None:
            print(f"Warning: Trajectory ID '{traj_id}' not found in ground truth data. Skipping...")
            continue

        try:
            if 'opath' not in trajectory_data or not trajectory_data['opath']:
                print(f"Warning: Empty 'opath' for trajectory '{traj_id}'. Skipping...")
                continue

            gt_road_id = str(trajectory_data['opath'][-1])
        except (KeyError, IndexError, TypeError) as e:
            print(f"Warning: Error getting ground truth for trajectory '{traj_id}': {e}. Skipping...")
            continue

        # Parse prediction file
        pred_top1, pred_top5_list = parse_prediction_file(prediction_file_path)
        print(pred_top1, pred_top5_list)
        print(gt_road_id)

        if pred_top1 is None and not pred_top5_list:
            print(f"Warning: Failed to parse predictions for trajectory '{traj_id}'. Skipping...")
            continue

        total_evaluated_samples += 1

        # Calculate ACC@1
        if pred_top1 is not None and pred_top1 == gt_road_id:
            correct_top1_predictions += 1

        # Calculate ACC@5
        if gt_road_id in pred_top5_list:
            correct_top5_predictions += 1

    # Calculate final metrics
    if total_evaluated_samples > 0:
        accuracy_at_1 = (correct_top1_predictions / total_evaluated_samples) * 100
        accuracy_at_5 = (correct_top5_predictions / total_evaluated_samples) * 100
        recall_at_5 = accuracy_at_5
    else:
        accuracy_at_1 = 0
        accuracy_at_5 = 0
        recall_at_5 = 0

    # Print results
    print("\n--- Evaluation Results ---")
    print(f"Total evaluated trajectories: {total_evaluated_samples}")
    print(f"ACC@1 (Top 1 Accuracy): {accuracy_at_1:.2f}%")
    print(f"ACC@5 (Top 5 Accuracy): {accuracy_at_5:.2f}%")
    print(f"Recall@5 (Top 5 Recall): {recall_at_5:.2f}%")


if __name__ == "__main__":
    main()