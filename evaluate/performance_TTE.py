import os
import re
import datetime
import numpy as np
import pickle
from typing import List, Dict

filename = 'o4-mini.txt'

# Load trajectories
with open('../Chengdu/chengdu_demo.pkl', 'rb') as f:
    trajectories = pickle.load(f)


def extract_time_from_llm_output(file_path: str) -> str:
    """Extract predicted time from LLM output file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Try matching triple bracket format
        matches = re.findall(r'<<<([\d-]+ [\d:]+)>>>', content)
        if matches:
            time_str = matches[-1]
            try:
                datetime.datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
                return time_str
            except ValueError:
                return None
        
        # Try matching double bracket format
        matches = re.findall(r'<<([\d-]+ [\d:]+)>>', content)
        if matches:
            time_str = matches[-1]
            try:
                datetime.datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
                return time_str
            except ValueError:
                return None
        
        return None
    except Exception:
        return None

def time_difference_seconds(time1_str: str, time2_str: str) -> float:
    """Calculate time difference in seconds"""
    try:
        time1 = datetime.datetime.strptime(time1_str, '%Y-%m-%d %H:%M:%S')
        time2 = datetime.datetime.strptime(time2_str, '%Y-%m-%d %H:%M:%S')
        return abs((time1 - time2).total_seconds())
    except Exception:
        return None

def calculate_metrics(errors: List[float], journey_durations: List[float]) -> Dict[str, float]:
    """Calculate evaluation metrics: MAE, RMSE, MAPE"""
    if not errors or not journey_durations or len(errors) != len(journey_durations):
        return {'MAE': None, 'RMSE': None, 'MAPE': None}
    
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean([e ** 2 for e in errors]))
    mape = np.mean([abs(errors[i]) / journey_durations[i] * 100 for i in range(len(errors))])
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape
    }

def evaluate_predictions():
    """Evaluate all predictions"""
    base_dir = "../Chengdu/TTE/consolidated_data"
    
    errors = []
    journey_durations = []
    successful_predictions = 0
    failed_predictions = 0
    
    # Find all consolidated_{traj_id} directories
    for item in os.listdir(base_dir):
        if not item.startswith('consolidated_'):
            continue
        
        trajectory_dir = os.path.join(base_dir, item)
        if not os.path.isdir(trajectory_dir):
            continue
        
        traj_id = item.replace('consolidated_', '')
        traj_result = next((t for t in trajectories if t["devid"] == traj_id), None)
        
        if not traj_result:
            failed_predictions += 1
            continue
        
        # Get LLM prediction
        llm_output_path = os.path.join(trajectory_dir, filename)
        predicted_time = extract_time_from_llm_output(llm_output_path)
        
        # Get ground truth from trajectory data
        start_time = traj_result['tms'][0]
        start_time = datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
        
        ground_truth_time = traj_result['tms'][-1]
        ground_truth_time = datetime.datetime.fromtimestamp(ground_truth_time).strftime('%Y-%m-%d %H:%M:%S')
        
        if predicted_time and ground_truth_time and start_time:
            # Calculate prediction error
            time_diff = time_difference_seconds(predicted_time, ground_truth_time)
            # Calculate journey duration
            journey_duration = time_difference_seconds(start_time, ground_truth_time)
            
            if time_diff is not None and journey_duration is not None:
                errors.append(time_diff)
                journey_durations.append(journey_duration)
                successful_predictions += 1
            else:
                failed_predictions += 1
        else:
            failed_predictions += 1
    
    # Calculate and display metrics
    if successful_predictions > 0:
        metrics = calculate_metrics(errors, journey_durations)
        
        print(f"\nDataset Statistics:")
        print(f"Successful evaluations: {successful_predictions}")
        print(f"Failed evaluations: {failed_predictions}")
        print(f"Average journey time: {np.mean(journey_durations):.2f} seconds")
        
        print(f"\nEvaluation Metrics:")
        print(f"Mean Absolute Error (MAE): {metrics['MAE']:.2f} seconds")
        print(f"Root Mean Square Error (RMSE): {metrics['RMSE']:.2f} seconds")
        print(f"Mean Absolute Percentage Error (MAPE): {metrics['MAPE']:.2f}%")
    else:
        print("No successful evaluations found.")

if __name__ == "__main__":
    evaluate_predictions()
