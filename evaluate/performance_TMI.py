import os
import re
from collections import defaultdict
from sklearn.metrics import f1_score


def calculate_classification_accuracy(base_result_path):
    """
    Calculate classification accuracy, Macro-F1, and Weighted-F1 for trajectory predictions.
    Args:
        base_result_path (str): Base directory containing trajectory result folders.
                               Each subfolder is named like {id}_{ground_truth_mode}_{timestamp}
                               and contains 'o4-mini-v1.txt' with predictions.
    """
    correct_predictions = 0
    total_files_processed = 0
    misclassified_folders = []

    # Lists for F1 score calculation
    y_true = []
    y_pred = []

    # Class distribution counter
    class_counts = defaultdict(int)

    # Check if base path exists
    if not os.path.exists(base_result_path):
        print(f"Error: Base result path '{base_result_path}' does not exist.")
        return

    # Iterate over each folder in the base path
    for folder_name in os.listdir(base_result_path):
        folder_path = os.path.join(base_result_path, folder_name)

        # Ensure it's a directory
        if os.path.isdir(folder_path):
            # Extract ground truth from folder name
            try:
                parts = folder_name.split('_')
                # Example: 010_train_20080330084134 -> parts[1] is 'train'
                if len(parts) > 1:
                    ground_truth_mode = parts[1]
                else:
                    print(f"Warning: Could not parse ground truth from folder name '{folder_name}'. Skipping.")
                    continue
            except IndexError:
                print(f"Warning: Folder name '{folder_name}' does not fit expected format. Skipping.")
                continue

            # Check for prediction file
            result_file_path = os.path.join(folder_path, "o4-mini-v1.txt")
            if os.path.exists(result_file_path):
                total_files_processed += 1

                try:
                    with open(result_file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # Extract predicted mode using regex
                    # Expected format: <<<mode>>>
                    match = re.search(r"<<<(.+?)>>>", content)
                    if match:
                        predicted_mode = match.group(1)

                        # Add to label lists for F1 calculation
                        y_true.append(ground_truth_mode)
                        y_pred.append(predicted_mode)
                        class_counts[ground_truth_mode] += 1

                        # Compare prediction with ground truth
                        if predicted_mode == ground_truth_mode:
                            correct_predictions += 1
                        else:
                            misclassified_folders.append(folder_name)
                    else:
                        print(f"Warning: Unexpected format in '{result_file_path}'. Skipping this prediction.")

                except Exception as e:
                    print(f"Error reading file '{result_file_path}': {e}")

    # Calculate and print metrics
    if total_files_processed > 0:
        accuracy = (correct_predictions / total_files_processed) * 100

        # Calculate F1 scores if sufficient data
        if len(y_true) > 0 and len(set(y_true)) > 1:
            macro_f1 = f1_score(y_true, y_pred, average='macro') * 100
            weighted_f1 = f1_score(y_true, y_pred, average='weighted') * 100

            print(f"\n--- Evaluation Results ---")
            print(f"Total trajectories processed: {total_files_processed}")
            print(f"Correct predictions: {correct_predictions}")
            print(f"Accuracy: {accuracy:.2f}%")
            print(f"Macro-F1: {macro_f1:.2f}%")
            print(f"Weighted-F1: {weighted_f1:.2f}%")

            # Print class distribution
            print(f"\n--- Class Distribution ---")
            for cls, count in class_counts.items():
                print(f"Class '{cls}': {count} samples")
        else:
            print(f"\n--- Evaluation Results ---")
            print(f"Total trajectories processed: {total_files_processed}")
            print(f"Correct predictions: {correct_predictions}")
            print(f"Accuracy: {accuracy:.2f}%")
            print("Warning: Cannot calculate F1 scores - insufficient data or classes.")
    else:
        print("\nNo result files found for accuracy calculation.")


if __name__ == "__main__":
    results_directory = "../Beijing"
    calculate_classification_accuracy(results_directory)