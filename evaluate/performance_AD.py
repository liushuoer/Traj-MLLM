import os
import re
import numpy as np
from sklearn.metrics import precision_recall_curve, auc, average_precision_score
import matplotlib.pyplot as plt


def process_result_folder(folder_path, all_results, normal_folders, abnormal_folders):
    """
    处理一个结果文件夹，提取预测结果

    Args:
        folder_path (str): 结果文件夹路径
        all_results (list): 用于存储所有结果的列表
        normal_folders (list): 用于存储正常轨迹文件夹的列表
        abnormal_folders (list): 用于存储异常轨迹文件夹的列表
    """
    # 检查路径是否存在
    if not os.path.exists(folder_path):
        print(f"错误：结果路径 '{folder_path}' 不存在。")
        return

    # 编译正则表达式模式，用于匹配新的结果格式
    judgment_pattern = re.compile(r"(?:\*\*)?Final\s+Judgment:(?:\*\*)?\s*(Normal|Abnormal)", re.IGNORECASE)

    # 遍历结果路径中的每个项目（预期是文件夹）
    for traj_id_folder_name in os.listdir(folder_path):
        traj_folder_full_path = os.path.join(folder_path, traj_id_folder_name)

        # 确保它是一个目录
        if os.path.isdir(traj_folder_full_path):
            result_file_path = os.path.join(traj_folder_full_path, "ad-o4-mini.txt")

            if os.path.exists(result_file_path):
                # 根据文件夹名确定真实标签
                if "consolidated_normal" in traj_id_folder_name:
                    ground_truth_label = "normal"
                    normal_folders.append(traj_id_folder_name)
                elif "consolidated_anomaly" in traj_id_folder_name:
                    ground_truth_label = "abnormal"
                    abnormal_folders.append(traj_id_folder_name)
                else:
                    print(f"警告：无法从文件夹名称 '{traj_id_folder_name}' 确定标签。跳过此轨迹。")
                    continue

                try:
                    with open(result_file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # 使用正则表达式查找判断结果
                    match = judgment_pattern.search(content)

                    if match:
                        # 提取判断结果（Normal或Abnormal）并转换为小写
                        judgment = match.group(1).lower()

                        # 保存结果
                        all_results.append({
                            'folder_name': traj_id_folder_name,
                            'ground_truth': ground_truth_label,
                            'prediction': judgment
                        })
                    else:
                        print(f"警告：在 '{result_file_path}' 中未找到匹配的判断格式。此预测将被跳过。")

                except Exception as e:
                    print(f"错误：读取或处理文件 '{result_file_path}' 失败: {e}")


def calculate_anomaly_detection_metrics(anomaly_result_path, normal_result_path):
    """
    计算异常检测的精确率、召回率和PR-AUC。
    从两个不同的路径读取结果并合并进行评估。

    Args:
        anomaly_result_path (str): 包含异常轨迹结果的目录
        normal_result_path (str): 包含正常轨迹结果的目录
    """
    # 用于存储所有结果的列表
    all_results = []
    normal_folders = []
    abnormal_folders = []

    # 处理两个结果文件夹
    print(f"处理异常轨迹结果文件夹: {anomaly_result_path}")
    process_result_folder(anomaly_result_path, all_results, normal_folders, abnormal_folders)

    print(f"处理正常轨迹结果文件夹: {normal_result_path}")
    process_result_folder(normal_result_path, all_results, normal_folders, abnormal_folders)

    # 如果没有找到结果，则退出
    if not all_results:
        print("\n未找到结果文件用于计算指标。")
        return

    # 计算当前正常和异常轨迹数量
    num_normal = len(normal_folders)
    num_abnormal = len(abnormal_folders)

    print(f"\n数据集统计:")
    print(f"正常轨迹数量: {num_normal}")
    print(f"异常轨迹数量: {num_abnormal}")
    print(f"比例: {num_normal / max(1, num_abnormal):.2f}:1")

    # 计算混淆矩阵的组成部分
    true_positives = sum(1 for r in all_results if r['ground_truth'] == 'abnormal' and r['prediction'] == 'abnormal')
    false_positives = sum(1 for r in all_results if r['ground_truth'] == 'normal' and r['prediction'] == 'abnormal')
    true_negatives = sum(1 for r in all_results if r['ground_truth'] == 'normal' and r['prediction'] == 'normal')
    false_negatives = sum(1 for r in all_results if r['ground_truth'] == 'abnormal' and r['prediction'] == 'normal')

    # 计算指标
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (true_positives + true_negatives) / len(all_results)

    # 输出混淆矩阵
    print("\n混淆矩阵:")
    print(f"{'':15} | {'预测:异常':15} | {'预测:正常':15}")
    print(f"{'-' * 15}-+-{'-' * 15}-+-{'-' * 15}")
    print(f"{'实际:异常':15} | {true_positives:15} | {false_negatives:15}")
    print(f"{'实际:正常':15} | {false_positives:15} | {true_negatives:15}")

    # 输出指标
    print("\n性能指标:")
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1分数: {f1_score:.4f}")

    # 输出错误预测的轨迹ID
    false_positive_ids = [r['folder_name'] for r in all_results if
                          r['ground_truth'] == 'normal' and r['prediction'] == 'abnormal']
    print("\n实际正常但预测为异常的轨迹ID (假阳性):")
    for traj_id in false_positive_ids:
        print(f"  - {traj_id}")

    false_negative_ids = [r['folder_name'] for r in all_results if
                          r['ground_truth'] == 'abnormal' and r['prediction'] == 'normal']
    print("\n实际异常但预测为正常的轨迹ID (假阴性):")
    for traj_id in false_negative_ids:
        print(f"  - {traj_id}")

    # 计算PR曲线和AUC
    # 为PR曲线准备数据
    y_true = np.array([1 if r['ground_truth'] == 'abnormal' else 0 for r in all_results])

    # 这里我们只有二分类结果，没有概率分数，所以PR曲线会比较简单
    # 为了计算PR-AUC，我们可以使用一个简单的技巧：将预测结果转换为0或1
    y_pred = np.array([1 if r['prediction'] == 'abnormal' else 0 for r in all_results])

    # 计算PR曲线点
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred)

    # 计算PR-AUC
    pr_auc = auc(recall_curve, precision_curve)

    # 也可以使用average_precision_score，它是PR曲线下面积的另一种计算方式
    ap = average_precision_score(y_true, y_pred)

    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"平均精确率 (Average Precision): {ap:.4f}")


if __name__ == "__main__":
    # 指定结果目录路径
    anomaly_results_directory = '../Chengdu/Anomaly Detection/consolidated_data'
    normal_results_directory = "../Chengdu/Anomaly Detection/consolidated_data"

    # 计算异常检测指标
    calculate_anomaly_detection_metrics(anomaly_results_directory, normal_results_directory)