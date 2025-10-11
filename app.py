from flask import Flask, request, jsonify, render_template, send_file
import os
import json
import glob
from pathlib import Path
import re

app = Flask(__name__)

# Dataset configuration - using relative paths
DATASET_CONFIG = {
    'Chengdu': {
        'base_path': './Chengdu',
        'trajectory_sources': [
            {
                'path': 'TTE/consolidated_data',
                'prefix': 'consolidated_',
                'supported_tasks': ['TTE', 'Anomaly Detection', 'Mobility Prediction'],
                'priority': 1  # Higher priority, displayed first
            },
            {
                'path': 'Anomaly Detection/consolidated_data',
                'prefix': 'consolidated_anomaly_',
                'supported_tasks': ['Anomaly Detection'],
                'priority': 2  # Lower priority, displayed later
            }
        ]
    },
    'Porto': {
        'base_path': './Porto',
        'trajectory_sources': [
            {
                'path': 'TTE/consolidated_data',
                'prefix': 'consolidated_',
                'supported_tasks': ['TTE', 'Anomaly Detection', 'Mobility Prediction'],
                'priority': 1
            },
            {
                'path': 'Anomaly Detection/consolidated_data',
                'prefix': 'consolidated_anomaly_',
                'supported_tasks': ['Anomaly Detection'],
                'priority': 2
            }
        ]
    },
    'Xian': {
        'base_path': './Xian',
        'trajectory_sources': [
            {
                'path': 'TTE/consolidated_data',
                'prefix': 'consolidated_',
                'supported_tasks': ['TTE', 'Anomaly Detection', 'Mobility Prediction'],
                'priority': 1
            },
            {
                'path': 'Anomaly Detection/consolidated_data',
                'prefix': 'consolidated_anomaly_',
                'supported_tasks': ['Anomaly Detection'],
                'priority': 2
            }
        ]
    },
    'Beijing': {
        'base_path': './Beijing',
        'trajectory_sources': [
            {
                'path': '',  # Direct under Beijing folder
                'prefix': '',  # No prefix for Beijing trajectories
                'supported_tasks': ['TMI'],
                'priority': 1
            }
        ]
    }
}

# Task configuration
TASK_CONFIG = {
    'TTE': {
        'system_prompt_paths': {
            'Chengdu': 'Chengdu/TTE/consolidated_data/system_prompt.txt',
            'Porto': 'Porto/TTE/consolidated_data/system_prompt.txt',
            'Xian': 'Xian/TTE/consolidated_data/system_prompt.txt'
        },
        'user_prompt_file': 'dynamically_generated_stats_prompt.txt',
        'output_file': 'o4-mini.txt'
    },
    'Anomaly Detection': {
        'system_prompt_paths': {
            'Chengdu': 'Chengdu/Anomaly Detection/consolidated_data/system_prompt_ad.txt',
            'Porto': 'Porto/Anomaly Detection/consolidated_data/system_prompt_ad.txt',
            'Xian': 'Xian/Anomaly Detection/consolidated_data/system_prompt_ad.txt'
        },
        'user_prompt_file': None,
        'output_file': 'ad-o4-mini.txt'  # TTE part uses ad-o4-mini.txt for anomaly detection
    },
    'Mobility Prediction': {
        'system_prompt_paths': {
            'Chengdu': 'Chengdu/TTE/consolidated_data/system_prompt_mp.txt',
            'Porto': 'Porto/TTE/consolidated_data/system_prompt_mp.txt',
            'Xian': 'Xian/TTE/consolidated_data/system_prompt_mp.txt'
        },
        'user_prompt_file': 'dynamically_generated_stats_prompt_MP.txt',
        'output_file': 'mp-o4-mini.txt'
    },
    'TMI': {
        'system_prompt_paths': {
            'Beijing': 'Beijing/system_prompt.txt'
        },
        'user_prompt_file': None,
        'output_file': 'o4-mini-v1.txt'
    }
}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/datasets')
def get_datasets():
    """Get all datasets"""
    return jsonify(list(DATASET_CONFIG.keys()))


@app.route('/api/trajectories')
def get_trajectories():
    """Get all trajectories for specified dataset"""
    dataset = request.args.get('dataset')

    if not dataset or dataset not in DATASET_CONFIG:
        return jsonify({'error': 'Invalid dataset'}), 400

    try:
        trajectories = []
        dataset_config = DATASET_CONFIG[dataset]

        # Sort trajectory sources by priority
        sorted_sources = sorted(dataset_config['trajectory_sources'], key=lambda x: x['priority'])

        for source in sorted_sources:
            if dataset == 'Beijing':
                # Special handling for Beijing dataset
                base_path = dataset_config['base_path']
                if os.path.exists(base_path):
                    for item in os.listdir(base_path):
                        item_path = os.path.join(base_path, item)
                        if os.path.isdir(item_path) and not item.startswith('.'):
                            # Beijing trajectories are direct folders
                            trajectories.append({
                                'id': item,
                                'full_name': item,
                                'source_path': source['path'],
                                'supported_tasks': source['supported_tasks']
                            })
            else:
                # Standard handling for other datasets
                source_path = os.path.join(dataset_config['base_path'], source['path'])
                if os.path.exists(source_path):
                    for item in os.listdir(source_path):
                        if item.startswith(source['prefix']) and os.path.isdir(os.path.join(source_path, item)):
                            trajectory_id = item.replace(source['prefix'], '')
                            trajectories.append({
                                'id': trajectory_id,
                                'full_name': item,
                                'source_path': source['path'],
                                'supported_tasks': source['supported_tasks']
                            })

        return jsonify(trajectories)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/tasks')
def get_tasks():
    """Get tasks supported by specified trajectory"""
    dataset = request.args.get('dataset')
    trajectory_id = request.args.get('trajectory_id')

    if not dataset or not trajectory_id:
        return jsonify({'error': 'Missing parameters'}), 400

    try:
        # Find trajectory source and supported tasks
        dataset_config = DATASET_CONFIG[dataset]
        for source in dataset_config['trajectory_sources']:
            if dataset == 'Beijing':
                # Beijing trajectories are direct folders
                trajectory_path = os.path.join(dataset_config['base_path'], trajectory_id)
            else:
                source_path = os.path.join(dataset_config['base_path'], source['path'])
                trajectory_full_name = source['prefix'] + trajectory_id
                trajectory_path = os.path.join(source_path, trajectory_full_name)

            if os.path.exists(trajectory_path):
                return jsonify(source['supported_tasks'])

        return jsonify({'error': 'Trajectory not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/trajectory-data')
def get_trajectory_data():
    """Get complete trajectory data"""
    dataset = request.args.get('dataset')
    trajectory_id = request.args.get('trajectory_id')
    task = request.args.get('task')
    contextual_type = request.args.get('contextual_type', 'POI')

    if not all([dataset, trajectory_id, task]):
        return jsonify({'error': 'Missing parameters'}), 400

    try:
        # Find trajectory path
        dataset_config = DATASET_CONFIG[dataset]
        trajectory_path = None
        source_info = None

        for source in dataset_config['trajectory_sources']:
            if dataset == 'Beijing':
                # Beijing trajectories are direct folders
                full_trajectory_path = os.path.join(dataset_config['base_path'], trajectory_id)
            else:
                source_path = os.path.join(dataset_config['base_path'], source['path'])
                trajectory_full_name = source['prefix'] + trajectory_id
                full_trajectory_path = os.path.join(source_path, trajectory_full_name)

            if os.path.exists(full_trajectory_path) and task in source['supported_tasks']:
                trajectory_path = full_trajectory_path
                source_info = source
                break

        if not trajectory_path:
            return jsonify({'error': 'Trajectory not found or task not supported'}), 404

        data = {}

        if contextual_type == 'POI':
            if dataset == 'Beijing':
                # Beijing has simple naming: poi.png, 0_segment_poi.png, 1_segment_poi.png, etc.
                global_path = os.path.join(trajectory_path, 'poi.png')
                data['global_images'] = ['poi.png'] if os.path.exists(global_path) else []

                # Beijing segment files: {segment_num}_segment_poi.png
                poi_segment_files = glob.glob(os.path.join(trajectory_path, "*_segment_poi.png"))
            else:
                # Other datasets: poi_{trajectory_id}.png, {segment_num}_segment_{trajectory_id}_poi.png
                global_pattern = f"poi_{trajectory_id}.png"
                global_path = os.path.join(trajectory_path, global_pattern)
                data['global_images'] = [global_pattern] if os.path.exists(global_path) else []

                poi_segment_files = glob.glob(os.path.join(trajectory_path, f"*_segment_{trajectory_id}_poi.png"))

            segment_data = []
            for img_path in sorted(poi_segment_files):
                basename = os.path.basename(img_path)
                if dataset == 'Beijing':
                    # Beijing format: 0_segment_poi.png
                    match = re.match(r'(\d+)_segment_poi\.png', basename)
                else:
                    # Other datasets format: 0_segment_{trajectory_id}_poi.png
                    match = re.match(r'(\d+)_segment_.*_poi\.png', basename)

                if match:
                    segment_num = int(match.group(1))
                    segment_data.append({
                        'filename': basename,
                        'segment_num': segment_num,
                        'label': f'segment {segment_num}'
                    })

            data['segment_images'] = sorted(segment_data, key=lambda x: x['segment_num'])

        elif contextual_type == 'Road network':
            if dataset == 'Beijing':
                # Beijing has simple naming: road_structure.png, 0_segment_road_structure.png, etc.
                global_path = os.path.join(trajectory_path, 'road_structure.png')
                data['global_images'] = ['road_structure.png'] if os.path.exists(global_path) else []

                road_segment_files = glob.glob(os.path.join(trajectory_path, "*_segment_road_structure.png"))
            else:
                # Other datasets
                global_pattern = f"road_structure_{trajectory_id}.png"
                global_path = os.path.join(trajectory_path, global_pattern)
                data['global_images'] = [global_pattern] if os.path.exists(global_path) else []

                road_segment_files = glob.glob(
                    os.path.join(trajectory_path, f"*_segment_{trajectory_id}_road_structure.png"))

            segment_data = []
            for img_path in sorted(road_segment_files):
                basename = os.path.basename(img_path)
                if dataset == 'Beijing':
                    match = re.match(r'(\d+)_segment_road_structure\.png', basename)
                else:
                    match = re.match(r'(\d+)_segment_.*_road_structure\.png', basename)

                if match:
                    segment_num = int(match.group(1))
                    segment_data.append({
                        'filename': basename,
                        'segment_num': segment_num,
                        'label': f'segment {segment_num}'
                    })

            data['segment_images'] = sorted(segment_data, key=lambda x: x['segment_num'])

            # last_trajectory images (only for non-Beijing datasets)
            if dataset != 'Beijing':
                last_traj_files = glob.glob(os.path.join(trajectory_path, f"last_trajectory_{trajectory_id}_*.png"))

                last_traj_data = []
                for img_path in sorted(last_traj_files):
                    basename = os.path.basename(img_path)
                    match = re.match(r'last_trajectory_.*_(\d+)\.png', basename)
                    if match:
                        number = int(match.group(1))
                        # Determine size based on number - need to sort and assign labels sequentially
                        if number <= 17:  # 假设17是small
                            label = 'last trajectory small'
                        elif number <= 18:  # 假设18是medium
                            label = 'last trajectory medium'
                        else:  # 20是large
                            label = 'last trajectory large'

                        last_traj_data.append({
                            'filename': basename,
                            'number': number,
                            'label': label
                        })

                data['last_trajectory_images'] = sorted(last_traj_data, key=lambda x: x['number'])

        return jsonify(data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/prompts')
def get_prompts():
    """Get prompts for specified task"""
    dataset = request.args.get('dataset')
    trajectory_id = request.args.get('trajectory_id')
    task = request.args.get('task')

    if not all([dataset, trajectory_id, task]):
        return jsonify({'error': 'Missing parameters'}), 400

    try:
        prompts = {}

        # Get system prompt
        task_config = TASK_CONFIG.get(task)
        if task_config and task_config['system_prompt_paths']:
            system_prompt_paths = task_config['system_prompt_paths']
            if dataset in system_prompt_paths:
                system_prompt_path = system_prompt_paths[dataset]
                if os.path.exists(system_prompt_path):
                    with open(system_prompt_path, 'r', encoding='utf-8') as f:
                        prompts['system_prompt'] = f.read()

        # Get user prompt (only for tasks that have user prompts)
        if task_config and task_config['user_prompt_file']:
            # Find trajectory path
            dataset_config = DATASET_CONFIG[dataset]
            for source in dataset_config['trajectory_sources']:
                if dataset == 'Beijing':
                    trajectory_path = os.path.join(dataset_config['base_path'], trajectory_id)
                else:
                    source_path = os.path.join(dataset_config['base_path'], source['path'])
                    trajectory_full_name = source['prefix'] + trajectory_id
                    trajectory_path = os.path.join(source_path, trajectory_full_name)

                if os.path.exists(trajectory_path) and task in source['supported_tasks']:
                    user_prompt_path = os.path.join(trajectory_path, task_config['user_prompt_file'])
                    if os.path.exists(user_prompt_path):
                        with open(user_prompt_path, 'r', encoding='utf-8') as f:
                            prompts['user_prompt'] = f.read()
                    break

        return jsonify(prompts)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/output')
def get_output():
    """Get LLM output results"""
    dataset = request.args.get('dataset')
    trajectory_id = request.args.get('trajectory_id')
    task = request.args.get('task')

    if not all([dataset, trajectory_id, task]):
        return jsonify({'error': 'Missing parameters'}), 400

    try:
        # Find trajectory path
        dataset_config = DATASET_CONFIG[dataset]
        trajectory_path = None
        source_info = None

        for source in dataset_config['trajectory_sources']:
            if dataset == 'Beijing':
                full_trajectory_path = os.path.join(dataset_config['base_path'], trajectory_id)
            else:
                source_path = os.path.join(dataset_config['base_path'], source['path'])
                trajectory_full_name = source['prefix'] + trajectory_id
                full_trajectory_path = os.path.join(source_path, trajectory_full_name)

            if os.path.exists(full_trajectory_path) and task in source['supported_tasks']:
                trajectory_path = full_trajectory_path
                source_info = source
                break

        if not trajectory_path:
            return jsonify({'error': 'Trajectory not found'}), 404

        # Determine output filename
        task_config = TASK_CONFIG.get(task)
        if task == 'Anomaly Detection' and source_info['path'] == 'TTE/consolidated_data':
            # TTE part anomaly detection uses different output file
            output_file = 'ad-o4-mini.txt'
        else:
            output_file = task_config['output_file'] if task_config else 'ad-o4-mini.txt'

        output_path = os.path.join(trajectory_path, output_file)
        if os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8') as f:
                return jsonify({'output': f.read()})
        else:
            return jsonify({'output': 'Output file not found'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/image/<path:image_path>')
def get_image(image_path):
    """Get image files"""
    try:
        # Parse path parameters
        parts = image_path.split('/')
        if len(parts) < 4:
            return jsonify({'error': 'Invalid image path'}), 400

        dataset = parts[0]
        trajectory_id = parts[1]
        task = parts[2]
        image_name = parts[3]

        # Find trajectory path
        dataset_config = DATASET_CONFIG[dataset]
        for source in dataset_config['trajectory_sources']:
            if dataset == 'Beijing':
                trajectory_path = os.path.join(dataset_config['base_path'], trajectory_id)
            else:
                source_path = os.path.join(dataset_config['base_path'], source['path'])
                trajectory_full_name = source['prefix'] + trajectory_id
                trajectory_path = os.path.join(source_path, trajectory_full_name)

            if os.path.exists(trajectory_path) and task in source['supported_tasks']:
                image_full_path = os.path.join(trajectory_path, image_name)
                if os.path.exists(image_full_path):
                    return send_file(image_full_path)

        return jsonify({'error': 'Image not found'}), 404

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)