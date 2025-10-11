import os
import time
import glob
import pickle
import json
from openai.types.chat import ChatCompletionChunk
import base64
import concurrent.futures
from tqdm import tqdm
from openai import OpenAI
import threading
import multiprocessing
from functools import partial
import queue
import asyncio
import httpx
import re

# --- Configuration ---
API_KEY = "your-api-key-here"  # Replace with your actual key
MODEL_NAME = "gpt-4o-mini"
BASE_DATA_DIR = "../consolidated_data"
SYSTEM_PROMPT_FILENAME = "system_prompt.txt"
IMAGE_ORDER_FILENAME = "file_list.pkl"
DYNAMIC_PROMPT_FILENAME = "dynamically_generated_stats_prompt.txt"
OUTPUT_FILENAME = "gpt-4o-mini.txt"

# Fully utilize multi-core CPU and high-performance hardware
CPU_COUNT = multiprocessing.cpu_count()
# Trajectory processing thread count - use higher concurrency
MAX_WORKERS = 32
# Image encoding dedicated process pool - assign multiple work units per CPU core
IMAGE_ENCODING_WORKERS = 32
# Maximum API concurrent requests per trajectory processing thread
MAX_API_CONCURRENCY = 32
# Image batch processing size
# Thread-safe queue for limiting global API concurrency
api_semaphore = threading.Semaphore(MAX_API_CONCURRENCY * 2)
# Use thread lock to prevent concurrency issues
api_client_lock = threading.Lock()
# Client connection pool for connection reuse
client_pool = queue.Queue()
MAX_CLIENT_POOL_SIZE = MAX_WORKERS

# Global cache to avoid duplicate processing
processed_images_cache = {}
processed_images_lock = threading.Lock()

# Load system prompt
system_prompt_path = os.path.join(BASE_DATA_DIR, SYSTEM_PROMPT_FILENAME)
with open(system_prompt_path, 'r', encoding='utf-8') as f:
    system_prompt_content = f.read()


def image_to_base64(file_path):
    """Convert image to Base64 encoding without compression"""
    # First check cache
    global processed_images_cache
    with processed_images_lock:
        if file_path in processed_images_cache:
            return processed_images_cache[file_path]

    try:
        with open(file_path, "rb") as img_file:
            encoded_str = base64.b64encode(img_file.read()).decode("utf-8")
        result = f"data:image/png;base64,{encoded_str}"
        # Store in cache
        with processed_images_lock:
            processed_images_cache[file_path] = result
        return result
    except Exception as e:
        print(f"[Warning] Image reading failed: {file_path} - {e}")
        return None


# Implement singleton pattern API client management with connection pool
class OpenAIClientManager:
    @staticmethod
    def create_client():
        """Create new OpenAI client"""
        return OpenAI(
            api_key=API_KEY,
            base_url="https://api.openai.com/v1",
            default_headers={"X-Custom-Header": "enable"},
            timeout=1200.0  # Increase timeout
        )

    @staticmethod
    def get_client():
        """Get client from connection pool, create new client if pool is empty"""
        try:
            return client_pool.get_nowait()
        except queue.Empty:
            return OpenAIClientManager.create_client()

    @staticmethod
    def release_client(client):
        """Return client to connection pool"""
        if client_pool.qsize() < MAX_CLIENT_POOL_SIZE:
            client_pool.put(client)


def process_image_worker(image_path):
    """Single image processing worker function"""
    return image_to_base64(image_path)


def process_images_parallel(image_paths_dict, traj_id):
    """Parallel processing of multiple image encoding tasks"""
    image_paths = [(key, path) for key, path in image_paths_dict.items()]
    results = {}

    # Use thread pool to process all images in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=IMAGE_ENCODING_WORKERS) as executor:
        # Submit all image processing tasks
        future_to_key = {executor.submit(process_image_worker, path): key for key, path in image_paths}
        # Collect results
        for future in concurrent.futures.as_completed(future_to_key):
            key = future_to_key[future]
            try:
                b64_result = future.result()
                if b64_result:
                    results[key] = b64_result
            except Exception as e:
                print(f"[Error] Trajectory {traj_id} image {key} processing failed: {e}")
    return results


async def call_api_async(content, system_prompt):
    """Async API call"""
    with api_semaphore:
        client = OpenAIClientManager.get_client()
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content}
                ],
                stream=True
            )
            # Collect response
            full_response = ""
            for chunk in completion:
                if isinstance(chunk, ChatCompletionChunk):
                    if chunk.choices and chunk.choices[0].delta.content:
                        content_chunk = chunk.choices[0].delta.content
                        full_response += content_chunk
            return full_response
        except Exception as e:
            raise e
        finally:
            OpenAIClientManager.release_client(client)


def process_single_trajectory(traj_id: str, base_dir: str):
    traj_folder_name = f"consolidated_{traj_id}"
    traj_folder_path = os.path.join(base_dir, traj_folder_name)

    if not os.path.isdir(traj_folder_path):
        print(f"[Warning] Trajectory folder not found: {traj_folder_path}. Skipping {traj_id}.")
        return

    image_order_path = os.path.join(traj_folder_path, IMAGE_ORDER_FILENAME)
    dynamic_prompt_path = os.path.join(traj_folder_path, DYNAMIC_PROMPT_FILENAME)
    output_path = os.path.join(traj_folder_path, OUTPUT_FILENAME)

    if os.path.exists(output_path):
        return

    if not os.path.exists(image_order_path) or not os.path.exists(dynamic_prompt_path):
        print(f"[Warning] Required files missing, skipping {traj_id}.")
        return

    try:
        # 1. Load image list
        with open(image_order_path, 'rb') as f:
            image_filenames = pickle.load(f)

        if not image_filenames:
            print(f"[Warning] {traj_id}: Image list is empty. Skipping.")
            return

        # 2. Load text prompt
        with open(dynamic_prompt_path, 'r', encoding='utf-8') as f:
            dynamic_text_content = f.read()

        # 3. Prepare image dictionary and description text
        image_dict = {}
        image_descriptions = {}
        image_paths_dict = {}

        # Regular images: image1, image2, ..., image10
        for i, img_filename in enumerate(image_filenames[:10], 1):
            segment_number = (i + 1) // 2  # Every two images belong to one segment
            is_poi = (i % 2) == 1  # Odd images are POI, even images are road structure
            image_dict[f'image{i}'] = img_filename
            if is_poi:
                image_descriptions[f'image{i}'] = f"**Segment {segment_number} POI & Route Detail Map:**"
            else:
                image_descriptions[f'image{i}'] = f"**Segment {segment_number} Road Network Map:**"

        # Special images: last_trajectory_view
        last_trajectory_images = [f for f in image_filenames if f.startswith('last_trajectory')]
        if last_trajectory_images:
            image_dict['last_trajectory_view'] = last_trajectory_images[0]
            image_descriptions['last_trajectory_view'] = "**End-Point Trajectory View:**"

        # Special images: poi_full_image
        poi_images = [f for f in image_filenames if f.startswith('poi_')]
        if poi_images:
            image_dict['poi_full_image'] = poi_images[0]
            image_descriptions['poi_full_image'] = "**Global POI & Route Detail Map:**"

        # Special images: road_structure_full_image
        road_structure_images = [f for f in image_filenames if f.startswith('road_structure_')]
        if road_structure_images:
            image_dict['road_structure_full_image'] = road_structure_images[0]
            image_descriptions['road_structure_full_image'] = "**Global Road Network Map:**"

        # Build complete image path dictionary
        for key, img_filename in image_dict.items():
            img_path = os.path.join(traj_folder_path, img_filename)
            if os.path.exists(img_path):
                image_paths_dict[key] = img_path
            else:
                print(f"[Warning] {traj_id}: Image file not found: {img_path}")

        # High-performance parallel processing of all images
        encoded_images = process_images_parallel(image_paths_dict, traj_id)

        # 4. Construct message content - interleaved text and images
        content = []

        # Split text, find image placeholders
        segments = re.split(r'(`[^`]+`)', dynamic_text_content)

        for segment in segments:
            # Check if it's an image placeholder
            placeholder_match = re.match(r'`([^`]+)`', segment)
            if placeholder_match:
                placeholder = placeholder_match.group(1)
                if placeholder in image_dict:
                    # First add preceding text
                    if content and content[-1]["type"] == "text":
                        # Ensure not adding empty text
                        if content[-1]["text"].strip():
                            pass
                    # Add image description
                    description = image_descriptions.get(placeholder, f"**{placeholder}:**")
                    if content and content[-1]["type"] == "text":
                        content[-1]["text"] += f" {description} "
                    else:
                        content.append({"type": "text", "text": description + " "})
                    # Add image
                    if placeholder in encoded_images:
                        content.append({"type": "image_url", "image_url": {"url": encoded_images[placeholder]}})
                    else:
                        # If image encoding failed, keep original placeholder
                        content.append({"type": "text", "text": segment})
                else:
                    # Corresponding image not found, keep original placeholder
                    content.append({"type": "text", "text": segment})
            else:
                # Add regular text, but remove image placeholder parts as they will be replaced by descriptions and actual images
                modified_segment = re.sub(r'\*\s+`[^`]+`\s+\([^)]+\)', '', segment)
                if modified_segment.strip():
                    if content and content[-1]["type"] == "text":
                        content[-1]["text"] += modified_segment
                    else:
                        content.append({"type": "text", "text": modified_segment})

        # Check if content was successfully added
        if not content:
            print(f"[Warning] {traj_id}: No valid content to send. Skipping.")
            return

        # 5. Call model
        try:
            # Use async API call
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            full_response = loop.run_until_complete(call_api_async(content, system_prompt_content))
            loop.close()

            # Write final concatenated result to file
            with open(output_path, 'w', encoding='utf-8') as out_f:
                out_f.write(full_response)

        except Exception as api_error:
            print(f"[Error] {traj_id}: API call failed: {api_error}")
            with open(os.path.join(traj_folder_path, f"{traj_id}_api_error.txt"), 'w', encoding='utf-8') as err_f:
                err_f.write(f"API call failed:\n{str(api_error)}")

    except FileNotFoundError as fnf_error:
        print(f"[Error] File not found while processing {traj_id}: {fnf_error}. Skipping.")
    except pickle.UnpicklingError as pkl_error:
        print(f"[Error] Unable to load {IMAGE_ORDER_FILENAME} while processing {traj_id}: {pkl_error}. Skipping.")
    except Exception as e:
        print(f"Unknown error occurred while processing {traj_id}: {e}")
        error_output_path = os.path.join(traj_folder_path, f"{traj_id}_general_error.txt")
        with open(error_output_path, 'w', encoding='utf-8') as err_f:
            err_f.write(f"Unknown error occurred during processing:\n{e}")

    return traj_id


def main():
    # Pre-populate client connection pool
    for _ in range(MAX_CLIENT_POOL_SIZE):
        client_pool.put(OpenAIClientManager.create_client())

    # Get all consolidated_{traj_id} folders from BASE_DATA_DIR
    consolidated_folders = glob.glob(os.path.join(BASE_DATA_DIR, "consolidated_*"))

    if not consolidated_folders:
        print(f"No 'consolidated_*' format subfolders found in '{BASE_DATA_DIR}' folder.")
        print("Please check path and folder naming.")
        return

    traj_ids = []
    for folder_path in consolidated_folders:
        folder_name = os.path.basename(folder_path)
        if folder_name.startswith("consolidated_") and os.path.isdir(folder_path):
            traj_id = folder_name.replace("consolidated_", "", 1)
            traj_ids.append(traj_id)

    if not traj_ids:
        print(f"No valid trajectory IDs extracted from '{BASE_DATA_DIR}'.")
        return

    print(f"Found {len(traj_ids)} trajectory IDs to process: {traj_ids[:5]}... (showing up to first 5)")

    print(f"Performance optimization configuration:")
    print(f"- CPU cores: {CPU_COUNT}")
    print(f"- Trajectory processing concurrent threads: {MAX_WORKERS}")
    print(f"- Image encoding concurrent threads: {IMAGE_ENCODING_WORKERS}")
    print(f"- API concurrency: {MAX_API_CONCURRENCY}")
    print(f"- Client connection pool size: {MAX_CLIENT_POOL_SIZE}")

    # Test API connection
    try:
        # Get client for testing
        client = OpenAIClientManager.get_client()
        test_completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Test connection"}],
            stream=True
        )
        # Consume streaming content, otherwise may cause connection resource leak
        for _ in test_completion:
            pass
        # Return test client to pool
        OpenAIClientManager.release_client(client)
        print("API connection test successful, starting high-performance parallel trajectory processing...")
    except Exception as e:
        print(f"API connection test failed: {e}")
        print("Please check API key and network connection before retrying.")
        return

    # Create progress bar
    progress_bar = tqdm(total=len(traj_ids), desc="Overall processing progress", ncols=100)
    processed_count = 0
    success_count = 0
    error_count = 0

    # Record start time
    start_time = time.time()

    # Use thread pool and submit tasks
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Create processing task for each trajectory ID
        futures = {executor.submit(process_single_trajectory, traj_id, BASE_DATA_DIR): traj_id for traj_id in traj_ids}

        # Process completed tasks
        for future in concurrent.futures.as_completed(futures):
            traj_id = futures[future]
            try:
                result = future.result()
                processed_count += 1
                if result:
                    success_count += 1
                else:
                    error_count += 1

                # Update progress bar info
                elapsed_time = time.time() - start_time
                items_per_second = processed_count / elapsed_time if elapsed_time > 0 else 0
                progress_bar.set_postfix({
                    "Success": success_count,
                    "Failed": error_count,
                    "Rate": f"{items_per_second:.2f}items/sec"
                })
                progress_bar.update(1)
            except Exception as exc:
                error_count += 1
                processed_count += 1
                print(f"[Error] Trajectory {traj_id} processing exception: {exc}")
                progress_bar.update(1)

    progress_bar.close()

    # Calculate total time and processing speed
    total_time = time.time() - start_time
    average_time_per_item = total_time / processed_count if processed_count > 0 else 0

    print(f"\nProcessing completion statistics:")
    print(f"- Total processed trajectories: {processed_count}/{len(traj_ids)}")
    print(f"- Successfully processed: {success_count}")
    print(f"- Failed processing: {error_count}")
    print(f"- Total time: {total_time:.2f}seconds")
    print(f"- Average time per trajectory: {average_time_per_item:.2f}seconds")
    print(f"- Processing rate: {processed_count / total_time:.2f}trajectories/sec")


if __name__ == "__main__":
    # Initialize client connection pool
    for _ in range(MAX_CLIENT_POOL_SIZE):
        try:
            client_pool.put(OpenAIClientManager.create_client())
        except:
            pass

    # Run main program
    main()