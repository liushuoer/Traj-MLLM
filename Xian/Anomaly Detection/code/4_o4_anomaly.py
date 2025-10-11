import os
import time
import glob
import base64
import concurrent.futures
from tqdm import tqdm
from openai import OpenAI
import threading
import multiprocessing
import queue
import asyncio
from openai.types.chat import ChatCompletionChunk

# --- Configuration ---
API_KEY = "your-api-key-here"  # Replace with your actual key
MODEL_NAME = "o4-mini"
BASE_DATA_DIR = "../../TTE/consolidated_data"
ANOMALY_DIR = ".../../TTE/TTE/consolidated_data"
NORMAL_DIR = "../../TTE/TTE/consolidated_data"
SYSTEM_PROMPT_FILENAME = "system_prompt_ad.txt"
OUTPUT_FILENAME = "o4-mini.txt"

# Fully utilize multi-core CPU and high-performance hardware
CPU_COUNT = multiprocessing.cpu_count()
# Trajectory processing thread count - use higher concurrency
MAX_WORKERS = max(32, CPU_COUNT)
# Image encoding dedicated process pool - assign multiple work units per CPU core
IMAGE_ENCODING_WORKERS = max(32, CPU_COUNT * 2)
# Maximum API concurrent requests per trajectory processing thread
MAX_API_CONCURRENCY = 32
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
            timeout=12000.0  # Increase timeout
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


def process_images_parallel(image_paths, traj_id):
    """Parallel processing of multiple image encoding tasks"""
    results = []
    # Use thread pool to process all images in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=IMAGE_ENCODING_WORKERS) as executor:
        # Submit all image processing tasks
        future_to_path = {executor.submit(process_image_worker, path): path for path in image_paths}
        # Collect results
        for future in concurrent.futures.as_completed(future_to_path):
            path = future_to_path[future]
            try:
                b64_result = future.result()
                if b64_result:
                    results.append({"type": "image_url", "image_url": {"url": b64_result}})
            except Exception as e:
                print(f"[Error] Trajectory {traj_id} image {path} processing failed: {e}")
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


def process_single_trajectory(traj_info):
    """Process single trajectory, traj_info is a dict containing traj_id, folder_path, traj_type"""
    traj_id = traj_info['traj_id']
    folder_path = traj_info['folder_path']
    traj_type = traj_info['traj_type']  # 'normal' or 'anomaly'

    if not os.path.isdir(folder_path):
        print(f"[Warning] Trajectory folder not found: {folder_path}. Skipping {traj_id} ({traj_type}).")
        return

    output_path = os.path.join(folder_path, OUTPUT_FILENAME)
    if os.path.exists(output_path):
        return

    try:
        # 1. Collect all images
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            image_files.extend(glob.glob(os.path.join(folder_path, ext)))

        if not image_files:
            print(f"[Warning] {traj_id} ({traj_type}): No images found. Skipping.")
            return

        # 2. Process all images
        image_content = process_images_parallel(image_files, traj_id)
        if not image_content:
            print(f"[Warning] {traj_id} ({traj_type}): All image processing failed. Skipping.")
            return

        # 3. Construct message content - text first, then all images
        content = [{"type": "text", "text": f"These are all images for trajectory {traj_id}:"}]
        content.extend(image_content)

        # 4. Call model
        try:
            # Use async API call
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            full_response = loop.run_until_complete(call_api_async(content, system_prompt_content))
            loop.close()

            # Write final concatenated result to file
            with open(output_path, 'w', encoding='utf-8') as out_f:
                out_f.write(full_response)
            return traj_id

        except Exception as api_error:
            print(f"[Error] {traj_id} ({traj_type}): API call failed: {api_error}")
            with open(os.path.join(folder_path, f"{traj_id}_api_error.txt"), 'w', encoding='utf-8') as err_f:
                err_f.write(f"API call failed:\n{str(api_error)}")

    except Exception as e:
        print(f"Unknown error occurred while processing {traj_id} ({traj_type}): {e}")
        error_output_path = os.path.join(folder_path, f"{traj_id}_general_error.txt")
        with open(error_output_path, 'w', encoding='utf-8') as err_f:
            err_f.write(f"Unknown error occurred during processing:\n{e}")

    return None


def collect_trajectory_info():
    """Collect all trajectory information that needs processing"""
    all_trajectories = []

    # Collect anomaly trajectories
    anomaly_folders = glob.glob(os.path.join(ANOMALY_DIR, "consolidated_anomaly_*"))
    for folder in anomaly_folders:
        if os.path.isdir(folder):
            folder_name = os.path.basename(folder)
            if folder_name.startswith("consolidated_anomaly_"):
                traj_id = folder_name[len("consolidated_anomaly_"):]
                all_trajectories.append({
                    'traj_id': traj_id,
                    'folder_path': folder,
                    'traj_type': 'anomaly'
                })

    # Collect normal trajectories
    normal_folders = glob.glob(os.path.join(NORMAL_DIR, "consolidated_normal_*"))
    for folder in normal_folders:
        if os.path.isdir(folder):
            folder_name = os.path.basename(folder)
            if folder_name.startswith("consolidated_normal_"):
                traj_id = folder_name[len("consolidated_normal_"):]
                all_trajectories.append({
                    'traj_id': traj_id,
                    'folder_path': folder,
                    'traj_type': 'normal'
                })

    return all_trajectories


def main():
    # Pre-populate client connection pool
    for _ in range(MAX_CLIENT_POOL_SIZE):
        client_pool.put(OpenAIClientManager.create_client())

    # Collect all trajectory information
    all_trajectories = collect_trajectory_info()
    if not all_trajectories:
        print(f"No trajectory folders found in directories matching the format.")
        print(f"Please check if paths are correct:")
        print(f"- Anomaly trajectory path: {ANOMALY_DIR}")
        print(f"- Normal trajectory path: {NORMAL_DIR}")
        return

    # Count found trajectories
    anomaly_count = sum(1 for t in all_trajectories if t['traj_type'] == 'anomaly')
    normal_count = sum(1 for t in all_trajectories if t['traj_type'] == 'normal')

    print(f"Found {len(all_trajectories)} trajectories to process:")
    print(f"- Anomaly trajectories: {anomaly_count}")
    print(f"- Normal trajectories: {normal_count}")
    print(f"Example: {[t['traj_id'] for t in all_trajectories[:3]]}... (showing first 3 only)")

    print(f"Performance optimization configuration:")
    print(f"- CPU cores: {CPU_COUNT}")
    print(f"- Trajectory processing concurrent threads: {MAX_WORKERS}")
    print(f"- Image encoding concurrent threads: {IMAGE_ENCODING_WORKERS}")
    print(f"- API concurrency: {MAX_API_CONCURRENCY * 2}")
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
    progress_bar = tqdm(total=len(all_trajectories), desc="Overall processing progress", ncols=100)
    processed_count = 0
    success_count = 0
    error_count = 0

    # Record start time
    start_time = time.time()

    # Use thread pool and submit tasks
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Create processing task for each trajectory
        futures = {executor.submit(process_single_trajectory, traj_info): traj_info for traj_info in all_trajectories}

        # Process completed tasks
        for future in concurrent.futures.as_completed(futures):
            traj_info = futures[future]
            traj_id = traj_info['traj_id']
            traj_type = traj_info['traj_type']

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
                print(f"[Error] Trajectory {traj_id} ({traj_type}) processing exception: {exc}")
                progress_bar.update(1)

    progress_bar.close()

    # Calculate total time and processing speed
    total_time = time.time() - start_time
    average_time_per_item = total_time / processed_count if processed_count > 0 else 0

    print(f"\nProcessing completion statistics:")
    print(f"- Total processed trajectories: {processed_count}/{len(all_trajectories)}")
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