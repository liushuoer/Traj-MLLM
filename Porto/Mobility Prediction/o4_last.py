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
BASE_DATA_DIR = "../TTE/consolidated_data"
OUTPUT_FILENAME = "mp-o4-mini.txt"

# System prompt
SYSTEM_PROMPT_FILENAME = "system_prompt_mp.txt"

# Load system prompt
system_prompt_path = os.path.join(BASE_DATA_DIR, SYSTEM_PROMPT_FILENAME)
with open(system_prompt_path, 'r', encoding='utf-8') as f:
    SYSTEM_PROMPT = f.read()

# Fully utilize multi-core CPU and high-performance hardware
CPU_COUNT = multiprocessing.cpu_count()
MAX_WORKERS = max(4, CPU_COUNT)
IMAGE_ENCODING_WORKERS = max(32, CPU_COUNT * 2)
MAX_API_CONCURRENCY = 16

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


def image_to_base64(file_path):
    """Convert image to Base64 encoding"""
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


def process_single_trajectory(traj_id: str, base_dir: str):
    """Process single trajectory folder"""
    traj_folder_path = os.path.join(base_dir, traj_id)
    output_path = os.path.join(traj_folder_path, OUTPUT_FILENAME)

    # Skip processing if output file already exists
    # if os.path.exists(output_path):
    #     print(f"[Info] Output file for trajectory {traj_id} already exists, skipping processing.")
    #     return traj_id

    try:
        # 1. Get all PNG images
        png_files = glob.glob(os.path.join(traj_folder_path, "*.png"))
        png_files = [img for img in png_files if os.path.basename(img).startswith('last_trajectory_')]

        if not png_files:
            print(f"[Warning] {traj_id}: No PNG images found. Skipping.")
            return None

        # 2. Get all TXT files
        txt_files = glob.glob(os.path.join(traj_folder_path, "dynamically_generated_stats_prompt_MP.txt"))
        txt_content = ""
        for txt_file in txt_files:
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    txt_content += f.read() + "\n\n"
            except Exception as e:
                print(f"[Warning] Failed to read text file {txt_file}: {e}")

        # 3. Process all images in parallel
        image_content = process_images_parallel(png_files, traj_id)

        # 4. Build complete content
        content = []
        # First add text description
        if txt_content.strip():
            content.append({"type": "text", "text": txt_content})
        # Add all images
        content.extend(image_content)

        # Check if content was successfully added
        if not content:
            print(f"[Warning] {traj_id}: No valid content to send. Skipping.")
            return None

        # 5. Call model
        try:
            # Use async API call
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            full_response = loop.run_until_complete(call_api_async(content, SYSTEM_PROMPT))
            loop.close()

            # Write final concatenated result to file
            with open(output_path, 'w', encoding='utf-8') as out_f:
                out_f.write(full_response)
            return traj_id

        except Exception as api_error:
            print(f"[Error] {traj_id}: API call failed: {api_error}")
            with open(os.path.join(traj_folder_path, f"{traj_id}_api_error.txt"), 'w', encoding='utf-8') as err_f:
                err_f.write(f"API call failed:\n{str(api_error)}")
            return None

    except Exception as e:
        print(f"Unknown error occurred while processing {traj_id}: {e}")
        error_output_path = os.path.join(traj_folder_path, f"{traj_id}_general_error.txt")
        with open(error_output_path, 'w', encoding='utf-8') as err_f:
            err_f.write(f"Unknown error occurred during processing:\n{e}")
        return None


def main():
    # Pre-populate client connection pool
    for _ in range(MAX_CLIENT_POOL_SIZE):
        client_pool.put(OpenAIClientManager.create_client())

    # Get all subfolders in BASE_DATA_DIR as trajectory IDs
    traj_folders = [folder for folder in os.listdir(BASE_DATA_DIR)
                    if os.path.isdir(os.path.join(BASE_DATA_DIR, folder))]

    if not traj_folders:
        print(f"No trajectory folders found in '{BASE_DATA_DIR}'.")
        return

    print(f"Found {len(traj_folders)} trajectory folders to process: {traj_folders[:5]}... (showing up to first 5)")

    print(f"Performance optimization configuration:")
    print(f"- CPU cores: {CPU_COUNT}")
    print(f"- Trajectory processing concurrent threads: {MAX_WORKERS}")
    print(f"- Image encoding concurrent threads: {IMAGE_ENCODING_WORKERS}")
    print(f"- API concurrency: {MAX_API_CONCURRENCY * 2}")
    print(f"- Client connection pool size: {MAX_CLIENT_POOL_SIZE}")

    # Test API connection
    try:
        client = OpenAIClientManager.get_client()
        test_completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Test connection"}],
            stream=True
        )
        for _ in test_completion:
            pass
        OpenAIClientManager.release_client(client)
        print("API connection test successful, starting high-performance parallel trajectory processing...")
    except Exception as e:
        print(f"API connection test failed: {e}")
        print("Please check API key and network connection before retrying.")
        return

    # Create progress bar
    progress_bar = tqdm(total=len(traj_folders), desc="Overall processing progress", ncols=100)
    processed_count = 0
    success_count = 0
    error_count = 0

    # Record start time
    start_time = time.time()

    # Use thread pool for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Create processing task for each trajectory folder
        futures = {executor.submit(process_single_trajectory, traj_id, BASE_DATA_DIR): traj_id
                   for traj_id in traj_folders}

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
    print(f"- Total processed trajectories: {processed_count}/{len(traj_folders)}")
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