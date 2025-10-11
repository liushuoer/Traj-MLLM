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

# --- 配置 ---
API_KEY = "sk-K6x8evTggP9V9p9Q15F25d45852f4c63824a74F7F5B5B672"  # 替换为您的实际密钥
MODEL_NAME = "o4-mini"
BASE_DATA_DIR = "../../TTE/consolidated_data"
ANOMALY_DIR = "../../TTE/consolidated_data"
NORMAL_DIR = "../../TTE/consolidated_data"
SYSTEM_PROMPT_FILENAME = "system_prompt_ad.txt"
OUTPUT_FILENAME = "ad-o4-mini.txt"

# 充分利用多核CPU和高性能硬件
CPU_COUNT = multiprocessing.cpu_count()
# 轨迹处理线程数 - 使用更高的并发
MAX_WORKERS = max(32, CPU_COUNT)
# 图片编码专用进程池 - 每个CPU核心分配多个工作单元
IMAGE_ENCODING_WORKERS = max(32, CPU_COUNT * 2)
# 每个轨迹处理线程最大API并发请求数
MAX_API_CONCURRENCY = 32

# 线程安全队列，用于限制全局API并发
api_semaphore = threading.Semaphore(MAX_API_CONCURRENCY * 2)
# 使用线程锁防止并发问题
api_client_lock = threading.Lock()

# 客户端连接池，用于复用连接
client_pool = queue.Queue()
MAX_CLIENT_POOL_SIZE = MAX_WORKERS

# 全局缓存，用于避免重复处理
processed_images_cache = {}
processed_images_lock = threading.Lock()

# 加载系统提示
system_prompt_path = os.path.join(BASE_DATA_DIR, SYSTEM_PROMPT_FILENAME)
with open(system_prompt_path, 'r', encoding='utf-8') as f:
    system_prompt_content = f.read()


def image_to_base64(file_path):
    """将图片转换为Base64编码，不压缩"""
    # 首先检查缓存
    global processed_images_cache
    with processed_images_lock:
        if file_path in processed_images_cache:
            return processed_images_cache[file_path]

    try:
        with open(file_path, "rb") as img_file:
            encoded_str = base64.b64encode(img_file.read()).decode("utf-8")
        result = f"data:image/png;base64,{encoded_str}"

        # 存入缓存
        with processed_images_lock:
            processed_images_cache[file_path] = result

        return result
    except Exception as e:
        print(f"[警告] 图片读取失败: {file_path} - {e}")
        return None


# 实现单例模式的API客户端管理，并增加连接池
class OpenAIClientManager:
    @staticmethod
    def create_client():
        """创建新的OpenAI客户端"""
        return OpenAI(
            api_key=API_KEY,
            base_url="https://www.apillm.online/v1",
            default_headers={"X-DashScope-OssResourceResolve": "enable"},
            timeout=12000.0  # 增加超时时间
        )

    @staticmethod
    def get_client():
        """从连接池获取客户端，如果池为空则创建新客户端"""
        try:
            return client_pool.get_nowait()
        except queue.Empty:
            return OpenAIClientManager.create_client()

    @staticmethod
    def release_client(client):
        """将客户端放回连接池"""
        if client_pool.qsize() < MAX_CLIENT_POOL_SIZE:
            client_pool.put(client)


def process_image_worker(image_path):
    """单个图片处理工作函数"""
    return image_to_base64(image_path)


def process_images_parallel(image_paths, traj_id):
    """并行处理多个图片的编码任务"""
    results = []

    # 使用线程池并行处理所有图片
    with concurrent.futures.ThreadPoolExecutor(max_workers=IMAGE_ENCODING_WORKERS) as executor:
        # 提交所有图片处理任务
        future_to_path = {executor.submit(process_image_worker, path): path for path in image_paths}

        # 收集结果
        for future in concurrent.futures.as_completed(future_to_path):
            path = future_to_path[future]
            try:
                b64_result = future.result()
                if b64_result:
                    results.append({"type": "image_url", "image_url": {"url": b64_result}})
            except Exception as e:
                print(f"[错误] 轨迹 {traj_id} 图片 {path} 处理失败: {e}")

    return results


async def call_api_async(content, system_prompt):
    """异步调用API"""
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

            # 收集响应
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
    """处理单个轨迹，traj_info是一个包含traj_id, folder_path, traj_type的字典"""
    traj_id = traj_info['traj_id']
    folder_path = traj_info['folder_path']
    traj_type = traj_info['traj_type']  # 'normal' 或 'anomaly'

    if not os.path.isdir(folder_path):
        print(f"[警告] 轨迹文件夹未找到: {folder_path}。跳过 {traj_id} ({traj_type})。")
        return

    output_path = os.path.join(folder_path, OUTPUT_FILENAME)

    # if os.path.exists(output_path):
    #     return

    try:
        # 1. 收集所有图片（排除以 last_trajectory_ 开头的图片）
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            image_files.extend(glob.glob(os.path.join(folder_path, ext)))
        # 过滤掉以 'last_trajectory_' 开头的图片
        image_files = [img for img in image_files if not os.path.basename(img).startswith('last_trajectory_')]

        if not image_files:
            print(f"[警告] {traj_id} ({traj_type}): 未找到任何图片。跳过。")
            return

        # 2. 处理所有图片
        image_content = process_images_parallel(image_files, traj_id)

        if not image_content:
            print(f"[警告] {traj_id} ({traj_type}): 所有图片处理失败。跳过。")
            return

        # 3. 构造消息内容 - 文本开头，然后是所有图片
        content = [{"type": "text", "text": f"这是轨迹 {traj_id} 的所有图片:"}]
        content.extend(image_content)

        # 4. 调用模型
        try:
            # 使用异步API调用
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            full_response = loop.run_until_complete(call_api_async(content, system_prompt_content))
            loop.close()

            # 将最终拼接的结果写入文件
            with open(output_path, 'w', encoding='utf-8') as out_f:
                out_f.write(full_response)

            return traj_id

        except Exception as api_error:
            print(f"[错误] {traj_id} ({traj_type}): API调用失败: {api_error}")
            with open(os.path.join(folder_path, f"{traj_id}_api_error.txt"), 'w', encoding='utf-8') as err_f:
                err_f.write(f"API调用失败:\n{str(api_error)}")

    except Exception as e:
        print(f"处理 {traj_id} ({traj_type}) 时发生未知错误: {e}")
        error_output_path = os.path.join(folder_path, f"{traj_id}_general_error.txt")
        with open(error_output_path, 'w', encoding='utf-8') as err_f:
            err_f.write(f"处理时发生未知错误:\n{e}")

    return None

def collect_trajectory_info():
    """收集所有需要处理的轨迹信息"""
    all_trajectories = []

    # 收集异常轨迹
    anomaly_folders = glob.glob(os.path.join(ANOMALY_DIR, "consolidated_*"))
    print(anomaly_folders)
    for folder in anomaly_folders:
        if os.path.isdir(folder):
            folder_name = os.path.basename(folder)
            traj_id = folder_name.replace("consolidated_", "", 1)
            all_trajectories.append({
                'traj_id': traj_id,
                'folder_path': folder,
                'traj_type': 'normal'
            })


    return all_trajectories


def main():
    # 预填充客户端连接池
    for _ in range(MAX_CLIENT_POOL_SIZE):
        client_pool.put(OpenAIClientManager.create_client())

    # 收集所有轨迹信息
    all_trajectories = collect_trajectory_info()

    if not all_trajectories:
        print(f"在目录中没有找到任何符合格式的轨迹文件夹。")
        print(f"请检查路径是否正确：")
        print(f"- 异常轨迹路径: {ANOMALY_DIR}")
        print(f"- 正常轨迹路径: {NORMAL_DIR}")
        return

    # 统计找到的轨迹数量
    anomaly_count = sum(1 for t in all_trajectories if t['traj_type'] == 'anomaly')
    normal_count = sum(1 for t in all_trajectories if t['traj_type'] == 'normal')

    print(f"找到 {len(all_trajectories)} 个轨迹需要处理:")
    print(f"- 异常轨迹: {anomaly_count}个")
    print(f"- 正常轨迹: {normal_count}个")
    print(f"示例: {[t['traj_id'] for t in all_trajectories[:3]]}... (仅显示前3个)")

    print(f"性能优化配置:")
    print(f"- CPU核心数: {CPU_COUNT}")
    print(f"- 轨迹处理并发线程: {MAX_WORKERS}")
    print(f"- 图片编码并发线程: {IMAGE_ENCODING_WORKERS}")
    print(f"- API并发数: {MAX_API_CONCURRENCY * 2}")
    print(f"- 客户端连接池大小: {MAX_CLIENT_POOL_SIZE}")

    # 测试API连接
    try:
        # 获取客户端进行测试
        client = OpenAIClientManager.get_client()
        test_completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "测试连接"}],
            stream=True
        )
        # 消费流式内容，否则可能导致连接资源泄漏
        for _ in test_completion:
            pass
        # 将测试使用的客户端放回池中
        OpenAIClientManager.release_client(client)
        print("API连接测试成功，开始高性能并行处理轨迹...")
    except Exception as e:
        print(f"API连接测试失败: {e}")
        print("请检查API密钥和网络连接后重试。")
        return

    # 创建进度条
    progress_bar = tqdm(total=len(all_trajectories), desc="总体处理进度", ncols=100)
    processed_count = 0
    success_count = 0
    error_count = 0

    # 记录开始时间
    start_time = time.time()

    # 使用线程池并提交任务
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 为每个轨迹创建一个处理任务
        futures = {executor.submit(process_single_trajectory, traj_info): traj_info for traj_info in all_trajectories}

        # 处理完成的任务
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

                # 更新进度条信息
                elapsed_time = time.time() - start_time
                items_per_second = processed_count / elapsed_time if elapsed_time > 0 else 0
                progress_bar.set_postfix({
                    "成功": success_count,
                    "失败": error_count,
                    "速率": f"{items_per_second:.2f}项/秒"
                })
                progress_bar.update(1)

            except Exception as exc:
                error_count += 1
                processed_count += 1
                print(f"[错误] 轨迹 {traj_id} ({traj_type}) 处理出现异常: {exc}")
                progress_bar.update(1)

    progress_bar.close()

    # 计算总耗时和处理速度
    total_time = time.time() - start_time
    average_time_per_item = total_time / processed_count if processed_count > 0 else 0

    print(f"\n处理完成统计:")
    print(f"- 总处理轨迹数: {processed_count}/{len(all_trajectories)}")
    print(f"- 成功处理数: {success_count}")
    print(f"- 失败处理数: {error_count}")
    print(f"- 总耗时: {total_time:.2f}秒")
    print(f"- 平均每轨迹耗时: {average_time_per_item:.2f}秒")
    print(f"- 处理速率: {processed_count / total_time:.2f}轨迹/秒")


if __name__ == "__main__":
    # 初始化客户端连接池
    for _ in range(MAX_CLIENT_POOL_SIZE):
        try:
            client_pool.put(OpenAIClientManager.create_client())
        except:
            pass

    # 运行主程序
    main()