import os
import requests
import time
import random
from PIL import Image
import numpy as np
import torch
from io import BytesIO
import base64
import threading
import hashlib
import json
from queue import Queue, Empty

# 尝试导入 oss2（按需）
OSS_AVAILABLE = False
try:
    import oss2
    OSS_AVAILABLE = True
except ImportError:
    pass


class RunningHubRhartImageToImageAlioss:
    # 添加全局请求节流器
    _last_request_times = {}
    _request_lock = threading.Lock()
    # ✅ 新增：任务缓存字典（线程安全）
    _task_cache = {}
    _cache_lock = threading.Lock()

    def __init__(self):
        self._stop_events = {}  # 存储每个组的停止事件

    @classmethod
    def INPUT_TYPES(s):
        optional_inputs = {}
        for i in range(10):  # A to J
            group_letter = chr(ord('A') + i)
            # ✅ 支持 5 张参考图：a, b, c, d, e
            for suffix in ['a', 'b', 'c', 'd', 'e']:
                optional_inputs[f"image_{group_letter}_{suffix}"] = ("IMAGE", {})
            optional_inputs[f"prompt_{i+1}"] = ("STRING", {"forceInput": True})
            optional_inputs[f"batch_count_{i+1}"] = ("INT", {
                "default": 1,
                "min": 1,
                "max": 10,
                "step": 1,
                "display": "number"
            })

        return {
            "required": {
                # === 阿里云 OSS 配置 ===
                "oss_access_key_id": ("STRING", {"default": "", "placeholder": "留空则使用 RunningHub 自带上传"}),
                "oss_access_key_secret": ("STRING", {"default": "", "placeholder": "阿里云 AccessKey Secret"}),
                "oss_bucket_name": ("STRING", {"default": "", "placeholder": "OSS Bucket 名称"}),
                "oss_endpoint": ("STRING", {"default": "oss-cn-beijing.aliyuncs.com", "placeholder": "OSS Endpoint"}),

                # === 各模型：最大尝试次数 + 优先级 ===
                "Seedream_v4_5_最大尝试次数": ("INT", {
                    "default": 2,
                    "min": 0,
                    "max": 5,
                    "step": 1,
                    "tooltip": "使用 RunningHub 的 seedream-v4.5-图生图 模型。设为0则跳过。"
                }),
                "Seedream_v4_5_优先级": ("INT", {
                    "default": 50,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "tooltip": "数值越大，调用优先级越高（仅当尝试次数>0时生效）"
                }),
                "V2_社区版_最大尝试次数": ("INT", {
                    "default": 2,
                    "min": 0,
                    "max": 5,
                    "step": 1,
                    "tooltip": "设为0则跳过该模式。"
                }),
                "V2_社区版_优先级": ("INT", {
                    "default": 40,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "tooltip": "数值越大，调用优先级越高（仅当尝试次数>0时生效）"
                }),
                "V2_官方稳定版_最大尝试次数": ("INT", {
                    "default": 2,
                    "min": 0,
                    "max": 5,
                    "step": 1,
                    "tooltip": "使用 RunningHub 的 全能图片V2-图生图-官方稳定版 模型。设为0则跳过。"
                }),
                "V2_官方稳定版_优先级": ("INT", {
                    "default": 45,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "tooltip": "数值越大，调用优先级越高（仅当尝试次数>0时生效）"
                }),
                "G2_社区版_最大尝试次数": ("INT", {
                    "default": 2,
                    "min": 0,
                    "max": 5,
                    "step": 1,
                    "tooltip": "使用 RunningHub 的 G2 社区版 模型。设为0则跳过。"
                }),
                "G2_社区版_优先级": ("INT", {
                    "default": 55,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "tooltip": "数值越大，调用优先级越高（仅当尝试次数>0时生效）"
                }),
                "社区版_最大尝试次数": ("INT", {
                    "default": 2,
                    "min": 0,
                    "max": 5,
                    "step": 1,
                    "tooltip": "设为0则跳过该模式。"
                }),
                "社区版_优先级": ("INT", {
                    "default": 30,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "tooltip": "数值越大，调用优先级越高（仅当尝试次数>0时生效）"
                }),
                "全能Xinbao_最大尝试次数": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 5,
                    "step": 1,
                    "tooltip": "设为0则跳过该模式。"
                }),
                "全能Xinbao_优先级": ("INT", {
                    "default": 20,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "tooltip": "数值越大，调用优先级越高（仅当尝试次数>0时生效）"
                }),
                "官方PRO版_最大尝试次数": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 5,
                    "step": 1,
                    "tooltip": "设为0则跳过该模式。"
                }),
                "官方PRO版_优先级": ("INT", {
                    "default": 10,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "tooltip": "数值越大，调用优先级越高（仅当尝试次数>0时生效）"
                }),

                # === API 密钥 ===
                "runninghub_api_key": ("STRING", {"default": "", "placeholder": "RunningHub API 密钥"}),
                "全能Xinbao_api_key": ("STRING", {"default": "", "placeholder": "全能Xinbao API 密钥"}),

                # === 全局参数 ===
                "resolution": (["1K", "2K", "3K", "4K", "8K"], {"default": "1K"}),
                "aspect_ratio": (["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "16:9", "9:16", "21:9", "自动"], {"default": "自动"}),
                "aspect_ratio_ref_image_index": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 5,
                    "step": 1,
                    "tooltip": "当「宽高比」设为「自动」时，使用第几张参考图的比例作为参考（1=第一张，2=第二张...最多5张）"
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "global_concurrent_tasks": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "tooltip": "全局最大处理组数（仅处理前 N 个有效组，1～10）"
                }),
                "max_wait_time": ("INT", {
                    "default": 180,
                    "min": 30,
                    "max": 600,
                    "step": 30,
                    "tooltip": "每个子任务最大等待时间（秒），适用于所有API模式"
                }),
                "全局任务总数": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "【全局】控制所有任务的总运行数量（-1=不限制）。例如设为1则只运行第1个任务，设为10则最多运行10个任务。无论是提示词行还是任务组，都受此控制。"
                }),
                "max_prompt_lines_global": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 50,
                    "step": 1,
                    "tooltip": "【全局】每组最多使用多少行提示词（-1 = 不限制）。若设为1，则整段提示词视为单行（不分割换行符）。此参数位于底部便于批量调试。"
                }),
                "每组成功数量": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "tooltip": "每组中需要多少个成功的变体结果。例如：设置为1时，任意一个变体成功就返回；设置为2时，需要2个变体成功才返回。"
                }),
                # === 新增：输出格式选择 ===
                "output_format": (["保持原格式", "JPEG", "PNG", "WEBP"], {"default": "保持原格式"}),
            },
            "optional": optional_inputs
        }

    RETURN_TYPES = ("IMAGE",) * 10 + ("IMAGE",)
    RETURN_NAMES = tuple(f"输出_{i}" for i in range(1, 11)) + ("所有成功图像",)
    FUNCTION = "generate"
    CATEGORY = "影客AI"

    # --- 工具方法 ---
    def tensor_to_pil(self, tensor):
        """将 PyTorch Tensor 转换为 PIL 图像"""
        if tensor.ndim == 4:
            tensor = tensor.squeeze(0)
        i = 255. * tensor.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        return img

    def pil_to_tensor(self, pil_img):
        """将 PIL 图像转换为 PyTorch Tensor"""
        img_array = np.array(pil_img).astype(np.float32) / 255.0
        return torch.from_numpy(img_array)[None,]

    def upload_to_aliyun_oss(self, pil_img, access_key_id, access_key_secret, bucket_name, endpoint, output_format="保持原格式"):
        """
        直接将 PIL 图像上传至阿里云 OSS，并返回公开 URL。
        
        Args:
            pil_img (PIL.Image): 待上传的图像
            access_key_id (str): 阿里云 AK ID
            access_key_secret (str): 阿里云 AK Secret
            bucket_name (str): OSS bucket 名称
            endpoint (str): OSS endpoint 地址
            output_format (str): 输出格式，可选值："保持原格式", "JPEG", "PNG", "WEBP"
        
        Returns:
            str: 生成的公开访问 URL
        
        Raises:
            RuntimeError: 上传失败时抛出
            ValueError: 配置信息不完整时抛出
        """
        if not OSS_AVAILABLE:
            raise RuntimeError("未安装 oss2 库，请运行: pip install oss2")
        if not all([access_key_id.strip(), access_key_secret.strip(), bucket_name.strip()]):
            raise ValueError("请填写完整的阿里云 OSS 配置信息")

        # 确定输出格式
        if output_format == "保持原格式":
            img_format = pil_img.format or "JPEG"  # 默认使用 JPEG 而不是 PNG
        else:
            img_format = output_format
        
        img_format = img_format.upper()
        if img_format not in ["PNG", "JPEG", "JPG", "WEBP"]:
            img_format = "JPEG"  # 默认使用 JPEG

        timestamp = str(int(time.time() * 1000))
        random_suffix = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=8))
        object_key = f"comfyui_rhart/{timestamp}_{random_suffix}.{img_format.lower()}"

        auth = oss2.Auth(access_key_id.strip(), access_key_secret.strip())
        bucket = oss2.Bucket(auth, f'https://{endpoint.strip()}', bucket_name.strip())

        buf = BytesIO()
        pil_img.save(buf, format=img_format)
        buf.seek(0)

        # 设置正确的 Content-Type
        content_type_map = {
            "PNG": "image/png",
            "JPEG": "image/jpeg",
            "JPG": "image/jpeg",
            "WEBP": "image/webp"
        }
        content_type = content_type_map.get(img_format, "image/jpeg")

        try:
            bucket.put_object(object_key, buf.getvalue(), headers={'Content-Type': content_type})
        except Exception as e:
            raise RuntimeError(f"阿里云 OSS 上传失败: {e}")

        return f"https://{bucket_name.strip()}.{endpoint.strip()}/{object_key}"

    def upload_to_runninghub(self, pil_img, runninghub_api_key, output_format="保持原格式"):
        """
        使用 RunningHub 官方 API 上传文件。

        Args:
            pil_img (PIL.Image): 待上传的图像
            runninghub_api_key (str): RunningHub API 密钥
            output_format (str): 输出格式，可选值："保持原格式", "JPEG", "PNG", "WEBP"

        Returns:
            str: 返回用于后续 API 调用的 download_url

        Raises:
            requests.HTTPError: HTTP 请求失败
            RuntimeError: 响应格式错误或缺少必要字段
        """
        # 确定输出格式
        if output_format == "保持原格式":
            img_format = pil_img.format or "JPEG"  # 默认使用 JPEG 而不是 PNG
        else:
            img_format = output_format
        
        img_format = img_format.upper()
        if img_format not in ["PNG", "JPEG", "JPG", "WEBP"]:
            img_format = "JPEG"  # 默认使用 JPEG

        buf = BytesIO()
        pil_img.save(buf, format=img_format)
        buf.seek(0)

        # 设置正确的文件名和 Content-Type
        content_type_map = {
            "PNG": "image/png",
            "JPEG": "image/jpeg",
            "JPG": "image/jpeg",
            "WEBP": "image/webp"
        }
        content_type = content_type_map.get(img_format, "image/jpeg")
        filename = f"input.{img_format.lower()}"

        files = {"file": (filename, buf, content_type)}
        headers = {"Authorization": f"Bearer {runninghub_api_key.strip()}"}

        # ✅ 修正：使用正确的 API 路径
        resp = requests.post(
            "https://www.runninghub.cn/openapi/v2/media/upload/binary",
            files=files,
            headers=headers,
            timeout=30
        )
        resp.raise_for_status()
        data = resp.json()

        # ✅ 修正：根据 OpenAPI 文档解析响应
        # 响应格式: {"code": 0, "message": "success", "data": {"type": "...", "download_url": "...", "fileName": "...", "size": "..."}}
        if data.get("code") != 0:
            raise RuntimeError(f"RunningHub 上传失败: {data.get('message', 'Unknown error')} (code: {data.get('code')})")

        result_data = data.get("data")
        if not result_data:
            raise RuntimeError(f"RunningHub 上传失败: 响应中缺少 data 字段")

        # ✅ 注意：根据文档，我们需要返回 download_url 用于后续 API 调用
        download_url = result_data.get("download_url")
        if not download_url:
            raise RuntimeError(f"RunningHub 上传失败: data 中缺少 download_url 字段")

        return download_url

    def _map_resolution_for_non_seedream(self, resolution):
        """映射非 Seedream 模型的分辨率"""
        if resolution == "3K":
            return "4K"
        return resolution

    def _infer_aspect_ratio_from_image(self, pil_img):
        """根据图像尺寸推断最接近的宽高比"""
        SUPPORTED_RATIOS = {
            "1:1": (1.0, 1.0),
            "2:3": (2.0, 3.0),
            "3:2": (3.0, 2.0),
            "3:4": (3.0, 4.0),
            "4:3": (4.0, 3.0),
            "4:5": (4.0, 5.0),
            "5:4": (5.0, 4.0),
            "16:9": (16.0, 9.0),
            "9:16": (9.0, 16.0),
            "21:9": (21.0, 9.0)
        }
        w, h = pil_img.size
        if w == 0 or h == 0:
            return "1:1"
        actual_ratio = w / h
        if abs(actual_ratio - 1.0) < 0.01:
            return "1:1"
        elif abs(actual_ratio - 2/3) < 0.01:
            return "2:3"
        elif abs(actual_ratio - 3/2) < 0.01:
            return "3:2"
        elif abs(actual_ratio - 3/4) < 0.01:
            return "3:4"
        elif abs(actual_ratio - 4/3) < 0.01:
            return "4:3"
        elif abs(actual_ratio - 4/5) < 0.01:
            return "4:5"
        elif abs(actual_ratio - 5/4) < 0.01:
            return "5:4"
        elif abs(actual_ratio - 16/9) < 0.01:
            return "16:9"
        elif abs(actual_ratio - 9/16) < 0.01:
            return "9:16"
        elif abs(actual_ratio - 21/9) < 0.01:
            return "21:9"
        else:
            best_match = "1:1"
            min_diff = float('inf')
            for ratio_str, (rw, rh) in SUPPORTED_RATIOS.items():
                target_ratio = rw / rh
                diff = abs(actual_ratio - target_ratio)
                if diff < min_diff:
                    min_diff = diff
                    best_match = ratio_str
            return best_match

    def _rate_limit_request(self, api_type, min_interval=1.0):
        """防止请求过于频繁"""
        with self._request_lock:
            current_time = time.time()
            last_time = self._last_request_times.get(api_type, 0)
            if current_time - last_time < min_interval:
                sleep_time = min_interval - (current_time - last_time)
                time.sleep(sleep_time)
            self._last_request_times[api_type] = current_time

    def _generate_task_id(self, api_type, image_urls, prompt, resolution, aspect_ratio, seed=None):
        """为任务参数生成唯一ID，用于任务复用"""
        task_params = {
            "api_type": api_type,
            "image_urls": sorted(image_urls) if image_urls else [],
            "prompt": prompt,
            "resolution": resolution,
            "aspect_ratio": aspect_ratio,
            "seed": seed
        }
        task_json = json.dumps(task_params, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(task_json.encode('utf-8')).hexdigest()

    def _get_cached_task(self, task_id):
        """获取缓存的任务信息"""
        with self._cache_lock:
            return self._task_cache.get(task_id)

    def _set_cached_task(self, task_id, task_info):
        """设置缓存的任务信息"""
        with self._cache_lock:
            self._task_cache[task_id] = task_info

    def _update_cached_task(self, task_id, **kwargs):
        """更新缓存的任务信息"""
        with self._cache_lock:
            if task_id in self._task_cache:
                self._task_cache[task_id].update(kwargs)

    def _poll_task_status(self, group_id, var_id, task_id, query_url, query_headers,
                          max_wait_time, poll_interval=2, cache_key=None, stop_event=None):
        """通用任务轮询方法，带简化日志和任务复用支持"""
        success_data = None
        dots_count = 0
        dot_pattern = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠏", "⠇"]
        max_attempts = max(1, min(max_wait_time, 600) // poll_interval)
        print(f"[组 {group_id} 变体 {var_id}] 开始轮询，总等待时间: {max_wait_time}秒，间隔: {poll_interval}秒，最大尝试次数: {max_attempts}", flush=True)

        for attempt in range(1, max_attempts + 1):
            if stop_event and stop_event.is_set():
                print(f"\n[组 {group_id} 变体 {var_id}] 任务被取消", flush=True)
                return None

            try:
                if cache_key:
                    cached_task = self._get_cached_task(cache_key)
                    if cached_task and cached_task.get("status") == "SUCCESS":
                        print(" ✓ (复用)", flush=True)
                        print(f"[组 {group_id} 变体 {var_id}] 复用先前成功的任务结果!", flush=True)
                        return cached_task.get("data")

                resp = requests.post(query_url, json={"taskId": task_id}, headers=query_headers, timeout=15)
                resp.raise_for_status()
                data = resp.json()
                current_status = data.get("status", "UNKNOWN")

                if current_status == "SUCCESS":
                    success_data = data
                    print(" ✓", flush=True)
                    print(f"[组 {group_id} 变体 {var_id}] 任务成功!", flush=True)
                    if cache_key:
                        self._set_cached_task(cache_key, {
                            "status": "SUCCESS",
                            "data": data,
                            "task_id": task_id
                        })
                    break
                elif current_status in ["FAILED", "ERROR"]:
                    error_msg = data.get("errorMessage", "未知错误")
                    print(" ✗", flush=True)
                    raise RuntimeError(f"任务失败: {error_msg}")
                else:
                    if attempt % 5 == 0:
                        progress_percent = min(int(attempt / max_attempts * 100), 100)
                        print(f"\r[组 {group_id} 变体 {var_id}] 正在生成中 ({progress_percent}%)", end="", flush=True)
                    else:
                        print(f"\r[组 {group_id} 变体 {var_id}] 正在生成中 {dot_pattern[dots_count % len(dot_pattern)]}", end="", flush=True)
                        dots_count += 1

                if stop_event and stop_event.wait(timeout=poll_interval):
                    print(f"\n[组 {group_id} 变体 {var_id}] 任务被取消", flush=True)
                    return None
                elif not stop_event:
                    time.sleep(poll_interval)

            except Exception as e:
                if cache_key:
                    cached_task = self._get_cached_task(cache_key)
                    if cached_task and cached_task.get("status") == "SUCCESS":
                        print(" ✓ (复用)", flush=True)
                        print(f"[组 {group_id} 变体 {var_id}] 复用先前成功的任务结果!", flush=True)
                        return cached_task.get("data")

                if attempt % 5 == 0:
                    progress_percent = min(int(attempt / max_attempts * 100), 100)
                    print(f"\r[组 {group_id} 变体 {var_id}] 正在生成中 ({progress_percent}%)", end="", flush=True)
                else:
                    print(f"\r[组 {group_id} 变体 {var_id}] 正在生成中 {dot_pattern[dots_count % len(dot_pattern)]}", end="", flush=True)
                    dots_count += 1

                if stop_event and stop_event.wait(timeout=poll_interval):
                    print(f"\n[组 {group_id} 变体 {var_id}] 任务被取消", flush=True)
                    return None
                elif not stop_event:
                    time.sleep(poll_interval)

        if success_data is None:
            print(f"\n[组 {group_id} 变体 {var_id}] 标准轮询阶段完成（{max_wait_time}秒），未获得成功结果", flush=True)
        return success_data

    def process_single_variation_seedream_v4_5(self, group_id, var_id, image_urls, prompt, seed,
                                              api_key, resolution, aspect_ratio, max_wait_time, stop_event=None):
        self._rate_limit_request("seedream_v4_5", 0.5)

        base_url = "https://www.runninghub.cn"
        headers = {
            "Authorization": f"Bearer {api_key.strip()}",
            "Content-Type": "application/json"
        }

        resolution_map = {
            "1K": "2k",
            "2K": "2k", 
            "3K": "4k",
            "4K": "4k",
            "8K": "4k"
        }

        payload = {
            "prompt": prompt,
            "imageUrls": image_urls[:10],
            "sequentialImageGeneration": "disabled",
            "maxImages": 1,
            "resolution": resolution_map.get(resolution, "2k")
        }

        try:
            submit_resp = requests.post(
                f"{base_url}/openapi/v2/seedream-v4.5/image-to-image",
                json=payload,
                headers=headers,
                timeout=min(max_wait_time + 10, 120)
            )
            submit_resp.raise_for_status()
            data = submit_resp.json()

            if data.get("errorCode"):
                raise RuntimeError(f"Seedream v4.5 API 错误: {data.get('errorMessage', 'Unknown')} (code: {data.get('errorCode')})")

            task_id = data.get("taskId")
            if not task_id:
                raise RuntimeError("未返回有效的 taskId")

            print(f"[组 {group_id} 变体 {var_id}] 任务已提交 ✓", flush=True)

            cache_key = self._generate_task_id(
                "seedream_v4_5", image_urls, prompt, resolution, "忽略", seed
            )

            query_url = f"{base_url}/openapi/v2/query"
            query_headers = {"Authorization": f"Bearer {api_key.strip()}"}

            success_data = self._poll_task_status(
                group_id, var_id, task_id, query_url, query_headers,
                max_wait_time, 2, cache_key, stop_event
            )

            if success_data is None:
                print(f"[组 {group_id} 变体 {var_id}] 轮询超时，启动最终确认阶段（最多等待额外30秒）...", flush=True)
                final_max_attempts = 10
                final_poll_interval = 3

                for final_attempt in range(final_max_attempts):
                    if stop_event and stop_event.is_set():
                        print(f"\n[组 {group_id} 变体 {var_id}] 任务被取消", flush=True)
                        return None

                    try:
                        final_resp = requests.post(
                            query_url,
                            json={"taskId": task_id},
                            headers=query_headers,
                            timeout=15
                        )
                        final_resp.raise_for_status()
                        final_data = final_resp.json()

                        current_status = final_data.get("status", "UNKNOWN")
                        if current_status == "SUCCESS":
                            success_data = final_data
                            print(f"[组 {group_id} 变体 {var_id}] 最终确认成功！任务已完成 ✅", flush=True)
                            if cache_key:
                                self._set_cached_task(cache_key, {
                                    "status": "SUCCESS",
                                    "data": final_data,
                                    "task_id": task_id
                                })
                            break
                        elif current_status == "FAILED":
                            error_msg = final_data.get("errorMessage", "Unknown error")
                            raise RuntimeError(f"最终确认：任务已失败: {error_msg}")
                        else:
                            if final_attempt < final_max_attempts - 1:
                                print(f"[组 {group_id} 变体 {var_id}] 最终确认第 {final_attempt + 1} 次：状态={current_status}，{final_poll_interval}秒后重试...", flush=True)
                                if stop_event and stop_event.wait(timeout=final_poll_interval):
                                    print(f"\n[组 {group_id} 变体 {var_id}] 任务被取消", flush=True)
                                    return None
                                else:
                                    time.sleep(final_poll_interval)
                            else:
                                raise RuntimeError(f"最终确认超时：经过 {final_max_attempts * final_poll_interval} 秒额外等待，状态仍为: {current_status}")
                    except Exception as e:
                        if final_attempt < final_max_attempts - 1:
                            print(f"[组 {group_id} 变体 {var_id}] 最终确认网络错误，{final_poll_interval}秒后重试: {e}", flush=True)
                            if stop_event and stop_event.wait(timeout=final_poll_interval):
                                print(f"\n[组 {group_id} 变体 {var_id}] 任务被取消", flush=True)
                                return None
                            else:
                                time.sleep(final_poll_interval)
                        else:
                            raise RuntimeError(f"最终确认网络错误且重试耗尽: {e}")

            results = success_data.get("results", [])
            if not results or not isinstance(results, list) or len(results) == 0:
                file_url = success_data.get("fileUrl") or success_data.get("imageUrl")
                if file_url:
                    results = [{"url": file_url}]
                else:
                    raise RuntimeError("无生成结果，且 results 字段为空")

            output_url = results[0].get("url")
            if not output_url:
                raise RuntimeError("结果 URL 为空")

            print(f"[组 {group_id} 变体 {var_id}] 正在下载结果图片...", flush=True)
            img_resp = requests.get(output_url, timeout=30)
            img_resp.raise_for_status()
            return Image.open(BytesIO(img_resp.content))

        except requests.exceptions.Timeout:
            raise RuntimeError(f"Seedream v4.5 请求超时（>{max_wait_time}秒）")
        except Exception as e:
            raise RuntimeError(f"Seedream v4.5 调用失败: {e}")

    def process_single_variation_banana(self, group_id, var_id, image_urls, prompt, seed,
                                      banana_api_key, model, resolution, aspect_ratio, max_wait_time):
        self._rate_limit_request("xinbao", 1.0)

        base_url = "https://xinbaoapi.dpdns.org"
        headers = {
            "Authorization": f"Bearer {banana_api_key.strip()}",
            "Content-Type": "application/json"
        }

        parts = [{"text": prompt}]
        for url in image_urls[:5]:
            parts.append({
                "inlineData": {
                    "mimeType": "image/jpeg",
                    "data": url
                }
            })

        image_config = {}
        mapped_res = self._map_resolution_for_non_seedream(resolution)
        if mapped_res in ["1K", "2K", "4K", "8K"]:
            api_res = "4K" if mapped_res == "8K" else mapped_res
            image_config["imageSize"] = api_res

        if aspect_ratio != "自动":
            image_config["aspectRatio"] = aspect_ratio

        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "responseModalities": ["IMAGE"],
                "output": "url",
                **({"topP": 0.95} if seed is not None else {}),
                **({"imageConfig": image_config} if image_config else {})
            }
        }

        resp = requests.post(
            f"{base_url}/v1beta/models/gemini-3-pro-image-preview:generateContent",
            json=payload,
            headers=headers,
            timeout=120
        )
        resp.raise_for_status()
        data = resp.json()

        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError(f"[组 {group_id} 变体 {var_id}] 全能Xinbao 无候选结果")

        parts_out = candidates[0].get("content", {}).get("parts", [])
        output_pil = None
        for part in parts_out:
            inline = part.get("inlineData", {})
            mime_type = inline.get("mimeType", "")
            img_data = inline.get("data", "")
            if mime_type.startswith("image/") and isinstance(img_data, str):
                try:
                    if img_data.startswith("http"):
                        img_resp = requests.get(img_data, timeout=30)
                        img_resp.raise_for_status()
                        output_pil = Image.open(BytesIO(img_resp.content))
                    else:
                        image_bytes = base64.b64decode(img_data)
                        output_pil = Image.open(BytesIO(image_bytes))
                    break
                except Exception as e:
                    continue

        if output_pil is None:
            raise RuntimeError(f"[组 {group_id} 变体 {var_id}] 全能Xinbao 未返回可解析图片")
        return output_pil

    def process_single_variation_g2_community(self, group_id, var_id, image_urls, prompt, seed,
                                            api_key, resolution, aspect_ratio, max_wait_time):
        self._rate_limit_request("g2_community", 0.8)

        base_url = "https://www.runninghub.cn"
        endpoint_path = "/openapi/v2/rhart-image-g-2/image-to-image"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        poll_interval = 2

        # 处理分辨率
        mapped_res = self._map_resolution_for_non_seedream(resolution)
        api_resolution = "4K" if mapped_res == "8K" else mapped_res

        submit_payload = {"prompt": prompt, "imageUrls": image_urls, "resolution": api_resolution.lower()}
        if aspect_ratio != "自动":
            # G2 社区版仅支持有限的宽高比
            ar_map = {"1:1":"1:1","2:3":"2:3","3:2":"3:2"}
            if aspect_ratio in ar_map:
                submit_payload["aspectRatio"] = ar_map[aspect_ratio]
            else:
                # 如果宽高比不在支持列表中，使用默认值
                print(f"[组 {group_id} 变体 {var_id}] G2_社区版不支持宽高比 {aspect_ratio}，使用默认值")

        print(f"[组 {group_id} 变体 {var_id}] G2_社区版 API 请求: {submit_payload}")
        submit_resp = requests.post(f"{base_url}{endpoint_path}", json=submit_payload, headers=headers, timeout=30)
        submit_resp.raise_for_status()
        
        # 打印响应内容以便调试
        response_json = submit_resp.json()
        print(f"[组 {group_id} 变体 {var_id}] G2_社区版 API 响应: {response_json}")
        
        task_id = response_json.get("taskId")
        if not task_id:
            raise RuntimeError(f"[组 {group_id} 变体 {var_id}] 未返回 taskId，响应: {response_json}")

        cache_key = self._generate_task_id(
            "runninghub_g2_community",
            image_urls, prompt, resolution, aspect_ratio, seed
        )

        query_url = f"{base_url}/openapi/v2/query"
        query_headers = {"Authorization": f"Bearer {api_key.strip()}"}

        success_data = self._poll_task_status(
            group_id, var_id, task_id, query_url, query_headers,
            max_wait_time, poll_interval, cache_key
        )

        if success_data is None:
            print(f"[组 {group_id} 变体 {var_id}] 轮询超时，启动最终确认阶段（最多等待额外30秒）...", flush=True)
            final_max_attempts = 10
            final_poll_interval = 3

            for final_attempt in range(final_max_attempts):
                try:
                    final_resp = requests.post(
                        query_url,
                        json={"taskId": task_id},
                        headers=query_headers,
                        timeout=15
                    )
                    final_resp.raise_for_status()
                    final_data = final_resp.json()

                    current_status = final_data.get("status", "UNKNOWN")
                    if current_status == "SUCCESS":
                        success_data = final_data
                        print(f"[组 {group_id} 变体 {var_id}] 最终确认成功！任务已完成 ✅", flush=True)
                        if cache_key:
                            self._set_cached_task(cache_key, {
                                "status": "SUCCESS",
                                "data": final_data,
                                "task_id": task_id
                            })
                        break
                    elif current_status == "FAILED":
                        error_msg = final_data.get("errorMessage", "Unknown error")
                        raise RuntimeError(f"最终确认：任务已失败: {error_msg}")
                    else:
                        if final_attempt < final_max_attempts - 1:
                            print(f"[组 {group_id} 变体 {var_id}] 最终确认第 {final_attempt + 1} 次：状态={current_status}，{final_poll_interval}秒后重试...", flush=True)
                            time.sleep(final_poll_interval)
                        else:
                            raise RuntimeError(f"最终确认超时：经过 {final_max_attempts * final_poll_interval} 秒额外等待，状态仍为: {current_status}")
                except Exception as e:
                    if final_attempt < final_max_attempts - 1:
                        print(f"[组 {group_id} 变体 {var_id}] 最终确认网络错误，{final_poll_interval}秒后重试: {e}", flush=True)
                        time.sleep(final_poll_interval)
                    else:
                        raise RuntimeError(f"最终确认网络错误且重试耗尽: {e}")

        results = success_data.get("results", [])
        if not results or not isinstance(results, list) or len(results) == 0:
            file_url = success_data.get("fileUrl") or success_data.get("imageUrl")
            if file_url:
                results = [{"url": file_url}]
            else:
                raise RuntimeError("无生成结果，且 results 字段为空")

        output_url = results[0].get("url")
        if not output_url:
            raise RuntimeError("结果 URL 为空")

        print(f"[组 {group_id} 变体 {var_id}] 正在下载结果图片...", flush=True)
        img_resp = requests.get(output_url, timeout=30)
        img_resp.raise_for_status()

        image_data = BytesIO(img_resp.content)
        output_pil = Image.open(image_data).convert("RGB")

        if output_pil is None:
            raise RuntimeError(f"[组 {group_id} 变体 {var_id}] G2_社区版 未返回可解析图片")
        return output_pil

    def process_single_variation_runninghub(self, group_id, var_id, image_urls, prompt, seed,
                                          api_key, resolution, aspect_ratio, max_wait_time, endpoint_path):
        self._rate_limit_request("community" if "community" in endpoint_path else "official", 0.8)

        base_url = "https://www.runninghub.cn"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        poll_interval = 2

        mapped_res = self._map_resolution_for_non_seedream(resolution)
        api_resolution = "4K" if mapped_res == "8K" else mapped_res

        submit_payload = {"prompt": prompt, "imageUrls": image_urls, "resolution": api_resolution.lower()}
        if aspect_ratio != "自动":
            ar_map = {"1:1":"1:1","2:3":"2:3","3:2":"3:2","3:4":"3:4","4:3":"4:3","4:5":"4:5","5:4":"5:4","16:9":"16:9","9:16":"9:16","21:9":"21:9"}
            submit_payload["aspectRatio"] = ar_map.get(aspect_ratio, "auto")

        submit_resp = requests.post(f"{base_url}{endpoint_path}", json=submit_payload, headers=headers, timeout=30)
        submit_resp.raise_for_status()
        task_id = submit_resp.json().get("taskId")
        if not task_id:
            raise RuntimeError(f"[组 {group_id} 变体 {var_id}] 未返回 taskId")

        cache_key = self._generate_task_id(
            "runninghub_" + ("official" if "official" in endpoint_path else "community"),
            image_urls, prompt, resolution, aspect_ratio, seed
        )

        query_url = f"{base_url}/openapi/v2/query"
        query_headers = {"Authorization": f"Bearer {api_key.strip()}"}

        success_data = self._poll_task_status(
            group_id, var_id, task_id, query_url, query_headers,
            max_wait_time, poll_interval, cache_key
        )

        if success_data is None:
            print(f"[组 {group_id} 变体 {var_id}] 轮询超时，启动最终确认阶段（最多等待额外30秒）...", flush=True)
            final_max_attempts = 10
            final_poll_interval = 3

            for final_attempt in range(final_max_attempts):
                try:
                    final_resp = requests.post(
                        query_url,
                        json={"taskId": task_id},
                        headers=query_headers,
                        timeout=15
                    )
                    final_resp.raise_for_status()
                    final_data = final_resp.json()

                    current_status = final_data.get("status", "UNKNOWN")
                    if current_status == "SUCCESS":
                        success_data = final_data
                        print(f"[组 {group_id} 变体 {var_id}] 最终确认成功！任务已完成 ✅", flush=True)
                        if cache_key:
                            self._set_cached_task(cache_key, {
                                "status": "SUCCESS",
                                "data": final_data,
                                "task_id": task_id
                            })
                        break
                    elif current_status == "FAILED":
                        error_msg = final_data.get("errorMessage", "Unknown error")
                        raise RuntimeError(f"最终确认：任务已失败: {error_msg}")
                    else:
                        if final_attempt < final_max_attempts - 1:
                            print(f"[组 {group_id} 变体 {var_id}] 最终确认第 {final_attempt + 1} 次：状态={current_status}，{final_poll_interval}秒后重试...", flush=True)
                            time.sleep(final_poll_interval)
                        else:
                            raise RuntimeError(f"最终确认超时：经过 {final_max_attempts * final_poll_interval} 秒额外等待，状态仍为: {current_status}")
                except Exception as e:
                    if final_attempt < final_max_attempts - 1:
                        print(f"[组 {group_id} 变体 {var_id}] 最终确认网络错误，{final_poll_interval}秒后重试: {e}", flush=True)
                        time.sleep(final_poll_interval)
                    else:
                        raise RuntimeError(f"最终确认网络错误且重试耗尽: {e}")

        results = success_data.get("results", [])
        if not results or not isinstance(results, list) or len(results) == 0:
            file_url = success_data.get("fileUrl") or success_data.get("imageUrl")
            if file_url:
                results = [{"url": file_url}]
            else:
                raise RuntimeError("无生成结果，且 results 字段为空")

        output_url = results[0].get("url")
        if not output_url:
            raise RuntimeError("结果 URL 为空")

        print(f"[组 {group_id} 变体 {var_id}] 正在下载结果图片...", flush=True)
        img_resp = requests.get(output_url, timeout=30)
        img_resp.raise_for_status()
        return Image.open(BytesIO(img_resp.content))

    def process_single_variation_g31_flash(self, group_id, var_id, image_urls, prompt, seed,
                                         api_key, resolution, aspect_ratio, max_wait_time, stop_event=None):
        self._rate_limit_request("g31_flash", 0.5)

        base_url = "https://www.runninghub.cn"
        headers = {
            "Authorization": f"Bearer {api_key.strip()}",
            "Content-Type": "application/json"
        }

        resolution_map = {
            "1K": "1k",
            "2K": "2k", 
            "3K": "4k",
            "4K": "4k",
            "8K": "4k"
        }

        payload = {
            "imageUrls": image_urls[:10],
            "prompt": prompt,
            "resolution": resolution_map.get(resolution, "1k")
        }

        if aspect_ratio != "自动":
            ar_map = {
                "1:1": "1:1", "2:3": "2:3", "3:2": "3:2", "3:4": "3:4", "4:3": "4:3",
                "4:5": "4:5", "5:4": "5:4", "16:9": "16:9", "9:16": "9:16", "21:9": "21:9",
                "1:4": "1:4", "4:1": "4:1", "1:8": "1:8", "8:1": "8:1"
            }
            payload["aspectRatio"] = ar_map.get(aspect_ratio, "1:1")

        try:
            submit_resp = requests.post(
                f"{base_url}/openapi/v2/rhart-image-n-g31-flash/image-to-image",
                json=payload,
                headers=headers,
                timeout=30
            )
            if submit_resp.status_code != 200:
                raise RuntimeError(f"HTTP {submit_resp.status_code}: {submit_resp.text}")
            submit_resp.raise_for_status()
            task_data = submit_resp.json()
            error_code = task_data.get("errorCode")
            error_message = task_data.get("errorMessage")
            if error_code or error_message:
                raise RuntimeError(f"API错误: {error_message or '未知错误'} (code: {error_code})")
            task_id = task_data.get("taskId")
            if not task_id:
                raise RuntimeError(f"未返回taskId，完整响应: {task_data}")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"网络请求失败: {e}")
        except Exception as e:
            raise RuntimeError(f"任务提交失败: {e}")

        print(f"[组 {group_id} 变体 {var_id}] 任务已提交 ✓", flush=True)

        cache_key = self._generate_task_id(
            "g31_flash", image_urls, prompt, resolution, aspect_ratio, seed
        )

        query_url = f"{base_url}/openapi/v2/query"
        query_headers = {"Authorization": f"Bearer {api_key.strip()}"}

        success_data = self._poll_task_status(
            group_id, var_id, task_id, query_url, query_headers,
            max_wait_time, 2, cache_key, stop_event
        )

        if success_data is None:
            print(f"[组 {group_id} 变体 {var_id}] 轮询超时，启动最终确认阶段（最多等待额外30秒）...", flush=True)
            final_max_attempts = 10
            final_poll_interval = 3

            for final_attempt in range(final_max_attempts):
                if stop_event and stop_event.is_set():
                    print(f"\n[组 {group_id} 变体 {var_id}] 任务被取消", flush=True)
                    return None

                try:
                    final_resp = requests.post(
                        query_url,
                        json={"taskId": task_id},
                        headers=query_headers,
                        timeout=15
                    )
                    final_resp.raise_for_status()
                    final_data = final_resp.json()

                    current_status = final_data.get("status", "UNKNOWN")
                    if current_status == "SUCCESS":
                        success_data = final_data
                        print(f"[组 {group_id} 变体 {var_id}] 最终确认成功！任务已完成 ✅", flush=True)
                        if cache_key:
                            self._set_cached_task(cache_key, {
                                "status": "SUCCESS",
                                "data": final_data,
                                "task_id": task_id
                            })
                        break
                    elif current_status == "FAILED":
                        error_msg = final_data.get("errorMessage", "Unknown error")
                        raise RuntimeError(f"最终确认：任务已失败: {error_msg}")
                    else:
                        if final_attempt < final_max_attempts - 1:
                            print(f"[组 {group_id} 变体 {var_id}] 最终确认第 {final_attempt + 1} 次：状态={current_status}，{final_poll_interval}秒后重试...", flush=True)
                            if stop_event and stop_event.wait(timeout=final_poll_interval):
                                print(f"\n[组 {group_id} 变体 {var_id}] 任务被取消", flush=True)
                                return None
                            else:
                                time.sleep(final_poll_interval)
                        else:
                            raise RuntimeError(f"最终确认超时：经过 {final_max_attempts * final_poll_interval} 秒额外等待，状态仍为: {current_status}")
                except Exception as e:
                    if final_attempt < final_max_attempts - 1:
                        print(f"[组 {group_id} 变体 {var_id}] 最终确认网络错误，{final_poll_interval}秒后重试: {e}", flush=True)
                        if stop_event and stop_event.wait(timeout=final_poll_interval):
                            print(f"\n[组 {group_id} 变体 {var_id}] 任务被取消", flush=True)
                            return None
                        else:
                            time.sleep(final_poll_interval)
                    else:
                        raise RuntimeError(f"最终确认网络错误且重试耗尽: {e}")

        results = success_data.get("results", [])
        if not results or not isinstance(results, list) or len(results) == 0:
            file_url = success_data.get("fileUrl") or success_data.get("imageUrl")
            if file_url:
                results = [{"url": file_url}]
            else:
                raise RuntimeError("无生成结果，且 results 字段为空")

        output_url = results[0].get("url")
        if not output_url:
            raise RuntimeError("结果 URL 为空")

        print(f"[组 {group_id} 变体 {var_id}] 正在下载结果图片...", flush=True)
        img_resp = requests.get(output_url, timeout=30)
        img_resp.raise_for_status()
        return Image.open(BytesIO(img_resp.content))

    def process_single_variation_g31_flash_official(self, group_id, var_id, image_urls, prompt, seed,
                                                  api_key, resolution, aspect_ratio, max_wait_time, stop_event=None):
        self._rate_limit_request("g31_flash_official", 0.5)

        base_url = "https://www.runninghub.cn"
        headers = {
            "Authorization": f"Bearer {api_key.strip()}",
            "Content-Type": "application/json"
        }

        resolution_map = {
            "1K": "1k",
            "2K": "2k", 
            "3K": "4k",
            "4K": "4k",
            "8K": "4k"
        }

        payload = {
            "imageUrls": image_urls[:14],
            "prompt": prompt,
            "resolution": resolution_map.get(resolution, "1k")
        }

        if aspect_ratio != "自动":
            ar_map = {
                "1:1": "1:1", "2:3": "2:3", "3:2": "3:2", "3:4": "3:4", "4:3": "4:3",
                "4:5": "4:5", "5:4": "5:4", "16:9": "16:9", "9:16": "9:16", "21:9": "21:9",
                "1:4": "1:4", "4:1": "4:1", "1:8": "1:8", "8:1": "8:1"
            }
            payload["aspectRatio"] = ar_map.get(aspect_ratio, "1:1")

        try:
            submit_resp = requests.post(
                f"{base_url}/openapi/v2/rhart-image-n-g31-flash-official/image-to-image",
                json=payload,
                headers=headers,
                timeout=30
            )
            if submit_resp.status_code != 200:
                raise RuntimeError(f"HTTP {submit_resp.status_code}: {submit_resp.text}")
            submit_resp.raise_for_status()
            task_data = submit_resp.json()
            error_code = task_data.get("errorCode")
            error_message = task_data.get("errorMessage")
            if error_code or error_message:
                raise RuntimeError(f"API错误: {error_message or '未知错误'} (code: {error_code})")
            task_id = task_data.get("taskId")
            if not task_id:
                raise RuntimeError(f"未返回taskId，完整响应: {task_data}")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"网络请求失败: {e}")
        except Exception as e:
            raise RuntimeError(f"任务提交失败: {e}")

        print(f"[组 {group_id} 变体 {var_id}] 任务已提交 ✓", flush=True)

        cache_key = self._generate_task_id(
            "g31_flash_official", image_urls, prompt, resolution, aspect_ratio, seed
        )

        query_url = f"{base_url}/openapi/v2/query"
        query_headers = {"Authorization": f"Bearer {api_key.strip()}"}

        success_data = self._poll_task_status(
            group_id, var_id, task_id, query_url, query_headers,
            max_wait_time, 2, cache_key, stop_event
        )

        if success_data is None:
            print(f"[组 {group_id} 变体 {var_id}] 轮询超时，启动最终确认阶段（最多等待额外30秒）...", flush=True)
            final_max_attempts = 10
            final_poll_interval = 3

            for final_attempt in range(final_max_attempts):
                if stop_event and stop_event.is_set():
                    print(f"\n[组 {group_id} 变体 {var_id}] 任务被取消", flush=True)
                    return None

                try:
                    final_resp = requests.post(
                        query_url,
                        json={"taskId": task_id},
                        headers=query_headers,
                        timeout=15
                    )
                    final_resp.raise_for_status()
                    final_data = final_resp.json()

                    current_status = final_data.get("status", "UNKNOWN")
                    if current_status == "SUCCESS":
                        success_data = final_data
                        print(f"[组 {group_id} 变体 {var_id}] 最终确认成功！任务已完成 ✅", flush=True)
                        if cache_key:
                            self._set_cached_task(cache_key, {
                                "status": "SUCCESS",
                                "data": final_data,
                                "task_id": task_id
                            })
                        break
                    elif current_status == "FAILED":
                        error_msg = final_data.get("errorMessage", "Unknown error")
                        raise RuntimeError(f"最终确认：任务已失败: {error_msg}")
                    else:
                        if final_attempt < final_max_attempts - 1:
                            print(f"[组 {group_id} 变体 {var_id}] 最终确认第 {final_attempt + 1} 次：状态={current_status}，{final_poll_interval}秒后重试...", flush=True)
                            if stop_event and stop_event.wait(timeout=final_poll_interval):
                                print(f"\n[组 {group_id} 变体 {var_id}] 任务被取消", flush=True)
                                return None
                            else:
                                time.sleep(final_poll_interval)
                        else:
                            raise RuntimeError(f"最终确认超时：经过 {final_max_attempts * final_poll_interval} 秒额外等待，状态仍为: {current_status}")
                except Exception as e:
                    if final_attempt < final_max_attempts - 1:
                        print(f"[组 {group_id} 变体 {var_id}] 最终确认网络错误，{final_poll_interval}秒后重试: {e}", flush=True)
                        if stop_event and stop_event.wait(timeout=final_poll_interval):
                            print(f"\n[组 {group_id} 变体 {var_id}] 任务被取消", flush=True)
                            return None
                        else:
                            time.sleep(final_poll_interval)
                    else:
                        raise RuntimeError(f"最终确认网络错误且重试耗尽: {e}")

        results = success_data.get("results", [])
        if not results or not isinstance(results, list) or len(results) == 0:
            file_url = success_data.get("fileUrl") or success_data.get("imageUrl")
            if file_url:
                results = [{"url": file_url}]
            else:
                raise RuntimeError("无生成结果，且 results 字段为空")

        output_url = results[0].get("url")
        if not output_url:
            raise RuntimeError("结果 URL 为空")

        print(f"[组 {group_id} 变体 {var_id}] 正在下载结果图片...", flush=True)
        img_resp = requests.get(output_url, timeout=30)
        img_resp.raise_for_status()
        return Image.open(BytesIO(img_resp.content))

    def _build_strategy_by_priority(self,
                                  seedream_v4_5_tries, g31_tries, g31_official_tries, g2_community_tries, community_tries, xinbao_tries, official_tries,
                                  seedream_v4_5_prio, g31_prio, g31_official_prio, g2_community_prio, community_prio, xinbao_prio, official_prio):
        models = [
            ("Seedream_v4_5", "seedream_v4_5", seedream_v4_5_tries, seedream_v4_5_prio),
            ("V2_社区版", "g31_flash", g31_tries, g31_prio),
            ("V2_官方稳定版", "g31_flash_official", g31_official_tries, g31_official_prio),
            ("G2_社区版", "g2_community", g2_community_tries, g2_community_prio),
            ("社区版", "community", community_tries, community_prio),
            ("全能Xinbao", "xinbao", xinbao_tries, xinbao_prio),
            ("官方PRO版", "official", official_tries, official_prio),
        ]

        active_models = []
        for name, api_type, tries, prio in models:
            if tries > 0:
                active_models.append({
                    "name": name,
                    "type": api_type,
                    "max_retries": tries,
                    "priority": prio,
                    "_index": len(active_models)
                })

        if not active_models:
            raise ValueError("所有模式的尝试次数均为0，请至少启用一个模型（将某个「最大尝试次数」设为 ≥1）")

        active_models.sort(key=lambda x: (-x["priority"], x["_index"]))
        strategy = [{"type": m["type"], "max_retries": m["max_retries"]} for m in active_models]
        return strategy

    API_TYPE_NAMES = {
        "community": "社区版",
        "official": "官方PRO版",
        "xinbao": "全能Xinbao",
        "g31_flash": "V2_社区版",
        "g31_flash_official": "V2_官方稳定版",
        "seedream_v4_5": "Seedream_v4_5",
        "g2_community": "G2_社区版"
    }

    def _attempt_with_strategy(self, group_id, var_id, image_urls, prompt,
                             runninghub_api_key, banana_api_key,
                             resolution, aspect_ratio, max_wait_time,
                             strategy, stop_event=None):
        total_attempt = 0
        base_cache_key = self._generate_task_id(
            "strategy_base", image_urls, prompt, resolution, aspect_ratio
        )

        for step in strategy:
            api_type = step["type"]
            max_retries = step["max_retries"]
            display_name = self.API_TYPE_NAMES.get(api_type, api_type)

            if stop_event and stop_event.is_set():
                return None

            api_cache_key = f"{base_cache_key}_{api_type}"

            cached_result = self._get_cached_task(base_cache_key)
            if cached_result and cached_result.get("status") == "SUCCESS":
                print(f"[组 {group_id} 变体 {var_id}] 复用先前其他API的成功结果! ✅", flush=True)
                return cached_result.get("image")

            for retry in range(max_retries):
                total_attempt += 1
                seed = random.randint(0, 0xffffffff)

                if stop_event and stop_event.is_set():
                    return None

                try:
                    if api_type == "seedream_v4_5":
                        img = self.process_single_variation_seedream_v4_5(
                            group_id, var_id, image_urls, prompt, seed,
                            runninghub_api_key,
                            resolution, "忽略", max_wait_time, stop_event
                        )
                    elif api_type == "community":
                        img = self.process_single_variation_runninghub(
                            group_id, var_id, image_urls, prompt, seed,
                            runninghub_api_key,
                            resolution, aspect_ratio, max_wait_time,
                            self._get_endpoint_paths("community")
                        )
                    elif api_type == "official":
                        img = self.process_single_variation_runninghub(
                            group_id, var_id, image_urls, prompt, seed,
                            runninghub_api_key,
                            resolution, aspect_ratio, max_wait_time,
                            self._get_endpoint_paths("official")
                        )
                    elif api_type == "xinbao":
                        img = self.process_single_variation_banana(
                            group_id, var_id, image_urls, prompt, seed,
                            banana_api_key,
                            "gemini-3-pro-image-preview",
                            resolution, aspect_ratio, max_wait_time
                        )
                    elif api_type == "g2_community":
                        img = self.process_single_variation_g2_community(
                            group_id, var_id, image_urls, prompt, seed,
                            runninghub_api_key,
                            resolution, aspect_ratio, max_wait_time
                        )
                    elif api_type == "g31_flash":
                        img = self.process_single_variation_g31_flash(
                            group_id, var_id, image_urls, prompt, seed,
                            runninghub_api_key,
                            resolution, aspect_ratio, max_wait_time,
                            stop_event
                        )
                    elif api_type == "g31_flash_official":
                        img = self.process_single_variation_g31_flash_official(
                            group_id, var_id, image_urls, prompt, seed,
                            runninghub_api_key,
                            resolution, aspect_ratio, max_wait_time,
                            stop_event
                        )

                    print(f"[组 {group_id} 变体 {var_id}] {display_name} 成功 ✅", flush=True)
                    success_data = {
                        "status": "SUCCESS",
                        "image": img,
                        "api_type": api_type,
                        "retry_count": retry + 1
                    }
                    self._set_cached_task(base_cache_key, success_data)
                    self._set_cached_task(api_cache_key, success_data)
                    return img

                except Exception as e:
                    error_str = str(e).lower()
                    is_network_error = any(keyword in error_str for keyword in [
                        "timeout", "timed out", "connection", "read timed out", 
                        "connectionpool", "network", "ssl", "reset", "broken pipe"
                    ])
                    is_server_error = any(keyword in error_str for keyword in [
                        "500", "502", "503", "504", "server", "service unavailable"
                    ])
                    is_rate_limit = "429" in error_str or "rate limit" in error_str

                    if is_network_error or is_server_error:
                        effective_max_retries = max(max_retries, 3)
                        base_delay = 2.0
                        max_delay = 30.0
                    elif is_rate_limit:
                        effective_max_retries = max_retries
                        base_delay = 5.0
                        max_delay = 60.0
                    else:
                        effective_max_retries = max_retries
                        base_delay = 1.0
                        max_delay = 10.0

                    if retry < effective_max_retries - 1:
                        wait_time = min(base_delay * (2 ** retry), max_delay)
                        jitter = 0.8 + random.random() * 0.4
                        wait_sec = wait_time * jitter

                        if stop_event and stop_event.wait(timeout=wait_sec):
                            return None

                        print(f"⚠️ [组 {group_id} 变体 {var_id}] {display_name} 第 {retry+1} 次失败: {e}")
                        print(f" → {wait_sec:.1f} 秒后重试...", flush=True)
                    else:
                        print(f"⚠️ [组 {group_id} 变体 {var_id}] {display_name} 第 {retry+1} 次失败: {e}")

        print(f"❌ [组 {group_id} 变体 {var_id}] 所有 {total_attempt} 次尝试均失败", flush=True)
        return None

    def _execute_variants_with_target_success(self, group_id, variant_configs, 
                                            target_success_count, runninghub_api_key, 
                                            banana_api_key, resolution, 
                                            aspect_ratio, max_wait_time, strategy):
        from queue import Queue, Empty
        import threading

        stop_event = threading.Event()
        self._stop_events[group_id] = stop_event

        result_queue = Queue()
        success_count = 0
        total_tasks = len(variant_configs)
        completed_tasks = 0
        lock = threading.Lock()

        def worker(var_id, prompt, image_urls):
            nonlocal success_count, completed_tasks
            try:
                if stop_event.is_set():
                    return

                result = self._attempt_with_strategy(
                    group_id, var_id, image_urls, prompt,
                    runninghub_api_key, banana_api_key,
                    resolution, aspect_ratio, max_wait_time, strategy,
                    stop_event
                )

                with lock:
                    completed_tasks += 1
                    if result is not None:
                        success_count += 1
                        result_queue.put((var_id, result))
            except Exception as e:
                with lock:
                    completed_tasks += 1
                    if not stop_event.is_set():
                        print(f"[组 {group_id}] 变体 {var_id} 失败: {e}", flush=True)
            finally:
                if completed_tasks >= total_tasks:
                    if group_id in self._stop_events:
                        del self._stop_events[group_id]

        threads = []
        for var_id, prompt, image_urls in variant_configs:
            t = threading.Thread(target=worker, args=(var_id, prompt, image_urls), daemon=True)
            t.start()
            threads.append(t)

        print(f"[组 {group_id}] 已启动 {len(variant_configs)} 个变体任务，目标成功数量: {target_success_count}", flush=True)

        successful_results = []
        try:
            while len(successful_results) < total_tasks:
                try:
                    var_id, result = result_queue.get(timeout=0.5)
                    successful_results.append((var_id, result))
                    print(f"[组 {group_id}] 变体 {var_id} 成功！当前成功数量: {len(successful_results)}/{total_tasks}", flush=True)
                except Empty:
                    with lock:
                        if completed_tasks >= total_tasks:
                            break
                    continue
                except Exception as e:
                    print(f"[组 {group_id}] 结果收集异常: {e}", flush=True)
                    stop_event.set()
        finally:
            if group_id in self._stop_events:
                del self._stop_events[group_id]

        successful_results.sort(key=lambda x: x[0])
        return [result for var_id, result in successful_results]

    def process_single_group_with_batch(self, group_id, image_inputs, prompt_list, batch_count,
                                      runninghub_api_key, banana_api_key,
                                      resolution, aspect_ratio, max_wait_time,
                                      strategy,
                                      oss_access_key_id, oss_access_key_secret, 
                                      oss_bucket_name, oss_endpoint,
                                      aspect_ratio_ref_image_index=1,
                                      target_success_count=1,
                                      output_format="保持原格式"):
        use_oss = all([
            oss_access_key_id.strip(),
            oss_access_key_secret.strip(),
            oss_bucket_name.strip()
        ])

        if use_oss:
            print(f"[组 {group_id}] 检测到阿里云OSS配置，使用OSS直接上传...", flush=True)
        else:
            if not runninghub_api_key.strip():
                raise ValueError(f"[组 {group_id}] 未配置阿里云OSS，且缺少 RunningHub API 密钥，无法上传参考图")
            print(f"[组 {group_id}] 未配置阿里云OSS，使用 RunningHub 自带上传...", flush=True)

        # 上传所有图片并建立映射
        image_urls_map = {}
        all_tensors = []
        for suffix, tensors in image_inputs.items():
            for idx, tensor in enumerate(tensors):
                all_tensors.append((suffix, idx, tensor))

        print(f"[组 {group_id}] 正在上传 {len(all_tensors)} 张参考图...", flush=True)
        for upload_idx, (suffix, idx, tensor) in enumerate(all_tensors, 1):
            try:
                pil_img = self.tensor_to_pil(tensor)
                if use_oss:
                    # ✅ 修改：如果配置了OSS，则直接上传至OSS，避免服务器中转
                    url = self.upload_to_aliyun_oss(
                        pil_img,
                        oss_access_key_id,
                        oss_access_key_secret,
                        oss_bucket_name,
                        oss_endpoint,
                        output_format  # 传递输出格式
                    )
                else:
                    # 否则通过RH中转上传
                    url = self.upload_to_runninghub(pil_img, runninghub_api_key, output_format)
                image_urls_map[(suffix, idx)] = url
                print(f"[组 {group_id}] 参考图 {suffix}[{idx}] 上传成功", flush=True)
            except Exception as e:
                print(f"[组 {group_id}] 跳过无效图像 {suffix}[{idx}]: {e}", flush=True)
                continue

        if not image_urls_map:
            raise RuntimeError(f"[组 {group_id}] 无有效参考图可上传")

        # 生成图片组合任务并过滤低分辨率
        max_image_count = max(len(tensors) for tensors in image_inputs.values())
        image_tasks = []

        for image_task_idx in range(max_image_count):
            task_image_urls = []
            task_valid = True

            for suffix in image_inputs:
                tensors = image_inputs[suffix]
                img_idx = min(image_task_idx, len(tensors)-1)
                tensor = tensors[img_idx]

                # 检查分辨率
                H, W = tensor.shape[1], tensor.shape[2]
                if H < 256 or W < 256:
                    print(f"[组 {group_id}] 图片组合任务 {image_task_idx+1} 跳过：{suffix}[{img_idx}] 分辨率 ({H}x{W}) 小于256x256", flush=True)
                    task_valid = False
                    break

                if (suffix, img_idx) not in image_urls_map:
                    print(f"[组 {group_id}] 图片组合任务 {image_task_idx+1} 跳过：{suffix}[{img_idx}] 上传失败", flush=True)
                    task_valid = False
                    break

                task_image_urls.append(image_urls_map[(suffix, img_idx)])
            if task_valid:
                image_tasks.append(task_image_urls)
                print(f"[组 {group_id}] 图片组合任务 {image_task_idx+1} 有效，使用 {len(task_image_urls)} 张图片", flush=True)

        if not image_tasks:
            raise RuntimeError(f"[组 {group_id}] 无有效的图片组合任务（所有任务包含分辨率<256x256的图片或上传失败）")

        # 处理宽高比
        effective_aspect_ratio = aspect_ratio
        if aspect_ratio == "自动":
            try:
                first_suffix = next(iter(image_inputs.keys()))
                first_tensors = image_inputs[first_suffix]
                first_img_idx = min(0, len(first_tensors)-1)
                ref_pil = self.tensor_to_pil(first_tensors[first_img_idx])
                inferred_ratio = self._infer_aspect_ratio_from_image(ref_pil)
                effective_aspect_ratio = inferred_ratio
                print(f"[组 {group_id}] 自动检测参考图比例为: {inferred_ratio} (参考图 {first_suffix}[{first_img_idx}], 尺寸 {ref_pil.size})", flush=True)
            except Exception as e:
                print(f"[组 {group_id}] 自动比例检测失败，使用默认 1:1: {e}", flush=True)
                effective_aspect_ratio = "1:1"
        else:
            effective_aspect_ratio = aspect_ratio if aspect_ratio != "自动" else "1:1"

        # 生成变体配置：图片组合 × 提示词行
        variant_configs = []
        var_id = 1
        for image_task in image_tasks:
            for prompt in prompt_list:
                variant_configs.append((var_id, prompt, image_task))
                var_id += 1

        print(f"[组 {group_id}] 参考图全部上传完成，开始生成 {len(variant_configs)} 个变体（{len(image_tasks)} 个图片组合 × {len(prompt_list)} 行提示词）", flush=True)

        # 执行所有任务
        successful_results = self._execute_variants_with_target_success(
            group_id, variant_configs, len(variant_configs),
            runninghub_api_key, banana_api_key,
            resolution, effective_aspect_ratio, max_wait_time, strategy
        )

        if not successful_results:
            print(f"[组 {group_id}] 所有变体均失败", flush=True)
            return torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        tensor_results = []
        for result in successful_results:
            if result:
                tensor_results.append(self.pil_to_tensor(result))

        if not tensor_results:
            return torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        _, H, W, C = tensor_results[0].shape
        aligned = []
        for t in tensor_results:
            if t.shape[1:] != (H, W, C):
                pil = self.tensor_to_pil(t)
                pil = pil.resize((W, H), Image.LANCZOS)
                t = self.pil_to_tensor(pil)
            aligned.append(t)

        final_output = torch.cat(aligned, dim=0)
        print(f"[组 {group_id}] 成功生成 {len(aligned)} 张图（{len(image_tasks)} 个图片组合 × {len(prompt_list)} 行提示词）", flush=True)
        return final_output

    def generate(self,
                 Seedream_v4_5_最大尝试次数,
                 V2_社区版_最大尝试次数,
                 V2_官方稳定版_最大尝试次数,
                 G2_社区版_最大尝试次数,
                 社区版_最大尝试次数,
                 全能Xinbao_最大尝试次数,
                 官方PRO版_最大尝试次数,
                 Seedream_v4_5_优先级,
                 V2_社区版_优先级,
                 V2_官方稳定版_优先级,
                 G2_社区版_优先级,
                 社区版_优先级,
                 全能Xinbao_优先级,
                 官方PRO版_优先级,
                 runninghub_api_key, 
                 全能Xinbao_api_key,
                 oss_access_key_id,
                 oss_access_key_secret,
                 oss_bucket_name,
                 oss_endpoint,
                 resolution, 
                 aspect_ratio, 
                 aspect_ratio_ref_image_index, 
                 seed, 
                 global_concurrent_tasks, 
                 max_wait_time, 
                 全局任务总数,
                 max_prompt_lines_global,
                 每组成功数量,
                 output_format,
                 **kwargs):

        strategy = self._build_strategy_by_priority(
            int(Seedream_v4_5_最大尝试次数),
            int(V2_社区版_最大尝试次数),
            int(V2_官方稳定版_最大尝试次数),
            int(G2_社区版_最大尝试次数),
            int(社区版_最大尝试次数),
            int(全能Xinbao_最大尝试次数),
            int(官方PRO版_最大尝试次数),
            int(Seedream_v4_5_优先级),
            int(V2_社区版_优先级),
            int(V2_官方稳定版_优先级),
            int(G2_社区版_优先级),
            int(社区版_优先级),
            int(全能Xinbao_优先级),
            int(官方PRO版_优先级)
        )

        need_runninghub = any(step["type"] in ["community", "official", "g31_flash", "g31_flash_official", "seedream_v4_5", "g2_community"] for step in strategy)
        need_xinbao = any(step["type"] == "xinbao" for step in strategy)

        if need_runninghub and not runninghub_api_key.strip():
            raise ValueError("当前策略需要 RunningHub API 密钥，请填写")
        if need_xinbao and not 全能Xinbao_api_key.strip():
            raise ValueError("当前策略包含「全能Xinbao」，请填写其 API 密钥")

        use_oss = all([
            oss_access_key_id.strip(),
            oss_access_key_secret.strip(),
            oss_bucket_name.strip()
        ])

        if use_oss and not OSS_AVAILABLE:
            raise ValueError("请安装 oss2: pip install oss2")

        global_concurrent_tasks = min(max(1, int(global_concurrent_tasks)), 10)
        max_wait_time = min(max(30, int(max_wait_time)), 600)
        每组成功数量 = min(max(1, int(每组成功数量)), 10)
        max_prompt_lines_global = int(max_prompt_lines_global)
        if max_prompt_lines_global == 0:
            max_prompt_lines_global = -1

        skipped_placeholder = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        results = [skipped_placeholder] * 10

        valid_tasks = []
        for i in range(1, 11):
            raw_prompt = kwargs.get(f"prompt_{i}", "").strip()
            prompt_lines = []
            if raw_prompt:
                if max_prompt_lines_global == 1:
                    prompt_lines = [raw_prompt.replace('\r\n', '\n').replace('\r', '\n')]
                else:
                    prompt_lines = [line.strip() for line in raw_prompt.split('\n') if line.strip()]

            if max_prompt_lines_global > 1 and len(prompt_lines) > max_prompt_lines_global:
                original_len = len(prompt_lines)
                prompt_lines = prompt_lines[:max_prompt_lines_global]
                group_letter = chr(ord('A') + i - 1)
                print(f"[组 {group_letter}] 提示词行数被全局限制为 {len(prompt_lines)} 行（max_prompt_lines_global={max_prompt_lines_global}）", flush=True)

            if not prompt_lines:
                continue

            # 收集每个输入点的图片列表
            image_inputs = {}
            group_letter = chr(ord('A') + i - 1)
            for suffix in ['a', 'b', 'c', 'd', 'e']:
                img = kwargs.get(f"image_{group_letter}_{suffix}")
                if img is not None and img.shape[0] > 0:
                    image_inputs[suffix] = [img[b:b+1] for b in range(img.shape[0])]

            if not image_inputs:
                continue

            effective_batch_count = len(prompt_lines)
            if effective_batch_count < 1:
                effective_batch_count = 1

            valid_tasks.append((i - 1, group_letter, image_inputs, prompt_lines, effective_batch_count))

        if not valid_tasks:
            raise ValueError("至少需要一组有效的（提示词 + 至少1张参考图）")

        valid_tasks = valid_tasks[:global_concurrent_tasks]
        print(f"▶ 仅处理前 {len(valid_tasks)} 个有效组（受 global_concurrent_tasks={global_concurrent_tasks} 限制）", flush=True)

        # === 全局任务总数限制处理 ===
        # 计算每个组的任务数：图片组合数 × 提示词行数
        # 图片组合数 = max(各接口图片数量)
        group_task_info = []
        total_task_count = 0
        for out_idx, group_id, image_inputs, prompt_lines, batch_count in valid_tasks:
            # 计算图片组合数（每组取图片数量最多的那个接口）
            max_images = max(len(tensors) for tensors in image_inputs.values()) if image_inputs else 1
            # 该组任务数 = 图片组合数 × 提示词行数
            group_tasks = max_images * len(prompt_lines)
            group_task_info.append({
                "out_idx": out_idx,
                "group_id": group_id,
                "image_inputs": image_inputs,
                "prompt_lines": prompt_lines,
                "batch_count": batch_count,
                "max_images": max_images,
                "group_tasks": group_tasks
            })
            total_task_count += group_tasks
        
        # 如果设置了全局任务总数限制，则重新分配任务
        if 全局任务总数 > 0 and total_task_count > 全局任务总数:
            remaining_slots = 全局任务总数
            final_task_info = []
            
            for info in group_task_info:
                if remaining_slots <= 0:
                    # 没有剩余任务槽，清空该组
                    info["prompt_lines"] = []
                    info["image_inputs"] = {}
                    info["group_tasks"] = 0
                    print(f"▶ 组 {info['group_id']} 已被全局任务限制跳过（剩余任务槽: 0）", flush=True)
                else:
                    prompts = len(info["prompt_lines"])
                    max_images = info["max_images"]
                    
                    # 计算该组需要的任务槽
                    needed_slots = info["group_tasks"]
                    
                    if needed_slots <= remaining_slots:
                        # 该组可以完整运行
                        remaining_slots -= needed_slots
                    else:
                        # 该组需要裁剪
                        if prompts > 0:
                            # 先尝试保留所有提示词，看需要裁剪多少图片
                            allowed_image_count = remaining_slots // prompts
                            if allowed_image_count >= max_images:
                                # 理论上不应该到这里，因为 needed_slots > remaining_slots
                                # 但为了安全，直接减去剩余槽位
                                remaining_slots -= needed_slots
                            elif allowed_image_count >= 1:
                                # 可以保留所有提示词，但需要裁剪图片数量
                                cropped_image_inputs = {}
                                for suffix, tensors in info["image_inputs"].items():
                                    cropped_image_inputs[suffix] = tensors[:allowed_image_count]
                                
                                old_image_count = max_images
                                info["image_inputs"] = cropped_image_inputs
                                info["group_tasks"] = allowed_image_count * prompts
                                remaining_slots -= info["group_tasks"]
                                print(f"▶ 组 {info['group_id']} 图片组合数从 {old_image_count} 裁剪至 {allowed_image_count}（提示词{prompts}行保留，全局任务限制: {全局任务总数}）", flush=True)
                            else:
                                # allowed_image_count < 1，意味着即使1张图片，提示词行数也超过了remaining_slots
                                # 优先裁剪提示词行数（全局任务总数优先级最高）
                                allowed_image_count = 1  # 至少保留1张图片
                                max_allowed_prompts = remaining_slots  # 1张图片时，最多 remaining_slots 行提示词
                                
                                # 裁剪提示词
                                old_prompt_count = prompts
                                info["prompt_lines"] = info["prompt_lines"][:max_allowed_prompts]
                                
                                # 同时也只保留1张图片
                                cropped_image_inputs = {}
                                for suffix, tensors in info["image_inputs"].items():
                                    cropped_image_inputs[suffix] = tensors[:allowed_image_count]
                                info["image_inputs"] = cropped_image_inputs
                                
                                info["group_tasks"] = allowed_image_count * max_allowed_prompts
                                remaining_slots -= info["group_tasks"]
                                print(f"▶ 组 {info['group_id']} 提示词行数从 {old_prompt_count} 裁剪至 {max_allowed_prompts}，图片保留 {allowed_image_count} 张（全局任务限制: {全局任务总数}）", flush=True)
                        else:
                            remaining_slots = 0
                
                final_task_info.append(info)
            
            # 重建 valid_tasks
            valid_tasks = []
            for info in final_task_info:
                valid_tasks.append((
                    info["out_idx"], info["group_id"], info["image_inputs"],
                    info["prompt_lines"], info["batch_count"]
                ))

        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=len(valid_tasks)) as executor:
            futures = {}
            for out_idx, group_id, image_inputs, prompt_lines, batch_count in valid_tasks:
                future = executor.submit(
                    self.process_single_group_with_batch,
                    group_id, image_inputs, prompt_lines, batch_count,
                    runninghub_api_key, 全能Xinbao_api_key,
                    resolution, aspect_ratio, max_wait_time,
                    strategy,
                    oss_access_key_id, oss_access_key_secret, oss_bucket_name, oss_endpoint,
                    aspect_ratio_ref_image_index,
                    每组成功数量,
                    output_format  # 传递输出格式
                )
                futures[future] = out_idx

            for future in as_completed(futures):
                out_idx = futures[future]
                try:
                    results[out_idx] = future.result()
                except Exception as e:
                    print(f"⚠️ 组 {chr(ord('A') + out_idx)} 整体失败: {e}", flush=True)

        all_real_images = []
        for img_tensor in results:
            if img_tensor.shape[1] > 64:
                all_real_images.append(img_tensor)

        if all_real_images:
            all_success_output = torch.cat(all_real_images, dim=0)
        else:
            all_success_output = torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        return tuple(results) + (all_success_output,)


NODE_CLASS_MAPPINGS = {
    "RunningHubRhartImageToImageAlioss": RunningHubRhartImageToImageAlioss
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RunningHubRhartImageToImageAlioss": "YK-影客AI-RUNHUB全能图片阿里OSS"
}