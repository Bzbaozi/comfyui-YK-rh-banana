# YK_oss_uploader.py
# 影客-OSS上传分支节点：接上即自动把图像上传到阿里云OSS，不接不影响任何流程
import time
import random
from io import BytesIO

import numpy as np
import torch
from PIL import Image

# 尝试导入 oss2（按需）
try:
    import oss2
    OSS_AVAILABLE = True
except ImportError:
    OSS_AVAILABLE = False


class YKOSSImageUploader:
    """分支上传节点：把上游 IMAGE 批量上传到阿里云 OSS，并透传图像与返回 URL 列表。

    - OUTPUT_NODE = True：只要连线就会执行（同 SaveImage），无需下游消费输出
    - 支持批量：上游 batch 中每张图独立上传
    - 输出：原样透传 IMAGE + 换行分隔的 OSS 链接（STRING），方便下游继续接或记录
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "oss_access_key_id": ("STRING", {"default": "", "placeholder": "阿里云 AccessKey ID"}),
                "oss_access_key_secret": ("STRING", {"default": "", "placeholder": "阿里云 AccessKey Secret", "password": True}),
                "oss_bucket_name": ("STRING", {"default": "", "placeholder": "OSS Bucket 名称"}),
                "oss_endpoint": ("STRING", {"default": "oss-cn-hangzhou.aliyuncs.com", "placeholder": "OSS Endpoint 地址"}),
                "oss_dir": ("STRING", {
                    "default": "comfyui_rhart/results/",
                    "tooltip": "上传到 Bucket 内的目录前缀，以 / 结尾"
                }),
                "output_format": (["JPEG", "PNG", "WEBP"], {
                    "default": "JPEG",
                    "tooltip": "上传到OSS的图片格式"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("图像", "OSS链接")
    FUNCTION = "upload"
    OUTPUT_NODE = True
    CATEGORY = "影客AI"

    # --- 工具方法 ---
    def tensor_to_pil(self, tensor):
        """将 PyTorch Tensor 转换为 PIL 图像"""
        if tensor.ndim == 4:
            tensor = tensor.squeeze(0)
        i = 255. * tensor.cpu().numpy()
        return Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

    def _upload_pil_to_oss(self, pil_img, bucket, bucket_name, endpoint, oss_dir, img_format):
        """上传单张 PIL 图像到 OSS，返回公开 URL"""
        img_format = img_format.upper()
        if img_format not in ["PNG", "JPEG", "JPG", "WEBP"]:
            img_format = "JPEG"

        # JPEG 不支持透明通道
        if img_format in ("JPEG", "JPG") and pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        # 目录前缀规范化
        oss_dir = oss_dir.strip().lstrip("/")
        if oss_dir and not oss_dir.endswith("/"):
            oss_dir += "/"

        timestamp = str(int(time.time() * 1000))
        random_suffix = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=8))
        object_key = f"{oss_dir}{timestamp}_{random_suffix}.{img_format.lower()}"

        buf = BytesIO()
        pil_img.save(buf, format=img_format)
        buf.seek(0)

        content_type_map = {
            "PNG": "image/png",
            "JPEG": "image/jpeg",
            "JPG": "image/jpeg",
            "WEBP": "image/webp"
        }
        content_type = content_type_map.get(img_format, "image/jpeg")

        bucket.put_object(object_key, buf.getvalue(), headers={'Content-Type': content_type})
        return f"https://{bucket_name}.{endpoint}/{object_key}"

    def upload(self, images, oss_access_key_id, oss_access_key_secret,
               oss_bucket_name, oss_endpoint, oss_dir, output_format):
        if not OSS_AVAILABLE:
            raise RuntimeError("未安装 oss2 库，请运行: pip install oss2")
        if not all([oss_access_key_id.strip(), oss_access_key_secret.strip(), oss_bucket_name.strip()]):
            raise ValueError("请填写完整的阿里云 OSS 配置信息（AK ID / AK Secret / Bucket）")

        bucket_name = oss_bucket_name.strip()
        endpoint = oss_endpoint.strip()
        auth = oss2.Auth(oss_access_key_id.strip(), oss_access_key_secret.strip())
        bucket = oss2.Bucket(auth, f'https://{endpoint}', bucket_name)

        # 批量上传：batch 中每张图独立上传
        batch_size = images.shape[0] if images.ndim == 4 else 1
        urls = []
        print(f"[OSS上传] 开始上传 {batch_size} 张图片到 {bucket_name}/{oss_dir.strip()}", flush=True)
        for idx in range(batch_size):
            single = images[idx] if images.ndim == 4 else images
            pil_img = self.tensor_to_pil(single)
            try:
                url = self._upload_pil_to_oss(pil_img, bucket, bucket_name, endpoint, oss_dir, output_format)
            except Exception as e:
                raise RuntimeError(f"[OSS上传] 第 {idx + 1}/{batch_size} 张上传失败: {e}")
            urls.append(url)
            print(f"[OSS上传] {idx + 1}/{batch_size} 上传成功: {url}", flush=True)

        return (images, "\n".join(urls))


NODE_CLASS_MAPPINGS = {
    "YKOSSImageUploader": YKOSSImageUploader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YKOSSImageUploader": "YK-OSS-图片上传"
}
