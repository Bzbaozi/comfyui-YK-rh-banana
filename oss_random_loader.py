import torch
import numpy as np
from PIL import Image
import random
import oss2
from io import BytesIO

class OSSRandomImageLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "access_key_id": ("STRING", {"default": "", "multiline": False}),
                "access_key_secret": ("STRING", {"default": "", "multiline": False, "password": True}),
                "bucket_name": ("STRING", {"default": "my-bucket", "multiline": False}),
                "region": ("STRING", {"default": "cn-hangzhou", "multiline": False}),
                "prefix": ("STRING", {"default": "images/", "multiline": False}),
                "batch_size": ("INT", {"default": 1, "min": 0, "max": 15}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "resize_mode": (["pad", "crop_width"], {"default": "pad"}),
                "target_width": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "target_height": ("INT", {"default": 512, "min": 64, "max": 2048}),
            },
            "optional": {
                "allowed_extensions": ("STRING", {"default": "jpg,jpeg,png", "multiline": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "filename")
    FUNCTION = "load_random_image_from_oss"
    CATEGORY = "image"

    def _is_image_file(self, filename: str, allowed_exts: str) -> bool:
        if not allowed_exts:
            allowed_exts = "jpg,jpeg,png,gif,bmp,webp"
        ext_list = [f".{ext.strip().lower()}" for ext in allowed_exts.split(",")]
        name_lower = filename.lower()
        return any(name_lower.endswith(ext) for ext in ext_list)

    def _pad_to_size(self, img, target_w, target_h):
        """等比缩放图像以完全适应目标区域（不裁剪），然后居中白色填充"""
        w, h = img.size
        if w == 0 or h == 0:
            return Image.new("RGB", (target_w, target_h), (255, 255, 255))
        
        # 计算缩放比例，确保图像完全 fit inside 目标框
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # 高质量缩放
        img = img.resize((new_w, new_h), Image.LANCZOS)
        
        # 创建白色背景画布
        new_img = Image.new("RGB", (target_w, target_h), (255, 255, 255))
        
        # 居中粘贴
        paste_x = (target_w - new_w) // 2
        paste_y = (target_h - new_h) // 2
        new_img.paste(img, (paste_x, paste_y))
        
        return new_img

    def _crop_by_width(self, img, target_w, target_h):
        w, h = img.size
        if w == 0:
            return Image.new("RGB", (target_w, target_h), (255, 255, 255))
        scale = target_w / w
        new_h = int(h * scale)
        img = img.resize((target_w, new_h), Image.LANCZOS)

        if new_h > target_h:
            top = (new_h - target_h) // 2
            img = img.crop((0, top, target_w, top + target_h))
        elif new_h < target_h:
            new_img = Image.new("RGB", (target_w, target_h), (255, 255, 255))
            paste_y = (target_h - new_h) // 2
            new_img.paste(img, (0, paste_y))
            img = new_img
        return img

    def load_random_image_from_oss(
        self,
        access_key_id: str,
        access_key_secret: str,
        bucket_name: str,
        region: str,
        prefix: str,
        batch_size: int,
        seed: int,
        resize_mode: str,
        target_width: int,
        target_height: int,
        allowed_extensions: str = "jpg,jpeg,png"
    ):
        # ✅ batch_size=0 时立即返回，不访问 OSS
        if batch_size == 0:
            placeholder = torch.ones((1, 1, 1, 3), dtype=torch.float32)
            return (placeholder, "[SKIPPED: batch_size=0]")

        random.seed(seed)

        if not access_key_id or not access_key_secret or not bucket_name:
            raise ValueError("AccessKey ID, AccessKey Secret, and Bucket Name are required.")

        try:
            auth = oss2.Auth(access_key_id.strip(), access_key_secret.strip())
            endpoint = f"https://oss-{region.strip()}.aliyuncs.com"
            bucket = oss2.Bucket(auth, endpoint, bucket_name.strip())

            image_keys = []
            for obj in oss2.ObjectIterator(bucket, prefix=prefix):
                if self._is_image_file(obj.key, allowed_extensions):
                    image_keys.append(obj.key)

            if not image_keys:
                raise FileNotFoundError(f"No image files found under prefix: '{prefix}'")

            selected_count = min(batch_size, len(image_keys))
            selected_keys = random.sample(image_keys, selected_count)
            print(f"[OSS Random Loader] Selected {selected_count} images from OSS")

            image_tensors = []
            for key in selected_keys:
                img_data = bucket.get_object(key).read()
                image = Image.open(BytesIO(img_data)).convert("RGB")

                if resize_mode == "pad":
                    image = self._pad_to_size(image, target_width, target_height)
                elif resize_mode == "crop_width":
                    image = self._crop_by_width(image, target_width, target_height)

                img_array = np.array(image).astype(np.float32) / 255.0
                image_tensors.append(torch.from_numpy(img_array))

            batch_tensor = torch.stack(image_tensors, dim=0)
            filenames_str = ",".join(selected_keys)
            return (batch_tensor, filenames_str)

        except Exception as e:
            error_msg = f"ERROR: {str(e)}"
            print(f"[OSS Random Loader] {error_msg}")
            fallback = torch.ones((1, target_height, target_width, 3), dtype=torch.float32)
            return (fallback, error_msg)

# ✅ 添加这两个必需的映射变量
NODE_CLASS_MAPPINGS = {
    "OSSRandomImageLoader": OSSRandomImageLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OSSRandomImageLoader": "OSS Random Image Loader"
}