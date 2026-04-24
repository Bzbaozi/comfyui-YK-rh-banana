# YK_Vision_ActionPrompt.py - Banana2 图像编辑专用提示词生成器（角度选择版）
# 核心目标：UI直接设置每个角度的数量（0=不生成），生成固定简洁格式的编辑提示词
import torch
import numpy as np
from PIL import Image
import base64
from io import BytesIO

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("警告：未安装openai库，API功能将不可用。请运行 'pip install openai'")


class YK_Vision_ActionPrompt_v2:
    # 品类 → 显示名称映射
    GARMENT_DISPLAY_MAP = {
        # 上装
        "上衣": "上衣",
        "T恤": "T恤",
        "衬衫": "衬衫",
        "卫衣": "卫衣",
        "背心": "背心",
        "毛衣": "毛衣",
        "针织衫": "针织衫",
        "毛衣/针织衫": "毛衣",
        # 外套
        "外套": "外套",
        "夹克": "夹克",
        "西装": "西装",
        "风衣": "风衣",
        "大衣": "大衣",
        "棉服/羽绒服": "棉服",
        # 裙装
        "裙子": "裙子",
        "短裙": "短裙",
        "长裙": "长裙",
        "连衣裙": "连衣裙",
        # 裤装
        "裤子": "裤子",
        "短裤": "短裤",
        "长裤": "长裤",
        "牛仔裤": "牛仔裤",
        "西装裤": "西装裤",
    }

    # 角度配置：(参数名, 显示名, 默认数量)
    ANGLE_CONFIG = [
        ("正面数量", "正面", 1),
        ("左侧面数量", "左侧面", 0),
        ("右侧面数量", "右侧面", 0),
        ("背面数量", "背面", 0),
        ("正面坐姿数量", "正面坐姿", 0),
        ("左侧坐姿数量", "左侧坐姿", 0),
        ("右侧坐姿数量", "右侧坐姿", 0),
    ]

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "图像": ("IMAGE",),
            "使用API": ("BOOLEAN", {"default": True}),
            "API密钥": ("STRING", {"default": "", "multiline": False}),
            "API模型": ("STRING", {"default": "qwen-vl-plus", "multiline": False}),
            "展示品类": ([
                "自动识别",
                # 上装
                "T恤", "衬衫", "卫衣", "背心/吊带",
                "毛衣", "针织衫", "毛衣/针织衫",
                # 外套
                "西装", "夹克", "风衣/大衣", "棉服/羽绒服",
                # 裙装
                "连衣裙", "短裙", "长裙", "裙子",
                # 裤装
                "短裤", "长裤", "牛仔裤", "西装裤",
                "裤子",
            ], {"default": "自动识别"}),
            "插口袋动作": ([
                "自动识别",
                "有口袋",
                "没有口袋"
            ], {"default": "自动识别"}),
            "细节数量": ("INT", {"default": 0, "min": 0, "max": 10}),
            "任务总数": ("INT", {"default": 1, "min": 1, "max": 100}),
        }

        # 动态添加角度数量参数（0=不生成）
        for param_name, display_name, default in cls.ANGLE_CONFIG:
            required[param_name] = ("INT", {"default": default, "min": 0, "max": 20})

        required["随机种子"] = ("INT", {"default": 0, "min": 0, "max": 0xffffffff})

        return {
            "required": required,
            "optional": {
                "自定义全局指令": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("编辑提示词", "分析日志")
    FUNCTION = "生成编辑提示词"
    CATEGORY = "YK/视觉理解"
    DESCRIPTION = "Banana2专用：设置各角度数量（0=不生成），固定简洁提示词格式"

    def __init__(self):
        self.console_log_messages = []

    def console_log(self, message):
        print(message)
        self.console_log_messages.append(message)

    def tensor_to_pil(self, image_tensor):
        if image_tensor.ndim == 4:
            image_tensor = image_tensor[0]
        image_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
        if image_np.shape[2] == 3:
            return Image.fromarray(image_np, 'RGB')
        elif image_np.shape[2] == 4:
            return Image.fromarray(image_np, 'RGBA')
        else:
            raise ValueError("不支持的通道数")

    def pil_to_base64(self, pil_image):
        buffered = BytesIO()
        if pil_image.mode == 'RGBA':
            background = Image.new('RGB', pil_image.size, (255, 255, 255))
            background.paste(pil_image, mask=pil_image.split()[-1])
            pil_image = background
        pil_image.save(buffered, format="JPEG", quality=95)
        return base64.b64encode(buffered.getvalue()).decode()

    def _detect_garment_type(self, pil_img, api_key, use_api, api_model):
        """
        快速识别图中主要展示的服装品类
        返回：品类名称（裙子/上衣/裤子/连衣裙/外套/毛衣）
        """
        if not (use_api and api_key.strip() and OPENAI_AVAILABLE):
            self.console_log("⚠️ API未启用，使用默认品类：上衣")
            return "上衣"

        try:
            image_base64 = self.pil_to_base64(pil_img)
            client = OpenAI(
                api_key=api_key.strip(),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )

            prompt = (
                "你是一个电商服装识别专家。请观察图片，判断图中主要展示的是什么服装品类。"
                "只需从以下选项中选择一个回答："
                "T恤、衬衫、卫衣、背心、毛衣、针织衫、西装、夹克、风衣、大衣、棉服、羽绒服、"
                "连衣裙、短裙、长裙、短裤、长裤、牛仔裤、西装裤、外套。"
                "只回答一个词，不要任何解释。"
            )

            self.console_log("🔍 识别图中主要服装品类...")
            response = client.chat.completions.create(
                model=api_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                        {"type": "text", "text": prompt}
                    ]
                }],
                temperature=0.1,
                max_tokens=16
            )
            result = response.choices[0].message.content.strip()
            self.console_log(f"📥 识别结果: {result}")

            # 映射到标准品类名（顺序：先精确匹配，再模糊匹配）
            result_lower = result.lower()
            if "连衣" in result and "裙" in result:
                return "连衣裙"
            elif "短" in result and "裙" in result:
                return "短裙"
            elif "裙" in result:
                return "长裙"
            elif "短" in result and "裤" in result:
                return "短裤"
            elif "牛仔" in result and "裤" in result:
                return "牛仔裤"
            elif "西装" in result and "裤" in result:
                return "西装裤"
            elif "裤" in result:
                return "长裤"
            elif "风衣" in result or "大衣" in result:
                return "风衣/大衣"
            elif "棉服" in result or "羽绒" in result or "棉袄" in result:
                return "棉服/羽绒服"
            elif "西装" in result or "西服" in result:
                return "西装"
            elif "夹克" in result:
                return "夹克"
            elif "外套" in result:
                return "外套"
            elif "毛衣" in result or "针织" in result or "毛衫" in result:
                return "毛衣/针织衫"
            elif "背心" in result or "吊带" in result:
                return "背心"
            elif "卫衣" in result:
                return "卫衣"
            elif "衬衫" in result:
                return "衬衫"
            elif "t恤" in result_lower or "T恤" in result or "短袖" in result:
                return "T恤"
            elif "上衣" in result:
                return "上衣"
            else:
                self.console_log(f"⚠️ 无法精确匹配，使用原始结果: {result}")
                return result if result in self.GARMENT_DISPLAY_MAP else "上衣"

        except Exception as e:
            self.console_log(f"⚠️ 品类识别失败: {e}，使用默认：上衣")
            return "上衣"

    def _detect_pocket(self, pil_img, api_key, use_api, api_model):
        """
        识别图中服装是否有口袋
        返回：True=有口袋，False=没有口袋
        """
        if not (use_api and api_key.strip() and OPENAI_AVAILABLE):
            self.console_log("⚠️ API未启用，无法识别口袋")
            return False

        try:
            image_base64 = self.pil_to_base64(pil_img)
            client = OpenAI(
                api_key=api_key.strip(),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )

            prompt = (
                "请仔细观察图片中人物穿着的服装，判断这件衣服是否有口袋"
                "（如裤袋、衣袋、裙袋、外套口袋等任何类型的口袋都算）。"
                "只需回答'有'或'没有'，不要任何解释。"
            )

            self.console_log("🔍 识别服装是否有口袋...")
            response = client.chat.completions.create(
                model=api_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                        {"type": "text", "text": prompt}
                    ]
                }],
                temperature=0.1,
                max_tokens=8
            )
            result = response.choices[0].message.content.strip()
            self.console_log(f"📥 口袋识别结果: {result}")

            # 判断是否有口袋
            if "有" in result and "没" not in result and "无" not in result:
                return True
            return False

        except Exception as e:
            self.console_log(f"⚠️ 口袋识别失败: {e}")
            return False

    def _detect_garment_details(self, pil_img, api_key, use_api, api_model, garment_type):
        """
        识别图中指定服装的可展示细节元素
        返回：细节元素列表，如 ['领口', '袖子', '刺绣']
        """
        if not (use_api and api_key.strip() and OPENAI_AVAILABLE):
            self.console_log("⚠️ API未启用，使用默认细节元素")
            return self._get_default_details(garment_type)

        try:
            image_base64 = self.pil_to_base64(pil_img)
            client = OpenAI(
                api_key=api_key.strip(),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )

            # 针对腰部优先的品类，增加明确引导
            waist_hint = ""
            if garment_type in ("裙子", "裤子", "连衣裙"):
                waist_hint = (
                    f"特别注意：请首先检查这件{garment_type}的腰部区域，"
                    f"判断是否有腰带、松紧腰、腰头设计、腰部装饰等可展示的腰部细节，如果有请务必列出。"
                )

            prompt = (
                f"你是一个服装细节分析专家。请仔细观察图片中人物穿着的{garment_type}，"
                f"分析它有哪些可供特写展示的细节元素。"
                f"常见细节包括：领口、袖口、袖子、口袋、衣摆、裙摆、腰部、腰带、腰头、拉链、扣子、纽扣、"
                f"刺绣、LOGO、印花、图案、面料纹理、褶皱、蕾丝、镂空、绑带、蝴蝶结等。"
                f"{waist_hint}"
                f"注意：如果这件{garment_type}是无袖款式（如马甲、吊带、无袖背心等），"
                f"则不要列出袖子、袖口相关细节。"
                f"请只列出图中实际清晰可见的细节元素，不要编造，不要推测。"
                f"用逗号分隔列出，不要编号，不要解释。"
            )

            self.console_log(f"🔍 识别{garment_type}的可展示细节元素...")
            response = client.chat.completions.create(
                model=api_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                        {"type": "text", "text": prompt}
                    ]
                }],
                temperature=0.2,
                max_tokens=64
            )
            result = response.choices[0].message.content.strip()
            self.console_log(f"📥 细节识别结果: {result}")

            # 解析逗号分隔的细节列表
            import re
            details = [d.strip() for d in re.split(r'[,，、]', result) if d.strip()]
            # 过滤无效项
            details = [d for d in details if len(d) > 1 and not re.match(r'^\d+', d)]

            if not details:
                self.console_log("⚠️ 未识别到有效细节，使用默认细节")
                return self._get_default_details(garment_type)

            # 对腰部优先品类：若识别结果中没有腰部相关细节，强制在头部补充"腰部"
            if garment_type in ("裙子", "裤子", "连衣裙"):
                has_waist = any("腰" in d for d in details)
                if not has_waist:
                    self.console_log("ℹ️ 未识别到腰部细节，自动补充腰部到优先位置")
                    details = ["腰部"] + details

            self.console_log(f"✅ 识别到 {len(details)} 个细节: {', '.join(details)}")
            return details

        except Exception as e:
            self.console_log(f"⚠️ 细节识别失败: {e}，使用默认细节")
            return self._get_default_details(garment_type)

    def _get_default_details(self, garment_type):
        """根据品类返回默认细节元素（无袖款可能没有袖子，故放在后面兜底）"""
        defaults = {
            "上衣": ["领口", "衣摆", "肩部", "面料纹理", "袖子"],
            "裙子": ["腰部", "裙摆", "面料纹理", "褶皱"],
            "裤子": ["腰部", "裤腿", "面料纹理", "口袋"],
            "连衣裙": ["腰部", "领口", "裙摆", "面料纹理", "袖子"],
            "外套": ["领口", "口袋", "袖口", "面料纹理", "拉链"],
            "毛衣": ["领口", "袖口", "针织纹理", "衣摆"],
        }
        # 具体品类映射到通用配置
        if garment_type in defaults:
            return defaults[garment_type]
        if any(kw in garment_type for kw in ["T恤", "衬衫", "卫衣", "背心", "吊带"]):
            return defaults["上衣"]
        if any(kw in garment_type for kw in ["短裙", "长裙", "裙子"]):
            return defaults["裙子"]
        if any(kw in garment_type for kw in ["短裤", "长裤", "牛仔裤", "西装裤", "裤子"]):
            return defaults["裤子"]
        if any(kw in garment_type for kw in ["风衣", "大衣", "棉服", "羽绒", "夹克", "西装", "外套"]):
            return defaults["外套"]
        if any(kw in garment_type for kw in ["毛衣", "针织", "毛衫"]):
            return defaults["毛衣"]
        return defaults.get("上衣", ["面料纹理", "领口", "整体做工"])

    def _sort_details_by_priority(self, garment_type, detail_elements):
        """
        根据品类优先级排序细节元素
        上衣优先：领口、衣摆、肩部；裙子/裤子优先：腰部
        返回：排序后的细节列表
        """
        priority_map = {
            "上衣": ["领口", "衣摆", "肩部", "肩线", "袖子", "袖口", "面料", "刺绣", "印花", "LOGO", "图案"],
            "裙子": ["腰部", "裙摆", "裙身", "褶皱", "蕾丝", "面料", "印花", "刺绣", "绑带", "蝴蝶结"],
            "裤子": ["腰部", "裤腿", "裤脚", "口袋", "褶皱", "面料", "拉链", "刺绣"],
            "连衣裙": ["腰部", "领口", "裙摆", "袖子", "面料", "褶皱", "蕾丝", "印花"],
            "外套": ["领口", "口袋", "袖口", "拉链", "面料", "纽扣", "刺绣", "LOGO"],
            "毛衣": ["领口", "袖口", "针织", "衣摆", "面料", "花纹", "刺绣"],
        }

        # 具体品类映射到通用配置
        base_type = garment_type
        if garment_type in priority_map:
            base_type = garment_type
        elif any(kw in garment_type for kw in ["T恤", "衬衫", "卫衣", "背心", "吊带", "上衣"]):
            base_type = "上衣"
        elif any(kw in garment_type for kw in ["短裙", "长裙", "裙子"]):
            base_type = "裙子"
        elif any(kw in garment_type for kw in ["短裤", "长裤", "牛仔裤", "西装裤", "裤子"]):
            base_type = "裤子"
        elif any(kw in garment_type for kw in ["风衣", "大衣", "棉服", "羽绒", "夹克", "西装", "外套"]):
            base_type = "外套"
        elif any(kw in garment_type for kw in ["毛衣", "针织", "毛衫"]):
            base_type = "毛衣"
        elif "连衣裙" in garment_type:
            base_type = "连衣裙"

        priorities = priority_map.get(base_type, [])

        def get_priority(detail):
            for i, keyword in enumerate(priorities):
                if keyword in detail:
                    return i
            return len(priorities)

        return sorted(detail_elements, key=get_priority)

    def _generate_detail_prompts(self, garment_type, detail_elements, detail_count):
        """
        生成细节特写提示词（去重+优先级排序）
        返回：提示词列表
        """
        if not detail_elements or detail_count <= 0:
            return []

        # 保持顺序去重
        seen = set()
        unique_elements = []
        for d in detail_elements:
            if d not in seen:
                seen.add(d)
                unique_elements.append(d)

        # 按品类优先级排序
        sorted_details = self._sort_details_by_priority(garment_type, unique_elements)

        # 最多生成不重复的数量，避免重复提示词
        actual_count = min(detail_count, len(sorted_details))
        if actual_count < detail_count:
            self.console_log(
                f"⚠️ 识别到 {len(sorted_details)} 个不同细节，"
                f"已去重处理为 {actual_count} 条（避免重复提示词）"
            )

        prompts = []
        for i in range(actual_count):
            detail = sorted_details[i]
            prompt = f"展示{garment_type}的{detail}细节特写，镜头拉近，微距"
            prompts.append(prompt)

        return prompts

    def _generate_action_descriptions(self, pil_img, api_key, use_api, api_model,
                                      angle_list, custom_instruction,
                                      use_pocket_action=False, has_pocket=False,
                                      seed=0):
        """
        根据角度列表生成对应的动作描述
        angle_list: [(角度名, 同角度序号), ...]
        use_pocket_action: 用户是否选择了插口袋动作
        has_pocket: AI识别结果，服装是否有口袋
        返回：动作描述列表
        """
        self.console_log(f"📝 开始为 {len(angle_list)} 个角度生成动作描述...")

        # 口袋提示信息
        if use_pocket_action:
            if has_pocket:
                self.console_log("✅ 识别到服装有口袋，插口袋动作将生效")
            else:
                self.console_log("⚠️ 识别到服装没有口袋，插口袋动作不生效")

        # API模式下让AI根据角度生成具体动作细节
        if use_api and api_key.strip() and OPENAI_AVAILABLE and len(angle_list) > 0:
            try:
                image_base64 = self.pil_to_base64(pil_img)
                client = OpenAI(
                    api_key=api_key.strip(),
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
                )

                # 构建角度清单文本
                angle_lines = []
                for i, (angle, idx) in enumerate(angle_list):
                    angle_lines.append(f"{i+1}. {angle}")
                angle_text = "\n".join(angle_lines)

                system_prompt = (
                    "你是一个专业的电商模特动作设计师。请根据原图和以下指定角度，生成对应的具体动作描述。\n"
                    "\n"
                    "【输出要求】\n"
                    f"1. 必须恰好生成 {len(angle_list)} 条动作描述，与给定的角度顺序一一对应\n"
                    "2. 每条描述必须简洁具体，格式为：'展示XX角度，[手部/身体动作细节]'\n"
                    "3. 例如：'展示正面，双手自然下垂'、'展示左侧面，单手叉腰'\n"
                    "4. 严禁输出任何解释、编号前缀、总结\n"
                    "5. 每条描述控制在15-40字之间\n"
                    "6. 多条描述之间用一个空行分隔\n"
                    "7. 使用中文输出\n"
                )

                if use_pocket_action and has_pocket:
                    system_prompt += (
                        "\n【插口袋动作要求】\n"
                        "部分动作可以自然地包含单手或双手插口袋的细节，"
                        "使姿势更生动自然。如果某个角度不适合插口袋（如背面），则正常生成其他动作。\n"
                    )
                elif use_pocket_action and not has_pocket:
                    system_prompt += (
                        "\n【插口袋限制】\n"
                        "这件服装没有口袋，严禁生成任何插口袋相关的动作描述。\n"
                    )

                if custom_instruction.strip():
                    system_prompt += f"\n【用户补充要求】\n{custom_instruction.strip()}\n"

                user_prompt = (
                    "请为以下每个展示角度生成一条具体的动作描述：\n"
                    f"{angle_text}\n\n"
                    "每条描述只需写出角度和简单动作细节，不要编号，不要解释。"
                )

                self.console_log("📤 发送动作描述生成请求...")
                response = client.chat.completions.create(
                    model=api_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                            {"type": "text", "text": user_prompt}
                        ]}
                    ],
                    temperature=0.9,
                    seed=seed,
                    max_tokens=min(200 + len(angle_list) * 80, 2000)
                )
                result = response.choices[0].message.content.strip()

                # 统一换行符并尝试多种方式拆分
                import re
                raw = result.replace('\r\n', '\n').replace('\r', '\n')

                # 先按 \n\n 拆分，如果数量不够再按 \n 拆分
                actions = [a.strip() for a in raw.split("\n\n") if a.strip()]
                if len(actions) < len(angle_list):
                    actions = [a.strip() for a in raw.split("\n") if a.strip()]

                # 清理编号前缀，过滤空行和过短行
                cleaned_actions = []
                for a in actions:
                    a = re.sub(r'^\s*[\d\-•]+[\.\)\s]+', '', a).strip()
                    # 过滤纯标点、空白、或明显是分隔线的行
                    if a and len(a) > 3 and not re.match(r'^[-=—_]+$', a):
                        cleaned_actions.append(a)

                # 截断/补齐到目标数量
                if len(cleaned_actions) != len(angle_list):
                    self.console_log(
                        f"⚠️ API返回动作数({len(cleaned_actions)})与目标({len(angle_list)})不符，自动补齐"
                    )
                    cleaned_actions = cleaned_actions[:len(angle_list)]
                    for i in range(len(cleaned_actions), len(angle_list)):
                        angle, _ = angle_list[i]
                        cleaned_actions.append(f"展示{angle}")

                self.console_log(f"✅ 动作描述生成成功，共 {len(cleaned_actions)} 条")
                return cleaned_actions

            except Exception as e:
                self.console_log(f"⚠️ API动作生成失败，使用本地模板: {e}")

        # 本地兜底：直接用角度名
        self.console_log("⚠️ 使用本地模板生成动作描述")
        return [f"展示{angle}" for angle, _ in angle_list]

    def _build_simple_prompt(self, garment_type, action_desc):
        """
        构建固定简洁格式的编辑提示词
        """
        return (
            f"展示{garment_type}为主，为图中人物换个动作，"
            f"保持场景不变，背景不变，服装搭配不变，构图不变，"
            f"{action_desc}，"
            f"禁止多手多脚，单人单图输出，画面中只能有一个人"
        )

    def 生成编辑提示词(self, 图像, 使用API, API密钥, API模型, 展示品类, 插口袋动作, 细节数量, 任务总数,
                     正面数量, 左侧面数量, 右侧面数量, 背面数量,
                     正面坐姿数量, 左侧坐姿数量, 右侧坐姿数量,
                     随机种子, 自定义全局指令=""):

        # 清空日志
        self.console_log_messages = []
        full_log = []

        start_log = "📋 === Banana2 角度选择编辑提示词生成器 ==="
        self.console_log(start_log)
        full_log.append(start_log)

        # 收集选中的角度（数量>0即启用）
        angle_kwargs = {
            "正面": 正面数量,
            "左侧面": 左侧面数量,
            "右侧面": 右侧面数量,
            "背面": 背面数量,
            "正面坐姿": 正面坐姿数量,
            "左侧坐姿": 左侧坐姿数量,
            "右侧坐姿": 右侧坐姿数量,
        }

        angle_list = []  # [(角度名, 同角度序号), ...]
        for angle_name, count in angle_kwargs.items():
            if count > 0:
                for i in range(count):
                    angle_list.append((angle_name, i))

        if not angle_list:
            if 细节数量 > 0:
                self.console_log("⚠️ 未选择任何展示角度，只生成细节提示词")
            else:
                self.console_log("⚠️ 未选择任何展示角度，默认使用正面×1")
                angle_kwargs["正面"] = 1
                angle_list = [("正面", 0)]

        # === 任务总数控制 ===
        orig_angle_count = len(angle_list)
        orig_total = orig_angle_count + 细节数量

        if orig_total > 任务总数:
            if orig_angle_count >= 任务总数:
                细节数量 = 0
                remaining = 任务总数
                for angle_name in list(angle_kwargs.keys()):
                    if remaining <= 0:
                        angle_kwargs[angle_name] = 0
                    else:
                        take = min(angle_kwargs[angle_name], remaining)
                        angle_kwargs[angle_name] = take
                        remaining -= take
            else:
                细节数量 = 任务总数 - orig_angle_count
            # 重新构建 angle_list
            angle_list = []
            for angle_name, count in angle_kwargs.items():
                if count > 0:
                    for i in range(count):
                        angle_list.append((angle_name, i))
            self.console_log(f"⚠️ 任务总数限制：原始 {orig_total} 条 → 裁剪为 {任务总数} 条")

        elif orig_total < 任务总数:
            deficit = 任务总数 - orig_total
            for angle_name in list(angle_kwargs.keys()):
                if angle_kwargs[angle_name] > 0:
                    angle_kwargs[angle_name] += deficit
                    break
            else:
                # 没有任何角度，补充正面
                angle_kwargs["正面"] = deficit
            # 重新构建 angle_list
            angle_list = []
            for angle_name, count in angle_kwargs.items():
                if count > 0:
                    for i in range(count):
                        angle_list.append((angle_name, i))
            self.console_log(f"ℹ️ 任务总数补充：原始 {orig_total} 条 → 补充为 {任务总数} 条")

        param_log = (
            f"🔧 参数设置:\n"
            f"   - 展示品类: {展示品类}\n"
            f"   - 细节数量: {细节数量}\n"
            f"   - 任务总数: {任务总数}\n"
            f"   - 选中角度及数量:\n"
        )
        for angle_name, count in angle_kwargs.items():
            if count > 0:
                param_log += f"      · {angle_name}: {count}张\n"
        total_prompts = len(angle_list) + 细节数量
        param_log += f"   - 角度提示词: {len(angle_list)} 条\n   - 细节提示词: {细节数量} 条\n   - 总计: {total_prompts} 条\n   - 使用API: {使用API}"
        self.console_log(param_log)
        full_log.append(param_log)

        # 转换图像
        pil_img = self.tensor_to_pil(图像)
        img_log = f"🖼️ 图像尺寸: {pil_img.size}"
        self.console_log(img_log)
        full_log.append(img_log)

        # === 第一步：确定展示品类 ===
        if 展示品类 == "自动识别":
            garment_type = self._detect_garment_type(pil_img, API密钥, 使用API, API模型)
        else:
            garment_type = self.GARMENT_DISPLAY_MAP.get(展示品类, 展示品类)
            self.console_log(f"🏷️ 使用手动指定品类: {garment_type}")

        garment_log = f"✅ 确定展示品类: {garment_type}"
        self.console_log(garment_log)
        full_log.append(garment_log)

        # === 第二步：确定口袋状态 ===
        if 插口袋动作 == "自动识别":
            if 使用API and API密钥.strip() and OPENAI_AVAILABLE:
                has_pocket = self._detect_pocket(pil_img, API密钥, 使用API, API模型)
                use_pocket = has_pocket
            else:
                self.console_log("⚠️ API未启用，无法自动识别口袋，按没有口袋处理")
                has_pocket = False
                use_pocket = False
        elif 插口袋动作 == "有口袋":
            has_pocket = True
            use_pocket = True
            self.console_log("✅ 手动指定：服装有口袋，插口袋动作生效")
        else:  # 没有口袋
            has_pocket = False
            use_pocket = False
            self.console_log("✅ 手动指定：服装没有口袋，插口袋动作不生效")

        # === 第三步：生成动作描述 ===
        action_descs = self._generate_action_descriptions(
            pil_img, API密钥, 使用API, API模型,
            angle_list, 自定义全局指令,
            use_pocket_action=use_pocket, has_pocket=has_pocket,
            seed=随机种子
        )

        actions_log = "🎯 动作描述列表:\n" + "\n".join([f"   {i+1}. {desc}" for i, desc in enumerate(action_descs)])
        self.console_log(actions_log)
        full_log.append(actions_log)

        # === 第四步：生成角度提示词 ===
        edit_prompts = []
        for desc in action_descs:
            prompt = self._build_simple_prompt(garment_type, desc)
            edit_prompts.append(prompt)

        # === 第五步：生成细节特写提示词 ===
        detail_prompts = []
        if 细节数量 > 0:
            detail_elements = self._detect_garment_details(
                pil_img, API密钥, 使用API, API模型, garment_type
            )
            detail_prompts = self._generate_detail_prompts(
                garment_type, detail_elements, 细节数量
            )
            for i, dp in enumerate(detail_prompts):
                detail_log = f"🔍 细节提示词 {i+1}/{len(detail_prompts)}:\n{dp}"
                self.console_log(detail_log)
                full_log.append(detail_log)

        # 合并角度提示词和细节提示词
        all_prompts = edit_prompts + detail_prompts
        combined_prompt = "\n\n".join(all_prompts)

        for i, prompt in enumerate(edit_prompts):
            prompt_log = f"🎨 角度提示词 {i+1}/{len(edit_prompts)}:\n{prompt}"
            self.console_log(prompt_log)
            full_log.append(prompt_log)

        finish_log = f"✅ 编辑提示词生成完成，共 {len(all_prompts)} 条（角度{len(edit_prompts)} + 细节{len(detail_prompts)}）"
        self.console_log(finish_log)
        full_log.append(finish_log)

        return (combined_prompt, "\n".join(full_log))


# 注册节点
NODE_CLASS_MAPPINGS = {
    "YK_Vision_ActionPrompt_v2": YK_Vision_ActionPrompt_v2
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YK_Vision_ActionPrompt_v2": "YK视觉编辑提示词生成器v2（Banana2角度选择版）"
}
