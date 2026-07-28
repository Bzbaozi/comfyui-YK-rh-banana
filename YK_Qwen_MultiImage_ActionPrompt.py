# YK_Qwen_MultiImage_ActionPrompt.py - Qwen-VL 多图动作提示词生成器
# 功能：多张图片输入 → 阿里 DashScope Qwen-VL 逐图识别 → 每张图对应输出一条动作编辑提示词（换行分隔）
import torch
import numpy as np
from PIL import Image
import base64
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("警告：未安装openai库，API功能将不可用。请运行 'pip install openai'")


# 强约束系统提示词：针对无状态API调用设计，每次请求自带完整规则，无需模型记忆
# 结构：Markdown分区 + 少样本示例（模型模仿例子的能力最强）+ 输出契约 + 自检清单
# 任务方向：转写参考图中人物的真实动作（动作提取），而非凭空设计新动作
# 提示词版本号：更新DEFAULT_SYSTEM_PROMPT时必须同步递增，用于自动升级工作流中固化的旧版提示词
PROMPT_VERSION = "PTv5-20260717"

# f-string把版本号自动嵌入ROLE行，保证常量与提示词内标记永远一致（只需改PROMPT_VERSION一处）
DEFAULT_SYSTEM_PROMPT = f"""# ROLE [prompt-version: {PROMPT_VERSION}]
You are a strict AI Pose Transcriber: a photography director and expert prompt engineer for AI image-editing models. Your ONLY output is ONE technical English editing prompt that transfers the pose shown in the input photo onto another image.

# OUTPUT CONTRACT (highest priority — overrides every other rule)
- Output ONLY the prompt itself: one single line, one paragraph.
- English only, 30-70 words.
- No quotes, no numbering, no markdown, no meta-talk ("Here is...", explanations, apologies).
- Start immediately with: "Change the person's pose to:"

# TASK
Look at the person in the photo and TRANSCRIBE their EXACT current pose into an editing prompt. Faithfully reproduce what you see: the same body orientation, the same limb positions, the same standing/sitting/lying/leaning/kneeling state, the same framing. Do NOT invent a different pose. Do NOT add anything that is not in the photo. If the pose is unusual and fits no common category, still describe exactly what you see — never replace it with a more common pose.

# STRICT RULES (violation = failure)
1. NO CLOTHING/HAIR: never mention clothing style, color, fabric, outfit pairing, hair, headwear, or accessories.
2. NO BACKGROUND: never mention scene, environment, lighting, facial expression, or image-quality words.
3. NO HANDHELD PROPS: never mention objects held in hands (books, phones, bags, poles...). Describe only the hand/arm position itself (e.g., "hands at chest level"). ONE EXCEPTION — selfie: if the person is taking a phone selfie (typically a mirror selfie), the phone IS the action itself and MUST be mentioned: which hand holds it, at what height, and whether it covers the face (e.g., "right hand holding a phone raised in front of the face, taking a mirror selfie").
4. SUPPORT OBJECTS ONLY: if the person sits, name the object sat on exactly as seen (chair, sofa, stool, bench, steps...). If lying, name what is lain on (bed, sofa, floor, grass...) and whether face up, face down, or on the side. If leaning, name what is leaned on (wall, railing, table, doorframe...). No other objects allowed.
5. REQUIRED DETAILS — all four must match the photo: (a) body angle/direction relative to the camera (facing the camera, left side profile, back to the camera...); (b) limb positions; (c) support object when sitting, lying or leaning; (d) composition/framing judged ONLY by which body parts are inside the frame: head AND feet visible = full-body shot; head visible, cut near thighs = three-quarter shot; head visible, cut at waist = waist-up shot; head NOT visible = NEVER full-body — write the actual visible range (e.g., waist-down shot with head out of frame).
6. LEFT/RIGHT PERSPECTIVE: all left/right words in your output refer to the VIEWER's (camera's) left/right, never the subject's own left/right.
7. The prompt MUST end with exactly: "Single person only, correct human anatomy, natural hands with five fingers each, no extra or missing limbs."

# EXAMPLE (format reference only — NEVER copy its content)
Change the person's pose to: standing with the back to the camera, head turned over the right shoulder, right hand resting on the hip, left arm relaxed at the side, weight on the right leg, full-body shot. Single person only, correct human anatomy, natural hands with five fingers each, no extra or missing limbs.

# SELF-CHECK (silently verify before answering)
One line? Starts with "Change the person's pose to:"? Pose matches the photo exactly? Angle + limbs + support + framing all match the photo? Zero clothing/hair/background/prop words (selfie phone is the only allowed prop)? Safety sentence at the end? If any check fails, rewrite first, then output only the corrected prompt."""

USER_PROMPT = "Transcribe the exact pose of the person in this photo into a single editing prompt now, following the OUTPUT CONTRACT strictly."

# 方向预检提示词：先用低温调用锁定朝向/姿态/景别三要素，再注入正式转写请求
# 强制模型按可靠线索顺序判断（脸部可见性→鼻尖朝向→肩线→脚尖），显著降低左右判反率
ORIENTATION_PROMPT = (
    "Look at the person in this photo and answer in EXACTLY this one-line format, nothing else:\n"
    "ORIENTATION: <one of: facing the camera / three-quarter turn to the viewer's left / "
    "three-quarter turn to the viewer's right / left side profile / right side profile / back to the camera>; "
    "POSTURE: <standing / sitting on (object) / leaning on (object) / "
    "lying on (object), face up or face down or on the side / crouching / kneeling / walking / "
    "standing taking a mirror selfie with the phone held in (left/right) hand>; "
    "FRAMING: <full-body shot / three-quarter shot / waist-up shot / close-up / "
    "chest-down shot with head out of frame / waist-down shot with head out of frame / legs-only shot>\n"
    "If NONE of the listed options matches what you actually see, do NOT force a wrong option — "
    "write a short precise phrase describing what you really see, in the same style.\n"
    "Judging rules: left/right is strictly from the VIEWER's perspective. "
    "Check cues in this order: 1) is the face fully/partially/not visible? "
    "2) which way does the nose point? 3) the shoulder and hip line direction; 4) where the toes point. "
    "If no face is visible and the back is shown, it is back to the camera. "
    "If the face is blocked by a phone held up in front of it, that is a SELFIE — the person still faces the camera; it is NOT back to the camera.\n"
    "FRAMING is decided ONLY by which body parts are inside the frame — first check: is the HEAD inside the frame? "
    "Head AND feet both visible = full-body shot; head visible but cut near the thighs = three-quarter shot; "
    "head visible but cut at the waist = waist-up shot; "
    "HEAD NOT visible = it is NEVER a full-body shot — use chest-down / waist-down / legs-only with head out of frame."
)

# 兜底提示词（API失败时保证输出行数与图片一一对应）
FALLBACK_PROMPT = ("Change the person's pose to a relaxed natural standing position with arms at the sides, "
                   "facing the camera, full-body shot, single person only, correct human anatomy, "
                   "two arms and two legs, natural hands with five fingers each, no extra or missing limbs, "
                   "no distorted body parts.")

MODEL_CHOICES = [
    "qwen3-vl-plus",
    "qwen3-vl-max",
    "qwen3.7-plus",
    "qwen-vl-plus",
    "qwen-vl-max",
    "qwen2.5-vl-72b-instruct",
]

MAX_IMAGE_INPUTS = 6


class YKQwenMultiImageActionPrompt:
    """多图输入 → Qwen-VL 逐图分析 → 每图一条动作编辑提示词（换行分隔）"""

    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for i in range(1, MAX_IMAGE_INPUTS + 1):
            optional[f"image_{i}"] = ("IMAGE", {})
        optional["补充要求"] = ("STRING", {
            "default": "", "multiline": True,
            "tooltip": "追加到系统提示词末尾的额外约束（可留空）"
        })

        return {
            "required": {
                "API密钥": ("STRING", {"default": "", "placeholder": "阿里云 DashScope API Key"}),
                "API模型": (MODEL_CHOICES, {
                    "default": "qwen3-vl-plus",
                    "tooltip": "阿里 DashScope 视觉模型"
                }),
                "自定义模型名": ("STRING", {
                    "default": "", "multiline": False,
                    "placeholder": "填写则覆盖上方下拉框，可用任意DashScope模型名"
                }),
                "系统提示词": ("STRING", {
                    "default": DEFAULT_SYSTEM_PROMPT, "multiline": True,
                    "tooltip": "强约束系统提示词，每次API调用自带完整规则（无状态安全）"
                }),
                "temperature": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 2.0, "step": 0.1,
                                          "tooltip": "动作转写任务建议低值（0.2-0.4），越低越忠实原图"}),
                "随机种子": ("INT", {"default": 0, "min": 0, "max": 0xffffffff}),
                "失败重试次数": ("INT", {"default": 2, "min": 0, "max": 5}),
                "方向校验": ("BOOLEAN", {"default": True,
                                        "tooltip": "两阶段调用：先低温预检人物朝向/姿态/景别再正式转写，提高方向准确率（每图多一次快速API调用）"}),
                "安全后缀": (["精简", "完整", "关闭"], {
                    "default": "精简",
                    "tooltip": "提示词结尾的防崩坏约束：精简=只保留Single person only（推荐，指令型编辑模型对负向约束基本无效）；完整=保留全部；关闭=全部去除"}),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("提示词", "分析日志")
    FUNCTION = "generate"
    CATEGORY = "YK/视觉理解"
    DESCRIPTION = "多张图片输入，Qwen-VL逐图识别人物动作，每张图对应输出一条英文动作编辑提示词（换行分隔）"

    # --- 工具方法 ---
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

    def _clean_single_line(self, text):
        """把模型输出压成单行提示词：去引号/编号/多余空白，多行时取最长的一行"""
        import re
        raw = text.replace('\r\n', '\n').replace('\r', '\n').strip()
        lines = [l.strip() for l in raw.split('\n') if l.strip()]
        if not lines:
            return ""
        # 多行输出时取最长的一行（正文），丢弃"Here is..."之类的前言
        line = max(lines, key=len)
        line = re.sub(r'^\s*[\d\-•*]+[\.\)\s]+', '', line)      # 去编号前缀
        line = line.strip().strip('"').strip("'").strip('`').strip()
        return line

    def _call_api_for_image(self, client, model, system_prompt, pil_img,
                            temperature, seed, max_retries, img_label, log,
                            verify_orientation=True):
        """单张图片调用API生成一条提示词，失败重试，最终兜底
        verify_orientation=True 时先做方向预检（低温锁定朝向/姿态/景别），再注入正式请求"""
        image_base64 = self.pil_to_base64(pil_img)

        # 阶段1：方向预检（失败不阻断，降级为单阶段）
        orientation_facts = ""
        if verify_orientation:
            try:
                pre = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                        {"type": "text", "text": ORIENTATION_PROMPT}
                    ]}],
                    temperature=0.1,
                    max_tokens=160,  # 逃生口自由描述比枚举回答长，留足余量防截断
                    timeout=60,
                )
                facts = pre.choices[0].message.content.strip().replace("\n", " ")
                if "ORIENTATION" in facts.upper():
                    orientation_facts = facts
                    msg = f"🧭 [{img_label}] 方向预检: {facts}"
                    print(msg, flush=True)
                    log.append(msg)
            except Exception as e:
                msg = f"⚠️ [{img_label}] 方向预检失败（降级为单阶段调用）: {e}"
                print(msg, flush=True)
                log.append(msg)

        # 阶段2：正式转写（预检结果作为已核实事实注入，强制输出与其一致）
        user_text = USER_PROMPT
        if orientation_facts:
            user_text += (
                "\nVERIFIED FACTS from a prior visual check — your output MUST use these exact "
                f"orientation, posture and framing: {orientation_facts}"
            )

        last_error = None
        for attempt in range(1, max_retries + 2):  # 首次 + 重试次数
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": [
                            {"type": "image_url",
                             "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                            {"type": "text", "text": user_text}
                        ]}
                    ],
                    temperature=temperature,
                    seed=seed,
                    max_tokens=300,
                    timeout=60,
                )
                result = self._clean_single_line(response.choices[0].message.content)
                if len(result) < 20:
                    raise RuntimeError(f"输出过短或为空: '{result}'")
                log.append(f"✅ [{img_label}] 第{attempt}次调用成功")
                return result
            except Exception as e:
                last_error = e
                msg = f"⚠️ [{img_label}] 第{attempt}次调用失败: {e}"
                print(msg, flush=True)
                log.append(msg)
        msg = f"❌ [{img_label}] 全部尝试失败（{last_error}），使用兜底提示词"
        print(msg, flush=True)
        log.append(msg)
        return FALLBACK_PROMPT

    def _apply_suffix_policy(self, text, policy):
        """按安全后缀策略处理提示词结尾
        模型固定输出完整后缀（格式锚点保持稳定），删减由此处精确完成"""
        import re
        if policy == "完整":
            return text
        # 去掉从 "Single person only" 开始到结尾的整段安全约束（含前面的标点）
        stripped = re.sub(r'[,.\s]*\bsingle person only\b.*$', '', text,
                          flags=re.IGNORECASE).rstrip(' ,.')
        if not stripped:
            return text  # 异常防御：整条都被剥掉时保留原文
        if policy == "精简":
            return stripped + ". Single person only."
        return stripped + "."  # 关闭

    def generate(self, API密钥, API模型, 自定义模型名, 系统提示词,
                 temperature, 随机种子, 失败重试次数, 方向校验=True, 安全后缀="精简", 补充要求="", **kwargs):
        if not OPENAI_AVAILABLE:
            raise RuntimeError("未安装 openai 库，请运行: pip install openai")
        if not API密钥.strip():
            raise ValueError("请填写阿里云 DashScope API 密钥")

        model = 自定义模型名.strip() if 自定义模型名.strip() else API模型

        # 收集所有输入图片（按接口顺序，batch内逐帧展开）
        import hashlib
        pil_images = []
        for i in range(1, MAX_IMAGE_INPUTS + 1):
            img = kwargs.get(f"image_{i}")
            if img is None or img.shape[0] == 0:
                continue
            for b in range(img.shape[0]):
                pil_images.append((f"图{i}-{b+1}", self.tensor_to_pil(img[b:b+1])))

        if not pil_images:
            raise ValueError("至少需要连接一张输入图片（image_1 ~ image_6）")

        # 诊断：打印每张图的尺寸和内容指纹，检测是否收到重复图片
        fingerprints = {}
        for label, pil_img in pil_images:
            fp = hashlib.md5(pil_img.tobytes()).hexdigest()[:8]
            fingerprints.setdefault(fp, []).append(label)
            print(f"📷 [{label}] 尺寸 {pil_img.size[0]}x{pil_img.size[1]} | 内容指纹 {fp}", flush=True)
        dup_groups = {fp: labels for fp, labels in fingerprints.items() if len(labels) > 1}
        if dup_groups:
            for fp, labels in dup_groups.items():
                print(f"⚠️⚠️ 警告：{'、'.join(labels)} 的图片内容完全相同（指纹 {fp}）！"
                      f"请检查上游节点接线，多个接口收到了同一张图", flush=True)

        # 组装系统提示词（补充要求追加在末尾，仍受 OUTPUT CONTRACT 约束）
        system_prompt = 系统提示词.strip() if 系统提示词.strip() else DEFAULT_SYSTEM_PROMPT
        # 检测工作流中固化的旧版提示词，强制升级为最新版
        # ComfyUI会把文本框的值保存在工作流里，代码更新默认值不会刷新已有节点
        # 判定规则：①命中旧版特征串 ②是本节点的转写提示词但版本号与当前不符
        old_markers = ("design a NEW pose", "Pose Editor", "=== OUTPUT CONTRACT (absolute")
        is_outdated = (any(m in system_prompt for m in old_markers)
                       or ("Pose Transcriber" in system_prompt and PROMPT_VERSION not in system_prompt))
        if is_outdated:
            print("⚠️ 检测到工作流中保存的是旧版本系统提示词，已自动替换为最新版 "
                  f"({PROMPT_VERSION})。如需自定义约束请使用「补充要求」输入框（不会被覆盖）", flush=True)
            system_prompt = DEFAULT_SYSTEM_PROMPT
        if 补充要求.strip():
            system_prompt += f"\n\n=== ADDITIONAL USER CONSTRAINTS ===\n{补充要求.strip()}"

        log = [f"📋 === Qwen-VL 多图动作提示词生成器 ===",
               f"🔧 模型: {model} | 图片数: {len(pil_images)} | temperature: {temperature} | 种子: {随机种子}"]
        print(log[-1], flush=True)

        client = OpenAI(
            api_key=API密钥.strip(),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        # 并发调用（每图独立请求，保证输出与输入一一对应）
        results = [None] * len(pil_images)

        def worker(idx, label, pil_img):
            # 每张图种子递增，避免同参数下多图输出雷同
            img_seed = (随机种子 + idx) % 0xffffffff
            results[idx] = self._call_api_for_image(
                client, model, system_prompt, pil_img,
                temperature, img_seed, 失败重试次数, label, log,
                verify_orientation=方向校验
            )

        max_workers = min(4, len(pil_images))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(worker, idx, label, pil_img)
                       for idx, (label, pil_img) in enumerate(pil_images)]
            for f in futures:
                f.result()

        # 每张图一条提示词，换行分隔（与图生图节点的提示词行格式对齐）
        fallback_count = sum(1 for r in results if r is FALLBACK_PROMPT)
        if fallback_count == len(results):
            # 全部失败：直接报错让用户看到真实原因，避免静默输出N条相同的兜底提示词
            error_detail = "\n".join(l for l in log if "失败" in l)
            raise RuntimeError(
                f"全部 {len(results)} 张图片的API调用均失败，请检查模型名和API密钥。失败详情：\n{error_detail}"
            )
        if fallback_count > 0:
            print(f"⚠️ 有 {fallback_count}/{len(results)} 张图片使用了兜底提示词，详见分析日志", flush=True)

        # 按安全后缀策略处理结尾（须在fallback计数之后，避免破坏 is FALLBACK_PROMPT 判定）
        results = [self._apply_suffix_policy(r, 安全后缀) for r in results]

        combined = "\n".join(results)

        for idx, ((label, _), prompt) in enumerate(zip(pil_images, results)):
            entry = f"🎨 [{label}] 提示词 {idx+1}/{len(results)}:\n{prompt}"
            print(entry, flush=True)
            log.append(entry)

        log.append(f"✅ 完成：共输出 {len(results)} 条提示词（每行一条，与输入图片顺序一致）")
        print(log[-1], flush=True)
        return (combined, "\n".join(log))


NODE_CLASS_MAPPINGS = {
    "YKQwenMultiImageActionPrompt": YKQwenMultiImageActionPrompt
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YKQwenMultiImageActionPrompt": "YK-Qwen多图动作提示词生成器"
}
