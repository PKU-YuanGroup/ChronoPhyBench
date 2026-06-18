import os
import base64
import json
import requests
import json
import base64
import requests
import cv2
import numpy as np
import torch
from PIL import Image
from typing import Dict, List, Any

class BaseEvaluator:
    def __init__(self, api_key, model_name):
        self.api_key = api_key
        self.model_name = model_name

    def _extract_frames(self, video_path, num_frames=8, target_size=640):
        frames_b64 = []
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0: return []
        
        indices = np.linspace(0, total - 1, num_frames, dtype=int)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret: break
            h, w = frame.shape[:2]
            scale = target_size / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frames_b64.append(base64.b64encode(buffer).decode('utf-8'))
        cap.release()
        return frames_b64

class GeminipredEvaluator(BaseEvaluator):
    def __init__(self, api_key, model_name, image_root):
        super().__init__(api_key, model_name)
        self.image_root = os.path.abspath(image_root)

    def _get_b64_from_path(self, relative_path):
        if self.image_root is None:
            print("❌ 错误：未设置 image_root，无法读取图片选项。")
            return None

        full_path = os.path.join(self.image_root, relative_path)
        
        try:
            if not os.path.exists(full_path):
                print(f"❌ 找不到文件: {full_path}")
                return None
            with open(full_path, "rb") as f:
                return base64.b64encode(f.read()).decode('utf-8')
        except Exception as e:
            print(f"❌ 读取图片异常: {e}")
            return None

    def call(self, video_path, item):
        video_frames = self._extract_frames(video_path)
        print(f"DEBUG: 当前视频提取了 {len(video_frames)} 帧")
        option_images = {}
        for label, rel_path in item.get('options', {}).items():
            b64 = self._get_b64_from_path(rel_path)
            if b64:
                option_images[label] = b64

        prompt_text = (
            "Note: The visual input contains a video and three separate option images (A, B, C).The last three images are options A, B and C in order."
            "I have provided them separately. Please do not confuse the option images with the video frames.\n"
            f"Question Context: {item['question']}\n\n"
            "You must follow the output format strictly. DO NOT provide extra explanations outside the format.\n"
            "Conclusion: Give the Option Letter and content."
            "Analysis: Give the reason why you chose this answer.\n"
        )

        parts = []

        parts.append({"text": 'The first is the initial video clip.'})
        for frame in video_frames:
            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": frame}})
        
        parts.append({"text": 'Next are the option pictures.'})

        for label in sorted(option_images.keys()):
            parts.append({"text": f"Option {label}:"})
            parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": option_images[label]
                }
            })
        
        parts.append({"text": prompt_text})

        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "temperature": 0.0,
                "topP": 0.8
            }
        }
        
        url = f"https://infra.chatexcel.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {"temperature": 0.01}
        }

        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=500)
            if response.status_code == 200:
                res_data = response.json()
                return res_data['candidates'][0]['content']['parts'][0]['text']
            else:
                print(f"请求失败 ({response.status_code}): {response.text}")
                return ""
        except Exception as e:
            print(f"API 异常: {e}")
            return ""

class BOLATUEvaluator(BaseEvaluator):
    def __init__(self, api_key, model_name, image_root):
        super().__init__(api_key, model_name)
        self.image_root = os.path.abspath(image_root)
        self.api_url = "https://one-api.bltcy.top/v1/chat/completions"

    def _get_b64_from_path(self, relative_path):
        if self.image_root is None:
            return None
        full_path = os.path.join(self.image_root, relative_path)
        try:
            if not os.path.exists(full_path):
                print(f"❌ 找不到文件: {full_path}")
                return None
            with open(full_path, "rb") as f:
                return base64.b64encode(f.read()).decode('utf-8')
        except Exception as e:
            print(f"❌ 读取图片异常: {e}")
            return None

    def call(self, video_path, item):
        video_frames = self._extract_frames(video_path)
        print(f"DEBUG: 当前视频提取了 {len(video_frames)} 帧")

        option_images = {}
        for label, rel_path in item.get('options', {}).items():
            b64 = self._get_b64_from_path(rel_path)
            if b64:
                option_images[label] = b64    

        content_list = []
        
        content_list.append({"type": "text", "text": "The first part is the initial video clip:"})
        
        for frame_b64 in video_frames:
            content_list.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}
            })
        
        content_list.append({"type": "text", "text": "Next are the option pictures."})
        for label in sorted(option_images.keys()):
            content_list.append({"type": "text", "text": f"Option {label}:"})
            content_list.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{option_images[label]}"}
            })
        
        prompt_text = (
            "Note: The visual input contains a video and three separate option images (A, B, C).The last three images are options A, B and C in order."
            "I have provided them separately. Please do not confuse the option images with the video frames.\n"
            f"Question Context: {item['question']}\n\n"
            "You must follow the output format strictly. DO NOT provide extra explanations outside the format.\n"
            "Conclusion: Give the Option Letter."
            "Analysis: Give the reason why you chose this answer.\n"
        )
        content_list.append({"type": "text", "text": prompt_text})

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": content_list
                }
            ],
            "temperature": 0.1,
            "max_tokens": 2048,
            "stream": False
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=500)
            if response.status_code == 200:
                res_data = response.json()
                return res_data['choices'][0]['message']['content']
            else:
                print(f"请求失败 ({response.status_code}): {response.text}")
                return ""
        except Exception as e:
            print(f"API 异常: {e}")
            return ""

import os
import base64
import requests
import json

class LocalBaseEvaluator(BaseEvaluator):
    def __init__(self, api_key, model_name, device="cuda"):
        super().__init__(api_key, model_name)
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self.tokenizer = None

from transformers import AutoProcessor, AutoModelForImageTextToText
class QwenVLpredLocalEvaluator(LocalBaseEvaluator):
    def __init__(self, api_key, model_name, image_root, device="cuda"):
        super().__init__(api_key, model_name, device=device)
        self.image_root = os.path.abspath(image_root) if image_root else None
        self._load_model()

    def _load_model(self):
        try:
            print(f"🔄 正在加载 Qwen2-VL 模型: {self.model_name}...")

            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )

            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map='auto',
                attn_implementation="flash_attention_2",
                trust_remote_code=True
            ).eval()
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            print("✅ Qwen2-VL 模型加载成功!")
        except Exception as e:
            raise RuntimeError(f"加载 Qwen2-VL 模型失败: {e}")

    def call(self, video_path, item):
        try:
            from qwen_vl_utils import process_vision_info
            
            content_list = []
            content_list.append({"type": "text", "text": 'The first is the initial video clip.'})

            content_list.append(
                {
                    "type": "video",
                    "video": video_path,
                    "nframes": 8,
                }
            )

            content_list.append({"type": "text", "text": 'Next are the option pictures.'})

            options = item.get('options', {})
            for label in sorted(options.keys()):
                rel_path = options[label]
                if self.image_root:
                    full_image_path = os.path.join(self.image_root, rel_path)
                    if os.path.exists(full_image_path):
                        content_list.append({"type": "text", "text": f"Option {label}:"})
                        content_list.append({"type": "image", "image": full_image_path})
                    else:
                        print(f"⚠️ 警告: 找不到图片 {full_image_path}")

            prompt_text = (
                "Note: The visual input contains a video and three separate option images (A, B, C).The last three images are options A, B and C in order."
                "I have provided them separately. Please do not confuse the option images with the video frames.\n"
                f"Question Context: {item['question']}\n\n"
                "You must follow the output format strictly. DO NOT provide extra explanations outside the format.\n"
                "Conclusion: Give the Option Letter and content."
                "Analysis: Give the reason why you chose this answer.\n"
            )

            content_list.append({"type": "text", "text": prompt_text})

            messages = [{"role": "user", "content": content_list}]

            text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True,
            )
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=8192,
                    do_sample=True,
                    temperature=1.0,
                    top_k=20,
                    top_p=0.95,
                    repetition_penalty=1.0,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            if "</think>" in output_text:
                output_text = output_text.split("</think>", 1)[-1].strip()
                if output_text.startswith("\n"):
                    output_text = output_text[1:].strip()
            else:
                output_text = output_text.strip()

            return output_text.strip()

        except Exception as e:
            return f"Runtime Error: {str(e)}"

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from decord import VideoReader, cpu
from transformers import AutoTokenizer, AutoProcessor, AutoModel
class InternVLpredLocalEvaluator(LocalBaseEvaluator):
    def __init__(self, api_key, model_name, image_root, device="cuda"):
        super().__init__(api_key, model_name, device=device)
        self.model_path = model_name
        self.image_root = os.path.abspath(image_root) if image_root else None
        
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        try:
            print(f"🔄 正在加载 InternVL 模型: {self.model_path}...")

            from transformers import BitsAndBytesConfig

            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True, fix_mistral_regex=True)
            self.model = AutoModel.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True,
                device_map='auto',
                low_cpu_mem_usage=True,
                attn_implementation="flash_attention_2",
            ).eval()
            print("✅ InternVL 模型加载成功!")
        except Exception as e:
            raise RuntimeError(f"加载InternVL模型失败: {e}")

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def build_transform(self, input_size):
        MEAN, STD = self.IMAGENET_MEAN, self.IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def get_index(self, bound, fps, max_frame, first_idx=0, num_segments=32):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / num_segments
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(num_segments)
        ])
        return frame_indices

    def load_video(self, video_path, bound=None, input_size=448, max_num=1, num_segments=32):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())

        pixel_values_list, num_patches_list = [], []
        
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
        
        for frame_index in frame_indices:
            _frame = vr[frame_index]
            if hasattr(_frame, 'asnumpy'):
                _frame_np = _frame.asnumpy()
            elif hasattr(_frame, 'numpy'):
                _frame_np = _frame.detach().cpu().numpy()
            else:
                import numpy as np
                _frame_np = np.array(_frame)

            img = Image.fromarray(_frame_np).convert('RGB')
            
            img = self.dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
            
            pixel_values = [self.build_transform(input_size)(tile) for tile in img]
            pixel_values = torch.stack(pixel_values)
            
            num_patches_list.append(pixel_values.shape[0])
            pixel_values_list.append(pixel_values)
            
        pixel_values = torch.cat(pixel_values_list)
        return pixel_values, num_patches_list
    
    def load_image(self, image_path, input_size=448, max_num=12):
        image = Image.open(image_path).convert('RGB')
        transform = self.build_transform(input_size=input_size)
        images = self.dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values
    
    def call(self, video_path, item):
        try:
            pixel_values_video, num_patches_video = self.load_video(
                video_path, num_segments=8, max_num=1 
            )
            
            pixel_values_options = []
            num_patches_options = []
            option_labels = sorted(item.get('options', {}).keys())
            
            for label in option_labels:
                rel_path = item['options'][label]
                full_path = os.path.join(self.image_root, rel_path)
                if os.path.exists(full_path):
                    pv_img = self.load_image(full_path, max_num=1) 
                    pixel_values_options.append(pv_img)
                    num_patches_options.append(pv_img.shape[0])
            
            all_pixel_values = torch.cat([pixel_values_video] + pixel_values_options)
            all_pixel_values = all_pixel_values.to(torch.bfloat16).to(self.model.device)
            
            total_patches_list = num_patches_video + num_patches_options

            video_prompt = "".join([f"Frame{i+1}: <image>\n" for i in range(len(num_patches_video))])
            
            options_prompt = ""
            for i, label in enumerate(option_labels):
                img_idx = len(num_patches_video) + i + 1
                options_prompt += f"Option {label}: <image>\n"

            question = (
                f"{video_prompt}\n"
                f"{options_prompt}\n"
                "Note: The visual input contains a video and three separate option images (A, B, C).The last three images are options A, B and C in order."
                "I have provided them separately. Please do not confuse the option images with the video frames.\n"
                f"Question: {item['question']}\n\n"
                "You must follow the output format strictly. DO NOT provide extra explanations outside the format.\n"
                "Conclusion: Give the Option Letter."
                "Analysis: Give the reason why you chose this answer.\n"
            )

            generation_config = dict(
                max_new_tokens=1024, 
                do_sample=True, 
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            response = self.model.chat(
                self.tokenizer, 
                all_pixel_values, 
                question, 
                generation_config,
                num_patches_list=total_patches_list,
                history=None, 
                return_history=False
            )
            return response

        except Exception as e:
            import traceback
            return f"Runtime Error: {str(e)}\n{traceback.format_exc()}"

from transformers import AutoProcessor, AutoModelForImageTextToText
class InternVL35predLocalEvaluator(LocalBaseEvaluator):
    def __init__(self, api_key, model_name, image_root, device="cuda"):
        super().__init__(api_key, model_name, device=device)
        self.model_name = model_name
        self.device = device
        self.image_root = os.path.abspath(image_root) if image_root else None
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        try:
            print(f"🔄 正在加载 InternVL3/3.5 模型: {self.model_name}...")

            from transformers import BitsAndBytesConfig

            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            
            self.processor = AutoProcessor.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            )

            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16, 
                trust_remote_code=True,
                device_map='auto',
                attn_implementation="flash_attention_2",
            ).eval()

            print(f"进程 [Rank {self.device}] 模型加载成功。")

        except Exception as e:
            raise RuntimeError(f"加载 InternVL 模型失败: {e}")

    def call(self, video_path, item):
        try:
            content_list = []
            content_list.append({"type": "text", "text": 'The first is the initial video clip.'})
            content_list.append(
                {"type": "video", "video": video_path}
            )

            content_list.append({"type": "text", "text": 'Next are the option pictures.'})

            options = item.get('options', {})
            option_labels = sorted(options.keys())
            
            for label in option_labels:
                rel_path = options[label]
                if self.image_root:
                    full_image_path = os.path.join(self.image_root, rel_path)
                    if os.path.exists(full_image_path):
                        content_list.append({"type": "text", "text": f"Option {label}:"})
                        content_list.append({"type": "image", "image": full_image_path})
                    else:
                        print(f"⚠️ 找不到图片: {full_image_path}")
            
            prompt = (
                "Note: The visual input contains a video and three separate option images (A, B, C).The last three images are options A, B and C in order."
                "I have provided them separately. Please do not confuse the option images with the video frames.\n"
                f"Question Context: {item['question']}\n\n"
                "You must follow the output format strictly. DO NOT provide extra explanations outside the format.\n"
                "Conclusion: Give the Option Letter."
                "Analysis: Give the reason why you chose this answer.\n"
            )

            content_list.append({"type": "text", "text": prompt})

            messages = [{"role": "user", "content": content_list}]

            inputs = self.processor.apply_chat_template(
                messages,
                return_tensors="pt",
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                size={"height": 448, "width": 448}, 
                num_frames=8, 
            ).to(self.model.device, dtype=torch.bfloat16)

            with torch.no_grad():
                output = self.model.generate(
                    **inputs, 
                    max_new_tokens=4096,
                    do_sample=True,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                )

            input_len = inputs["input_ids"].shape[1]
            output_text = self.processor.decode(
                output[0, input_len:], 
                skip_special_tokens=True
            )

            import re
            output_text = re.sub(r"<think>.*?</think>\s*", "", output_text, flags=re.DOTALL).strip()

            if not output_text:
                output_text = re.sub(r"<think>.*?</think>\s*", "", original_text, flags=re.DOTALL).strip()

            return output_text.strip()

        except Exception as e:
            import traceback
            print(f"推理出错 [Video: {video_path}]: {e}\n{traceback.format_exc()}")
            return f"Error: {e}"

from transformers import AutoModelForCausalLM, AutoProcessor
class KimiVLpredEvaluator(LocalBaseEvaluator):
    def __init__(self, api_key, model_name, image_root, device="cuda"):
        super().__init__(api_key, model_name, device=device)
        self.model_path = model_name
        self.image_root = os.path.abspath(image_root) if image_root else None
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        try:
            print(f"🔄 正在加载 Kimi-VL 模型: {self.model_path}...")
            
            from transformers import BitsAndBytesConfig

            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map='auto',
                attn_implementation="flash_attention_2",
                quantization_config=quant_config,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            ).eval()
            self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
            print("✅ Kimi-VL 模型加载成功!")
        except Exception as e:
            raise RuntimeError(f"加载 Kimi-VL 失败: {e}")

    def _extract_frame_images(self, video_path, num_frames=8):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0: return []
        
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret: break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
        cap.release()
        return frames

    def call(self, video_path, item):
        try:
            video_frames = self._extract_frame_images(video_path, num_frames=8)
            
            content_list = []
            content_list.append({"type": "text", "text": 'The first is the initial video clip.'})
            
            for i, frame in enumerate(video_frames):
                content_list.append({"type": "image", "image": frame})
            
            content_list.append({"type": "text", "text": 'Next are the option pictures.'})

            options = item.get('options', {})
            option_labels = sorted(options.keys())
            images_to_process = list(video_frames)

            for label in option_labels:
                rel_path = options[label]
                if self.image_root:
                    full_path = os.path.join(self.image_root, rel_path)
                    if os.path.exists(full_path):
                        img = Image.open(full_path).convert('RGB')
                        content_list.append({"type": "text", "text": f"Option {label}:"})
                        content_list.append({"type": "image", "image": img})
                        images_to_process.append(img)

            prompt = (
                "Note: The visual input contains a video and three separate option images (A, B, C).The last three images are options A, B and C in order."
                "I have provided them separately. Please do not confuse the option images with the video frames.\n"
                f"Question Context: {item['question']}\n\n"
                "You must follow the output format strictly. DO NOT provide extra explanations outside the format.\n"
                "Conclusion: Give the Option Letter and content."
                "Analysis: Give the reason why you chose this answer.\n"
            )
            content_list.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content_list}]

            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
            
            inputs = self.processor(
                images=images_to_process, 
                text=prompt, 
                return_tensors="pt", 
                padding=True
            ).to(self.model.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=512,
                    temperature=0.2,
                )

            in_len = inputs.input_ids.shape[1]
            response = self.processor.batch_decode(
                generated_ids[:, in_len:], 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]

            return response.strip()

        except Exception as e:
            return f"Runtime Error: {str(e)}"

class GLM46VpredLocalEvaluator(LocalBaseEvaluator):
    def __init__(self, api_key, model_name, image_root, device="cuda"):
        super().__init__(api_key, model_name, device=device)
        self.model_path = model_name
        self.image_root = os.path.abspath(image_root) if image_root else None
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        try:
            print(f"🔄 正在加载 GLM-4.6V 模型: {self.model_path}...")
            self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
            self.model = Glm4vForConditionalGeneration.from_pretrained(
                pretrained_model_name_or_path=self.model_path,
                torch_dtype="auto",
                device_map="auto",
                attn_implementation="flash_attention_2",
                trust_remote_code=True
            ).eval()
            print("✅ GLM-4.6V 加载成功!")
        except Exception as e:
            raise RuntimeError(f"加载 GLM 失败: {e}")

    def _extract_frames(self, video_path, nframes=8):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, total_frames - 1, nframes, dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret: break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
        cap.release()
        return frames

    def call(self, video_path, item):
        try:
            video_frames = self._extract_frames(video_path, nframes=8)
            
            content = []
            content.append({"type": "text", "text": 'The first is the initial video clip.'})
            
            for frame in video_frames:
                content.append({"type": "image", "image": frame})
            
            content.append({"type": "text", "text": 'Next are the option pictures.'})

            options = item.get('options', {})
            option_labels = sorted(options.keys())
            
            for label in option_labels:
                rel_path = options[label]
                if self.image_root:
                    full_path = os.path.join(self.image_root, rel_path)
                    if os.path.exists(full_path):
                        content.append({"type": "text", "text": f"Option {label}:"})
                        content.append({"type": "image", "image": Image.open(full_path).convert('RGB')})

            prompt = (
                "Note: The visual input contains a video and three separate option images (A, B, C).The last three images are options A, B and C in order."
                "I have provided them separately. Please do not confuse the option images with the video frames.\n"
                f"Question Context: {item['question']}\n\n"
                "Structure your response as:"
                "1.Conclusion: \n"
                "2.Analysis: Give the reason why you chose this answer.\n"
            )
            content.append({"type": "text", "text": prompt})

            messages = [{"role": "user", "content": content}]

            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.model.device)
            
            inputs.pop("token_type_ids", None)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=2048, 
                    do_sample=True,
                    temperature=0.8,
                    top_k=2,
                    top_p=0.6,
                )
            
            output_text = self.processor.decode(
                generated_ids[0][inputs["input_ids"].shape[1]:], 
                skip_special_tokens=True
            )

            if "<think>" in output_text and "</think>" in output_text:
                output_text = output_text.split("</think>")[-1]
            elif "<think>" in output_text:
                output_text = output_text.split("<think>")[0]

            return output_text.strip()

        except Exception as e:
            import traceback
            return f"Runtime Error: {str(e)}\n{traceback.format_exc()}"

class GLM41VpredLocalEvaluator(LocalBaseEvaluator):
    def __init__(self, api_key, model_name, image_root, device="cuda"):
        super().__init__(api_key, model_name, device=device)
        self.model_path = model_name
        self.image_root = os.path.abspath(image_root) if image_root else None
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        try:
            print(f"🔄 正在加载 GLM-4.1V-Thinking: {self.model_path}...")
            self.processor = AutoProcessor.from_pretrained(self.model_path, use_fast=True, trust_remote_code=True)
            self.model = Glm4vForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            ).eval()
            print("✅ 模型加载成功!")
        except Exception as e:
            raise RuntimeError(f"加载 GLM 失败: {e}")

    def _extract_frames(self, video_path, nframes=8):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, total_frames - 1, nframes, dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret: break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
        cap.release()
        return frames

    def call(self, video_path, item):
        try:
            video_frames = self._extract_frames(video_path, nframes=8)
            
            content = []
            content.append({"type": "text", "text": 'The first is the initial video clip.'})
            
            for i, frame in enumerate(video_frames):
                content.append({"type": "image", "image": frame})
            
            content.append({"type": "text", "text": 'Next are the option pictures.'})

            options = item.get('options', {})
            option_labels = sorted(options.keys())
            for label in option_labels:
                rel_path = options[label]
                full_path = os.path.join(self.image_root, rel_path)
                if os.path.exists(full_path):
                    content.append({"type": "text", "text": f"Option {label}:"})
                    content.append({"type": "image", "image": Image.open(full_path).convert('RGB')})

            prompt = (
                "Note: The visual input contains a video and three separate option images (A, B, C).The last three images are options A, B and C in order."
                "I have provided them separately. Please do not confuse the option images with the video frames.\n"
                f"Question Context: {item['question']}\n\n"
                "You must follow the output format strictly. DO NOT provide extra explanations outside the format.\n"
                "Conclusion: Give the Option Letter and content."
                "Analysis: Give the reason why you chose this answer.\n"
            )
            content.append({"type": "text", "text": prompt})

            messages = [{"role": "user", "content": content}]

            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.model.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=2048, 
            )
            
            output_text = self.processor.decode(
                generated_ids[0][inputs["input_ids"].shape[1]:], 
                skip_special_tokens=True
            )

            if "</think>" in output_text:
                output_text = output_text.split("</think>")[-1]
            elif "<think>" in output_text:
                output_text = output_text.split("<think>")[0]

            return output_text.strip()

        except Exception as e:
            import traceback
            return f"Runtime Error: {str(e)}\n{traceback.format_exc()}"

import math
from transformers import AutoModel, AutoTokenizer
from scipy.spatial import cKDTree

class MiniCPM45VpredLocalEvaluator(LocalBaseEvaluator):
    def __init__(self, api_key, model_name, image_root, device="cuda"):
        super().__init__(api_key, model_name, device=device)
        self.model_path = model_name
        self.image_root = os.path.abspath(image_root) if image_root else None
        self.model = None
        self.tokenizer = None
        self.TIME_SCALE = 0.1
        self._load_model()

    def _load_model(self):
        try:
            print(f"🔄 正在加载 MiniCPM-V-4.5: {self.model_path}...")
            self.model = AutoModel.from_pretrained(
                self.model_path, 
                device_map="auto",
                trust_remote_code=True,
                attn_implementation='flash_attention_2', 
                torch_dtype=torch.bfloat16
            ).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, 
                trust_remote_code=True
            )
            print("✅ MiniCPM 加载成功!")
        except Exception as e:
            raise RuntimeError(f"加载 MiniCPM 失败: {e}")

    def _map_to_nearest_scale(self, values, scale):
        tree = cKDTree(np.asarray(scale)[:, None])
        _, indices = tree.query(np.asarray(values)[:, None])
        return np.asarray(scale)[indices]

    def _extract_frames_with_ts(self, video_path, nframes=8):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / fps
        
        indices = np.linspace(0, total_frames - 1, nframes, dtype=int)
        frames = []
        ts_ids = []
        
        scale = np.arange(0, video_duration + self.TIME_SCALE, self.TIME_SCALE)

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret: break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
            
            current_ts = idx / fps
            ts_id = int(self._map_to_nearest_scale([current_ts], scale)[0] / self.TIME_SCALE)
            ts_ids.append(ts_id)
            
        cap.release()
        return frames, [ts_ids]

    def call(self, video_path, item):
        try:
            video_frames, video_temporal_ids = self._extract_frames_with_ts(video_path, nframes=8)
            
            option_images = []
            option_temporal_ids = []
            
            options = item.get('options', {})
            option_labels = sorted(options.keys())
            
            for label in option_labels:
                rel_path = options[label]
                full_path = os.path.join(self.image_root, rel_path)
                if os.path.exists(full_path):
                    img = Image.open(full_path).convert('RGB')
                    option_images.append(img)
                    option_temporal_ids.append(-1)

            all_media = video_frames + option_images
            combined_temporal_ids = [video_temporal_ids[0] + option_temporal_ids]

            prompt_text = (
                "Note: The visual input contains a video and three separate option images (A, B, C).The last three images are options A, B and C in order."
                "I have provided them separately. Please do not confuse the option images with the video frames.\n"
                f"Question Context: {item['question']}\n\n"
                "You must follow the output format strictly. DO NOT provide extra explanations outside the format.\n"
                "Conclusion: Give the Option Letter and content."
                "Analysis: Give the reason why you chose this answer.\n"
            )
            msgs = [{'role': 'user', 'content': all_media + [prompt_text]}]
            enable_thinking=True

            output_text = self.model.chat(
                msgs=msgs,
                tokenizer=self.tokenizer,
                use_image_id=False,
                max_slice_nums=2, 
                temporal_ids=combined_temporal_ids,
                max_new_tokens=4096,
                enable_thinking=enable_thinking,
            )

            if "</think>" in output_text:
                output_text = output_text.split("</think>", 1)[-1].strip()
                if output_text.startswith("\n"):
                    output_text = output_text[1:].strip()
            else:
                output_text = output_text.strip()

            return output_text.strip()

        except Exception as e:
            import traceback
            return f"Runtime Error: {str(e)}\n{traceback.format_exc()}"
            
from transformers import AutoProcessor, AutoModelForCausalLM

class Step3VLPredLocalEvaluator(LocalBaseEvaluator):
    def __init__(self, api_key, model_name, image_root, device="cuda"):
        super().__init__(api_key, model_name, device=device)
        self.image_root = os.path.abspath(image_root) if image_root else None
        self.model = None
        self.processor = None
        self.key_mapping = {
            "^vision_model": "model.vision_model",
            r"^model(?!\.(language_model|vision_model))": "model.language_model",
            "vit_large_projector": "model.vit_large_projector",
        }
        self._load_model()

    def _load_model(self):
        try:
            print(f"🔄 正在加载 Step3-VL 模型: {self.model_name}...")

            from transformers import BitsAndBytesConfig

            self.processor = AutoProcessor.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                device_map='auto',
                torch_dtype=torch.bfloat16,
                key_mapping=self.key_mapping
            ).eval()
            
            print("✅ Step3-VL 模型加载成功!")
        except Exception as e:
            raise RuntimeError(f"加载 Step3-VL 失败: {e}")

    def _extract_frames(self, video_path, nframes=8):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            return []
        indices = np.linspace(0, total_frames - 1, nframes, dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret: break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
        cap.release()
        return frames

    def call(self, video_path, item):
        try:
            video_frames = self._extract_frames(video_path, nframes=8)
            
            content_list = []
            content_list.append({"type": "text", "text": 'The first is the initial video clip.'})
            
            for frame in video_frames:
                content_list.append({"type": "image", "image": frame})
            
            content_list.append({"type": "text", "text": 'Next are the option pictures.'})
            options = item.get('options', {})
            for label in sorted(options.keys()):
                rel_path = options[label]
                if self.image_root:
                    full_image_path = os.path.join(self.image_root, rel_path)
                    if os.path.exists(full_image_path):
                        content_list.append({"type": "text", "text": f"Option {label}:"})
                        content_list.append({"type": "image", "image": Image.open(full_image_path).convert('RGB')})
                    else:
                        print(f"⚠️ 警告: 找不到图片 {full_image_path}")

            prompt = (
                "Note: The visual input contains a video and six separate option images (A, B, C, D, E, F).The last six images are options A, B, C, D, E and F in order."
                "I have provided them separately. Please do not confuse the option images with the video frames.\n"
                f"Question Context: {item['question']}\n\n"
                "You must follow the output format strictly. DO NOT provide extra explanations outside the format.\n"
                "Conclusion: Give the Option Letter and content."
                "Analysis: Give the reason why you chose this answer.\n"
            )
            content_list.append({"type": "text", "text": prompt})

            messages = [{"role": "user", "content": content_list}]
            
            inputs = self.processor.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                tokenize=True,
                return_dict=True, 
                return_tensors="pt"
            ).to(self.model.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    do_sample=True,
                )
            
            def clean_step3_output(output_text):
                if "</think>" in output_text:
                    output_text = output_text.split("</think>")[-1]
                elif "<think>" in output_text:
                    output_text = output_text.split("<think>")[0]
                return output_text

            output_text = self.processor.decode(
                generated_ids[0, inputs["input_ids"].shape[-1]:], 
                skip_special_tokens=True
            )
            final_answer = clean_step3_output(output_text)
            return final_answer

        except Exception as e:
            import traceback
            return f"Runtime Error: {str(e)}\n{traceback.format_exc()}"

from transformers import AutoModelForCausalLM
from moviepy.editor import VideoFileClip
class Ovis26PredLocalEvaluator(LocalBaseEvaluator):
    def __init__(self, api_key, model_name, image_root, device="cuda"):
        super().__init__(api_key, model_name, device=device)

        self.image_root = os.path.abspath(image_root) if image_root else None
        self.model = None
        self.enable_thinking = True
        self.thinking_budget = 2048
        self.max_new_tokens = 3072
        self._load_model()

    def _load_model(self):
        try:
            print(f"🔄 正在加载 Ovis 模型: {self.model_name}...")
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
                device_map="auto"
            )
            print("✅ Ovis 模型加载成功!")
        except Exception as e:
            raise RuntimeError(f"加载 Ovis 失败: {e}")

    def _extract_frames(self, video_path, nframes=8):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0: return []
        indices = np.linspace(0, total_frames - 1, nframes, dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret: break
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        cap.release()
        return frames

    def call(self, video_path, item):
        try:
            num_frames = 8

            with VideoFileClip(video_path) as clip:
                total_frames = int(clip.fps * clip.duration)
                indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
                video_frames = [Image.fromarray(clip.get_frame(t)) for t in (idx / clip.fps for idx in indices)]
            
            content = []
            content.append({"type": "text", "text": 'The first is the initial video clip.'})
            for frame in video_frames:
                content.append({"type": "image", "image": frame})
            
            content.append({"type": "text", "text": 'Next are the option pictures.'})
            options = item.get('options', {})
            for label in sorted(options.keys()):
                img_path = os.path.join(self.image_root, options[label])
                if os.path.exists(img_path):
                    content.append({"type": "text", "text": f"Option {label}:"})
                    content.append({"type": "image", "image": Image.open(img_path).convert('RGB')})

            prompt = (
                "Note: The visual input contains a video and three separate option images (A, B, C).The last three images are options A, B and C in order."
                "I have provided them separately. Please do not confuse the option images with the video frames.\n"
                f"Question Context: {item['question']}\n\n"
                "You must follow the output format strictly. DO NOT provide extra explanations outside the format.\n"
                "Conclusion: Give the Option Letter and content."
                "Analysis: Give the reason why you chose this answer.\n"
            )
            content.append({"type": "text", "text": prompt})

            messages = [{"role": "user", "content": content}]

            input_ids, pixel_values, grid_thws = self.model.preprocess_inputs(
                messages=messages,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking
            )

            input_ids = input_ids.to(self.model.device)
            pixel_values = pixel_values.to(self.model.device, dtype=torch.bfloat16) if pixel_values is not None else None
            grid_thws = grid_thws.to(self.model.device) if grid_thws is not None else None

            with torch.no_grad():
                outputs = self.model.generate(
                    inputs=input_ids,
                    pixel_values=pixel_values,
                    grid_thws=grid_thws,
                    enable_thinking=self.enable_thinking,
                    enable_thinking_budget=True,
                    max_new_tokens=self.max_new_tokens,
                    thinking_budget=self.thinking_budget,
                    do_sample=True
                )

            full_response = self.model.text_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if "</think>" in full_response:
                final_output = full_response.split("</think>")[-1].strip()
            else:
                final_output = full_response.strip()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return final_output

        except Exception as e:
            import traceback
            return f"Runtime Error: {str(e)}\n{traceback.format_exc()}"

from transformers import AutoModelForCausalLM

from transformers import AutoModelForCausalLM, BitsAndBytesConfig

class Ovis2PredLocalEvaluator(LocalBaseEvaluator):
    def __init__(self, api_key, model_name, image_root, device="cuda"):
        super().__init__(api_key, model_name, device=device)
        self.model_path = model_name
        self.image_root = os.path.abspath(image_root) if image_root else None
        self.model = None
        self.text_tokenizer = None
        self.visual_tokenizer = None
        self._load_model()

    def _load_model(self):
        try:
            print(f"🔄 按照官方 Ovis2-16B 规范加载模型...")

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                multimodal_max_length=32768,
                trust_remote_code=True,
                device_map="auto",
                low_cpu_mem_usage=True
            ).eval()

            self.text_tokenizer = self.model.get_text_tokenizer()
            self.visual_tokenizer = self.model.get_visual_tokenizer()
            print("✅ Ovis2-16B 官方配置加载成功!")
        except Exception as e:
            raise RuntimeError(f"加载失败: {e}")

    def call(self, video_path, item):
        try:
            num_frames = 4
            max_partition = 1
            
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames <= num_frames:
                sampled_indices = range(total_frames)
            else:
                stride = total_frames / num_frames
                sampled_indices = [min(total_frames - 1, int((stride * i + stride * (i + 1)) / 2)) for i in range(num_frames)]
            
            frames = []
            for index in sampled_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ret, frame = cap.read()
                if not ret: break
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            cap.release()

            options = item.get('options', {})
            option_labels = sorted(options.keys())
            option_images = []
            for label in option_labels:
                img_path = os.path.join(self.image_root, options[label])
                if os.path.exists(img_path):
                    option_images.append(Image.open(img_path).convert('RGB'))
                else:
                    option_images.append(Image.new('RGB', (224, 224), (0, 0, 0)))

            images = frames + option_images

            video_query = '\n'.join(['<image>'] * len(frames))
            
            options_query = '\n'.join([f'Option {label}: <image>' for label in option_labels])
            
            text_instruction = (
                f"\nTask: {item['question']}\n"
                "The first set of images are frames from a video showing a physical process. "
                "The subsequent images are potential next states. Which option is correct?"
            )
            
            query = f"{video_query}\n{options_query}\n{text_instruction}"

            prompt, input_ids, pixel_values = self.model.preprocess_inputs(
                query, images, max_partition=max_partition
            )
            
            attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)
            input_ids = input_ids.unsqueeze(0).to(device=self.model.device)
            attention_mask = attention_mask.unsqueeze(0).to(device=self.model.device)
            
            if pixel_values is not None:
                pixel_values = pixel_values.to(
                    dtype=self.visual_tokenizer.dtype, 
                    device=self.visual_tokenizer.device
                )
            pixel_values = [pixel_values]

            with torch.inference_mode():
                gen_kwargs = dict(
                    max_new_tokens=512,
                    do_sample=False,
                    eos_token_id=self.model.generation_config.eos_token_id,
                    pad_token_id=self.text_tokenizer.pad_token_id,
                    use_cache=True
                )
                output_ids = self.model.generate(
                    input_ids, 
                    pixel_values=pixel_values, 
                    attention_mask=attention_mask, 
                    **gen_kwargs
                )[0]
                
                response = self.text_tokenizer.decode(output_ids, skip_special_tokens=True)
                
            return response.strip()

        except Exception as e:
            return f"Runtime Error: {str(e)}"

from openai import OpenAI

import os
import torch
from openai import OpenAI

class Qwen36VLpredAPIEvaluator(LocalBaseEvaluator):
    
    def __init__(self, api_key, model_name, image_root, device="cuda"):
        self.model_name = model_name
        self.image_root = os.path.abspath(image_root) if image_root else None
        
        self.client = OpenAI(
            api_key=api_key if api_key else "EMPTY",
            base_url="http://127.0.0.1:8000/v1"
        )
        print(f"📡 API Evaluator 启动成功: {self.model_name}")

    def call(self, video_path, item):

        try:
            content_list = []
            content_list.append({"type": "text", "text": 'The first is the initial video clip.'})
            abs_video_path = f"file://{os.path.abspath(video_path)}"
            content_list.append(
                {
                    "type": "video_url",
                    "video_url": {"url": abs_video_path},
                }
            )

            content_list.append({"type": "text", "text": 'Next are the option pictures.'})
            options = item.get('options', {})
            for label in sorted(options.keys()):
                rel_path = options[label]
                if self.image_root:
                    full_image_path = os.path.join(self.image_root, rel_path)
                    if os.path.exists(full_image_path):
                        content_list.append({"type": "text", "text": f"Option {label}:"})
                        abs_image_path = f"file://{os.path.abspath(full_image_path)}"
                        content_list.append({
                            "type": "image_url",
                            "image_url": {"url": abs_image_path}
                        })
            prompt_text = (
                "Note: The visual input contains a video and six separate option images (A, B, C, D, E, F).The last six images are options A, B, C, D, E and F in order."
                "I have provided them separately. Please do not confuse the option images with the video frames.\n"
                f"Question Context: {item['question']}\n\n"
                "You must follow the output format strictly. DO NOT provide extra explanations outside the format.\n"
                "Conclusion: Give the Option Letter and content."
                "Analysis: Give the reason why you chose this answer.\n"
            )

            content_list.append({"type": "text", "text": prompt_text})

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": content_list}],
                max_tokens=65536,
                temperature=1.0,
                top_p=0.95,
                presence_penalty=1.5,
                extra_body={
                    "top_k":20,
                    "mm_processor_kwargs": {
                        "num_frames": 8,
                    },
                    "chat_template_kwargs": {"enable_thinking": False},
                }, 
            )

            res_message = response.choices[0].message
            
            content = getattr(res_message, "content", None)
            reasoning = getattr(res_message, "reasoning", None) or getattr(res_message, "reasoning_content", None)

            if content and content.strip():
                return content.strip()
            elif reasoning:
                return f"[Reasoning Output]: {reasoning.strip()}"
            else:
                return "Error: Model returned empty response."

        except Exception as e:
            return f"API Runtime Error: {str(e)}"
