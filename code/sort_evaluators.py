import os
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
        self.image_root = os.path.abspath(image_root) if image_root else None

    def _get_b64_from_abs_path(self, absolute_path):
        try:
            if not os.path.exists(absolute_path):
                print(f"❌ 找不到文件: {absolute_path}")
                return None
            with open(absolute_path, "rb") as f:
                return base64.b64encode(f.read()).decode('utf-8')
        except Exception as e:
            print(f"❌ 读取图片异常: {e}")
            return None

    def call(self, video_path, item):
        try:
            video_filename = os.path.basename(video_path)
            try:
                dir_num = video_filename.split('_')[-1].split('.')[0]
                img_sub_dir = os.path.join(self.image_root, f"data_{dir_num}")
            except Exception as e:
                print(f"❌ 视频路径解析失败: {video_filename}, 错误: {e}")
                return ""

            video_frames = self._extract_frames(video_path)
            
            parts = []

            parts.append({"type": "text", "text": "The first is the initial video clip:"})
            for frame_b64 in video_frames:
                parts.append({
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": frame_b64
                    }
                })

            parts.append({"type": "text", "text": "The following three images (fig1, fig2, fig3) are captured after the video ends:"})
            
            for i in range(1, 4):
                fig_path = os.path.join(img_sub_dir, f"fig{i}.jpg")
                b64_data = self._get_b64_from_abs_path(fig_path)
                if b64_data:
                    parts.append({"text": f"Figure {i}:"})
                    parts.append({
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": b64_data
                        }
                    })

            prompt_text = (
                "Note: The visual input consists of a video and three images. The last three images are fig1, fig2 and fig3 in sequence."
                "I have provided them separately. Please do not confuse the option images with the video frames.\n"
                f"Question: {item['question']}\n\n"
                "You must follow the output format strictly. DO NOT provide extra explanations outside the format.\n"
                "Conclusion: The 3-digit sequence.\n"
                "Analysis: Your physical reasoning."
            )
            parts.append({"text": prompt_text})

            url = f"https://infra.chatexcel.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
            headers = {"Content-Type": "application/json"}
            payload = {
                "contents": [{"parts": parts}],
                "generationConfig": {
                    "temperature": 0.0,
                    "maxOutputTokens": 2048
                }
            }

            response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=500)
            
            if response.status_code == 200:
                res_data = response.json()
                return res_data['candidates'][0]['content']['parts'][0]['text'].strip()
            else:
                return f"API Error ({response.status_code}): {response.text}"

        except Exception as e:
            import traceback
            return f"Runtime Error: {str(e)}\n{traceback.format_exc()}"

class BOLATUEvaluator(BaseEvaluator):
    def __init__(self, api_key, model_name, image_root):
        super().__init__(api_key, model_name)
        self.image_root = os.path.abspath(image_root)
        self.api_url = "https://one-api.bltcy.top/v1/chat/completions"

    def _get_b64_from_path(self, full_path):
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
        
        video_filename = os.path.basename(video_path)
        try:
            dir_num = video_filename.split('_')[-1].split('.')[0]
            img_sub_dir = os.path.join(self.image_root, f"data_{dir_num}")
        except Exception as e:
            print(f"❌ 视频路径解析失败: {video_filename}, 错误: {e}")
            return ""

        content_list = []
        
        content_list.append({"type": "text", "text": "The first is the initial video clip:"})
        for frame_b64 in video_frames:
            content_list.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}
            })
        
        content_list.append({"type": "text", "text": "The following four images (fig1, fig2, fig3, fig4) are captured after the video ends:"})
        for i in range(1, 5):
            fig_name = f"fig{i}.jpg"
            full_img_path = os.path.join(img_sub_dir, fig_name)
            b64 = self._get_b64_from_path(full_img_path)
            if b64:
                content_list.append({"type": "text", "text": f"Image fig{i}:"})
                content_list.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                })

        prompt_text = (
            "Note: The visual input consists of a video and four images. The last four images are fig1, fig2, fig3 and fig4 in sequence."
            "I have provided them separately. Please do not confuse the option images with the video frames.\n"
            f"Question: {item['question']}\n\n"
            "You must follow the output format strictly. DO NOT provide extra explanations outside the format.\n"
            "Conclusion: Only the 4-digit sequence.\n"
            "Analysis: Your physical reasoning."
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
                output_text = res_data['choices'][0]['message']['content']
                if "</think>" in output_text:
                    output_text = output_text.split("</think>", 1)[-1].strip()
                    if output_text.startswith("\n"):
                        output_text = output_text[1:].strip()
                else:
                    output_text = output_text.strip()
                return output_text
            else:
                print(f"请求失败 ({response.status_code}): {response.text}")
                return ""
        except Exception as e:
            print(f"API 异常: {e}")
            return ""

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
            raise RuntimeError(f"加载 Qwen-VL 模型失败: {e}")
    def call(self, video_path, item):
        try:
            from qwen_vl_utils import process_vision_info
            import torch
            import os

            video_filename = os.path.basename(video_path)
            try:
                dir_num = video_filename.split('_')[-1].split('.')[0]
                img_sub_dir = os.path.join(self.image_root, f"data_{dir_num}")
            except Exception as e:
                return f"Path Parsing Error: {e}"

            content_list = []
            
            content_list.append({"type": "text", "text": "The first is the initial video clip:"})
            content_list.append({
                "type": "video",
                "video": video_path,
                "nframes": 8,
            })

            content_list.append({"type": "text", "text": "The following four images (fig1, fig2, fig3, fig4) are captured after the video ends:"})
        
            for i in range(1, 5):
                fig_name = f"fig{i}.jpg"
                full_img_path = os.path.join(img_sub_dir, fig_name)
                if os.path.exists(full_img_path):
                    content_list.append({"type": "text", "text": f"Image fig{i}:"})
                    content_list.append({"type": "image", "image": full_img_path})
                else:
                    print(f"⚠️ 警告: 找不到图片 {full_img_path}")

            prompt_text = (
                "Note: The visual input consists of a video and four images. The last four images are fig1, fig2, fig3 and fig4 in sequence."
                "I have provided them separately. Please do not confuse the option images with the video frames.\n"
                f"Question: {item['question']}\n\n"
                "You must follow the output format strictly. DO NOT provide extra explanations outside the format.\n"
                "Conclusion: The 4-digit sequence.\n"
                "Analysis: Your physical reasoning."
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
                    max_new_tokens=2048,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.8,
                    top_k=20,
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

            return output_text

        except Exception as e:
            import traceback
            print(traceback.format_exc())
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
            video_filename = os.path.basename(video_path)
            try:
                dir_num = video_filename.split('_')[-1].split('.')[0]
                img_sub_dir = os.path.join(self.image_root, f"data_{dir_num}")
            except Exception as e:
                return f"Path Parsing Error: {e}"

            pixel_values_video, num_patches_video = self.load_video(
                video_path, num_segments=8, max_num=1 
            )
            
            pixel_values_figs = []
            num_patches_figs = []
            
            for i in range(1, 5):
                fig_name = f"fig{i}.jpg"
                full_fig_path = os.path.join(img_sub_dir, fig_name)
                if os.path.exists(full_fig_path):
                    pv_img = self.load_image(full_fig_path, max_num=1) 
                    pixel_values_figs.append(pv_img)
                    num_patches_figs.append(pv_img.shape[0])
                else:
                    print(f"⚠️ 警告: 找不到图片 {full_fig_path}")

            all_pixel_values = torch.cat([pixel_values_video] + pixel_values_figs)
            all_pixel_values = all_pixel_values.to(torch.bfloat16).to(self.model.device)
            
            total_patches_list = num_patches_video + num_patches_figs

            video_prompt = "".join([f"Video_Frame{i+1}: <image>\n" for i in range(len(num_patches_video))])
            
            figs_prompt = ""
            for i in range(len(pixel_values_figs)):
                figs_prompt += f"Figure {i+1}: <image>\n"

            question = (
                f"{video_prompt}\n"
                f"{figs_prompt}\n"
                "Note: The visual input consists of a video and four images. The last four images are fig1, fig2, fig3 and fig4 in sequence."
                "I have provided them separately. Please do not confuse the option images with the video frames.\n"
                f"Question: {item['question']}\n\n"
                "You must follow the output format strictly. DO NOT provide extra explanations outside the format.\n"
                "Conclusion: The 4-digit sequence.\n"
                "Analysis: Your physical reasoning."
            )
            
            generation_config = dict(
                max_new_tokens=2048, 
                do_sample=False,
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
            
            if "</think>" in response:
                response = response.split("</think>", 1)[-1].strip()

            return response.strip()

        except Exception as e:
            import traceback
            return f"Runtime Error: {str(e)}\n{traceback.format_exc()}"

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
            video_filename = os.path.basename(video_path)
            try:
                dir_num = video_filename.split('_')[-1].split('.')[0]
                img_sub_dir = os.path.join(self.image_root, f"data_{dir_num}")
            except Exception as e:
                return f"Path Parsing Error: {e}"

            content_list = []
            
            content_list.append({"type": "text", "text": "The first is the initial video clip:"})
            content_list.append(
                {"type": "video", "video": video_path}
            )

            content_list.append({"type": "text", "text": "The following three images (fig1, fig2, fig3) are captured after the video ends:"})

            for i in range(1, 4):
                fig_name = f"fig{i}.jpg"
                full_fig_path = os.path.join(img_sub_dir, fig_name)
                if os.path.exists(full_fig_path):
                    content_list.append({"type": "text", "text": f"Figure {i}:"})
                    content_list.append({"type": "image", "image": full_fig_path})
                else:
                    print(f"⚠️ 警告: 找不到图片 {full_fig_path}")

            prompt_text = (
                "Note: The visual input consists of a video and three images. The last three images are fig1, fig2 and fig3 in sequence."
                "I have provided them separately. Please do not confuse the option images with the video frames.\n"
                f"Question: {item['question']}\n\n"
                "You must follow the output format strictly. DO NOT provide extra explanations outside the format.\n"
                "Conclusion: The 3-digit sequence.\n"
                "Analysis: Your physical reasoning."
            )
            content_list.append({"type": "text", "text": prompt_text})

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
                    max_new_tokens=2048,
                    do_sample=False,
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

    def _extract_frame_images(self, video_path, num_frames=6):
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
            import re
            video_filename = os.path.basename(video_path)
            try:
                dir_num = re.search(r'(\d+)', video_filename).group(1)
                img_sub_dir = os.path.join(self.image_root, f"data_{dir_num}")
            except Exception as e:
                return f"Path Parsing Error: {e}"

            video_frames = self._extract_frame_images(video_path, num_frames=8)
            
            content_list = []
            images_to_process = []

            content_list.append({"type": "text", "text": "The first is the initial video clip:"})
            
            for i, frame in enumerate(video_frames):
                content_list.append({"type": "image", "image": frame})
                images_to_process.append(frame)
            
            content_list.append({"type": "text", "text": "The following three images (fig1, fig2, fig3) are captured after the video ends:"})

            for i in range(1, 4):
                fig_name = f"fig{i}.jpg"
                full_fig_path = os.path.join(img_sub_dir, fig_name)
                if os.path.exists(full_fig_path):
                    img = Image.open(full_fig_path).convert('RGB')
                    content_list.append({"type": "text", "text": f"Figure {i}:"})
                    content_list.append({"type": "image", "image": img})
                    images_to_process.append(img)
                else:
                    print(f"⚠️ 找不到文件: {full_fig_path}")

            prompt_text = (
                "Note: The visual input consists of a video and three images. The last three images are fig1, fig2 and fig3 in sequence."
                "I have provided them separately. Please do not confuse the option images with the video frames.\n"
                f"Question: {item['question']}\n\n"
                "You must follow the output format strictly. DO NOT provide extra explanations outside the format.\n"
                "Conclusion: The 3-digit sequence.\n"
                "Analysis: Your physical reasoning."
            )
            content_list.append({"type": "text", "text": prompt_text})
            
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
                    max_new_tokens=2048,
                )

            in_len = inputs.input_ids.shape[1]
            response = self.processor.batch_decode(
                generated_ids[:, in_len:], 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]

            return response.strip()

        except Exception as e:
            import traceback
            return f"Runtime Error: {str(e)}\n{traceback.format_exc()}"

from transformers import AutoProcessor, Glm4vForConditionalGeneration
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
        if total_frames <= 0: return []
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
            video_filename = os.path.basename(video_path)
            try:
                dir_num = video_filename.split('_')[-1].split('.')[0]
                img_sub_dir = os.path.join(self.image_root, f"data_{dir_num}")
            except Exception as e:
                return f"Path Parsing Error: {e}"

            video_frames = self._extract_frames(video_path, nframes=8)
            
            content = []
            content.append({"type": "text", "text": "The first is the initial video clip:"})
            
            for i, frame in enumerate(video_frames):
                content.append({"type": "image", "image": frame})
            
            content.append({"type": "text", "text": "The following four images (fig1, fig2, fig3, fig4) are captured after the video ends:"})

            for i in range(1, 5):
                fig_name = f"fig{i}.jpg"
                full_fig_path = os.path.join(img_sub_dir, fig_name)
                if os.path.exists(full_fig_path):
                    content.append({"type": "text", "text": f"Figure {i}:"})
                    content.append({"type": "image", "image": Image.open(full_fig_path).convert('RGB')})
                else:
                    print(f"⚠️ 找不到文件: {full_fig_path}")

            prompt_text = (
                "Note: The visual input consists of a video and four images. The last four images are fig1, fig2, fig3 and fig4 in sequence."
                "I have provided them separately. Please do not confuse the option images with the video frames.\n"
                f"Question: {item['question']}\n\n"
                "You must follow the output format strictly. DO NOT provide extra explanations outside the format.\n"
                "Conclusion:  The 4-digit sequence.\n"
                "Analysis: Your physical reasoning."
            )
            content.append({"type": "text", "text": prompt_text})

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
                    max_new_tokens=16384, 
                    do_sample=False,
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

from transformers import AutoModelForCausalLM
from moviepy.editor import VideoFileClip
class Ovis26PredLocalEvaluator(LocalBaseEvaluator):
    def __init__(self, api_key, model_name, image_root, device="cuda"):
        super().__init__(api_key, model_name, device=device)

        self.image_root = os.path.abspath(image_root) if image_root else None
        self.model = None
        self.enable_thinking = False
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
                quantization_config=quantization_config,
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
            video_filename = os.path.basename(video_path)
            try:
                dir_num = video_filename.split('_')[-1].split('.')[0]
                img_sub_dir = os.path.join(self.image_root, f"data_{dir_num}")
            except Exception as e:
                return f"Path Parsing Error: {e}"

            num_frames = 8

            with VideoFileClip(video_path) as clip:
                total_frames = int(clip.fps * clip.duration)
                indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
                video_frames = [Image.fromarray(clip.get_frame(t)) for t in (idx / clip.fps for idx in indices)]
            
            content = []
            content.append({"type": "text", "text": 'The first is the initial video clip.'})
            for frame in video_frames:
                content.append({"type": "image", "image": frame})
            
            content.append({"type": "text", "text": "The following four images (fig1, fig2, fig3, fig4) are captured after the video ends:"})

            for i in range(1, 5):
                fig_name = f"fig{i}.jpg"
                full_fig_path = os.path.join(img_sub_dir, fig_name)
                if os.path.exists(full_fig_path):
                    content.append({"type": "text", "text": f"Figure {i}:"})
                    content.append({"type": "image", "image": Image.open(full_fig_path).convert('RGB')})
                else:
                    print(f"⚠️ 找不到文件: {full_fig_path}")

            prompt_text = (
                "Note: The visual input consists of a video and four images. The last four images are fig1, fig2, fig3 and fig4 in sequence."
                "I have provided them separately. Please do not confuse the option images with the video frames.\n"
                f"Question: {item['question']}\n\n"
                "You must follow the output format strictly. DO NOT provide extra explanations outside the format.\n"
                "Conclusion: The 4-digit sequence.\n"
                "Analysis: Your physical reasoning."
            )
            content.append({"type": "text", "text": prompt_text})

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

import math
from transformers import AutoModel, AutoTokenizer

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
            video_filename = os.path.basename(video_path)
            try:
                import re
                dir_num = re.search(r'(\d+)', video_filename).group(1)
                img_sub_dir = os.path.join(self.image_root, f"data_{dir_num}")
            except Exception as e:
                return f"Path Parsing Error: {e}"

            video_frames, video_temporal_ids = self._extract_frames_with_ts(video_path, nframes=8)
            
            sequencing_images = []
            sequencing_temporal_ids = []
            
            for i in range(1, 5):
                fig_name = f"fig{i}.jpg"
                full_fig_path = os.path.join(img_sub_dir, fig_name)
                if os.path.exists(full_fig_path):
                    img = Image.open(full_fig_path).convert('RGB')
                    sequencing_images.append(img)
                    sequencing_temporal_ids.append(-1)
                else:
                    print(f"⚠️ Warning: File not found: {full_fig_path}")

            all_media = video_frames + sequencing_images
            combined_temporal_ids = [video_temporal_ids[0] + sequencing_temporal_ids]

            prompt_text = (
                "Note: The visual input consists of a video and four images. The last four images are fig1, fig2, fig3 and fig4 in sequence."
                "I have provided them separately. Please do not confuse the option images with the video frames.\n"
                f"Question: {item['question']}\n\n"
                "You must follow the output format strictly. DO NOT provide extra explanations outside the format.\n"
                "Conclusion: The 4-digit sequence.\n"
                "Analysis: Your physical reasoning."
            )

            msgs = [{'role': 'user', 'content': all_media + [prompt_text]}]

            output_text = self.model.chat(
                msgs=msgs,
                tokenizer=self.tokenizer,
                use_image_id=False,
                max_slice_nums=2, 
                temporal_ids=combined_temporal_ids, 
                max_new_tokens=4096,
                enable_thinking=True,
                do_sample=False
            )

            if "</think>" in output_text:
                output_text = output_text.split("</think>", 1)[-1].strip()
            
            return output_text.strip()

        except Exception as e:
            import traceback
            return f"Runtime Error: {str(e)}\n{traceback.format_exc()}"
