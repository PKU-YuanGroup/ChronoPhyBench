import json
import re
import os
import argparse
from datetime import datetime

import os

raw_visible = os.environ.get("CUDA_VISIBLE_DEVICES")

if raw_visible:
    visible_list = raw_visible.split(',')
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    gpus_per_process = 1
    
    start_idx = local_rank * gpus_per_process
    end_idx = start_idx + gpus_per_process
    my_gpu_slice = visible_list[start_idx:end_idx]
    
    if my_gpu_slice:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(my_gpu_slice)
        print(f"--- [Rank {local_rank}] 智能切分: 物理卡 {os.environ['CUDA_VISIBLE_DEVICES']} ---")
    else:
        print(f"--- [Rank {local_rank}] 警告: 显卡不够分了！ ---")
else:
    print("--- 提示: 未在命令行指定显卡，将尝试使用默认环境 ---")

import torch

import torch.distributed as dist
import torch

from evaluators_v3 import *
from sort_evaluators import *
from predict_evaluators import *

from utils_predict import *
# from utils_1 import extract_answer, run_evaluation, merge_results_and_report, retry_na_simple
torch.cuda.empty_cache()

def main():
    if "RANK" in os.environ:
        from datetime import timedelta
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(minutes=10)
        )
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
    else:
        local_rank = 0
        world_size = 1
        rank = 0
        print("⚠️ 未检测到 torchrun 环境，将以单卡模式运行")

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        torch.cuda.set_device(0) 
    else:
        torch.cuda.set_device(local_rank)
    current_device = f"cuda:{local_rank}"
    current_device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser(description="物理视频大模型评测工具 V2.0")
    parser.add_argument("--model_name", type=str, help="具体模型名称 (如 internvl2-26b, qwen-vl-plus)")
    parser.add_argument("--api_key", type=str, required=True, help="API密钥")
    parser.add_argument("--json_path", type=str, required=True, help="评测JSON数据集路径")
    parser.add_argument("--video_dir", type=str, required=True, help="视频文件存放目录")
    parser.add_argument("--output_dir", type=str, default="./results", help="结果保存目录")
    parser.add_argument("--reinfer", type=str, default="false", help="是否进入N/A重推理模式 (true/false)")
    parser.add_argument("--image_root", type=str, default=None, help="图片选项大文件夹的路径 (仅用于多模态选项任务)")
    args = parser.parse_args()

    m_name_lower = args.model_name.lower()
    local_model = False
    if rank == 0:
        print(f"📡 总进程数: {world_size} | 正在加载模型: {args.model_name}")

    if 'local' in m_name_lower or '/' in args.model_name:
        local_model = True
        print(f"🔍 检测到本地部署模型模式")

    if 'internvl' in m_name_lower:
        if local_model:
            if 'internvl3_5' in m_name_lower:
                if args.image_root is not None:
                    engine = InternVL35predLocalEvaluator(args.api_key, model_name=args.model_name, image_root=args.image_root, device=current_device)
                else:
                    engine = InternVL35LocalEvaluator(args.api_key, model_name=args.model_name, device=current_device)
            else:
                if args.image_root is not None:
                    engine = InternVLpredLocalEvaluator(args.api_key, model_name=args.model_name, image_root=args.image_root, device=current_device)
                else:
                    engine = InternVLLocalEvaluator(args.api_key, model_name=args.model_name, device=current_device)
            print(f"🚀 识别成功，正在使用本地 InternVL 模型: {args.model_name}")
        else:
            engine = InternVLEvaluator(args.api_key, model_name=args.model_name)
            print(f"🚀 识别成功，正在使用 InternVL API 模型: {args.model_name}")
    elif 'qwen' in m_name_lower or 'mimo' in m_name_lower:
        if local_model:
            if args.image_root is not None:
                if '3.5' in m_name_lower or '3.6' in m_name_lower:
                    engine = Qwen36VLpredAPIEvaluator(args.api_key, model_name=args.model_name, image_root=args.image_root, device=current_device)
                else:
                    engine = QwenVLpredLocalEvaluator(args.api_key, model_name=args.model_name, image_root=args.image_root, device=current_device)
            else:
                engine = QwenLocalEvaluator(args.api_key, model_name=args.model_name, device=current_device)
            print(f"🚀 识别成功，正在使用本地 Qwen 模型: {args.model_name}")
        else:
            engine = BOLATUEvaluator(args.api_key, model_name=args.model_name, image_root=args.image_root)
            print(f"🚀 识别成功，正在使用 Qwen API 模型: {args.model_name}")
    elif 'gpt' in m_name_lower:
        engine = BOLATUEvaluator(args.api_key, model_name=args.model_name, image_root=args.image_root)
        print(f"🚀 识别成功，正在使用 GPT 模型: {args.model_name}")
    elif 'seed' in m_name_lower:
        engine = BOLATUEvaluator(args.api_key, model_name=args.model_name, image_root=args.image_root)
        print(f"🚀 识别成功，正在使用 Doubao 模型: {args.model_name}")
    elif 'gemini' in m_name_lower:
        if args.image_root is not None:
            engine = BOLATUEvaluator(args.api_key, model_name=args.model_name, image_root=args.image_root)
        else:
            engine = GeminiEvaluator(args.api_key, model_name=args.model_name)
        print(f"🚀 识别成功，正在使用 Gemini 模型: {args.model_name}")
    elif 'glm' in m_name_lower:
        if local_model:
            if '4.1' in m_name_lower:
                if args.image_root is not None:
                    engine = GLM46VpredLocalEvaluator(args.api_key, model_name=args.model_name, image_root=args.image_root, device=current_device)
                else:
                    engine = GLM41VLocalEvaluator(args.api_key, model_name=args.model_name, device=current_device)
            else:
                if args.image_root is not None:
                    engine = GLM46VpredLocalEvaluator(args.api_key, model_name=args.model_name, image_root=args.image_root, device=current_device)
                else:
                    engine = GLM46VLocalEvaluator(args.api_key, model_name=args.model_name, device=current_device)
            print(f"🚀 识别成功，正在使用本地 GLM 模型: {args.model_name}")
    elif 'kimi' in m_name_lower:
            engine = KimiVLpredEvaluator(args.api_key, model_name=args.model_name, image_root=args.image_root,device=current_device)
            print(f"🚀 识别成功，正在使用 kimi vl 模型: {args.model_name}")
    elif 'minicpm' in m_name_lower:
            engine = MiniCPM45VpredLocalEvaluator(args.api_key, model_name=args.model_name, image_root=args.image_root, device=current_device)
            print(f"🚀 识别成功，正在使用 minicpm 模型: {args.model_name}")
    elif 'ovis' in m_name_lower:
        if '2.6' in m_name_lower or '2.5' in m_name_lower:
            engine = Ovis26PredLocalEvaluator(args.api_key, model_name=args.model_name, image_root=args.image_root, device=current_device)
        else:
            engine = Ovis2PredLocalEvaluator(args.api_key, model_name=args.model_name, image_root=args.image_root, device=current_device)
            print(f"🚀 识别成功，正在使用 ovis 模型: {args.model_name}")
    else:
        print(f"❌ 错误: 无法识别的模型型号 '{args.model_name}")
        return


    if args.image_root is not None:
        run_evaluation_pred(args, engine, rank, world_size)
        if rank == 0:
            merge_results_and_report_pred(args)
    else:
        run_evaluation(args, engine, rank, world_size)
            
        merge_results_and_report(args, world_size, rank)

    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
