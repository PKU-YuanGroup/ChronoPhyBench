<p align="center">
    <h1>🔬 PhysVideo-Bench</h1>
    <h3>A Comprehensive Benchmark for Evaluating Physical Reasoning in Video-based Large Language Models</h3>
</p>

<h5 align="center">

[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](./LICENSE)
[![GitHub issues](https://img.shields.io/github/issues/your-org/PhysVideo-Bench?color=critical&label=Issues)](https://github.com/your-org/PhysVideo-Bench/issues)
[![GitHub closed issues](https://img.shields.io/github/issues-closed/your-org/PhysVideo-Bench?color=success&label=Issues)](https://github.com/your-org/PhysVideo-Bench/issues?q=is%3Aissue+is%3Aclosed)

</h5>

---

* **We introduce PhysVideo-Bench, the first comprehensive evaluation benchmark for Video-LLMs on physical reasoning, featuring a multi-level ability assessment that systematically evaluates models in video-exclusive physical understanding, prior knowledge incorporation, and physics-based prediction.**
* **We provide a unified evaluation toolkit that supports 30+ mainstream Video-LLMs — including GPT-4o, Gemini, Claude, Qwen-VL, InternVL, DeepSeek-VL, Kimi-VL, LLaVA, GLM-4V, and more — covering both API-based and locally deployed models.**
* **We design three distinct task types — Multiple-Choice QA, Image-based Outcome Prediction, and Temporal Frame Sorting — to probe models' physical reasoning capabilities from complementary angles.**
* **We conduct extensive experiments to evaluate prominent Video-LLMs, summarizing their behaviors, analyzing the root causes of their limitations in physical understanding, and proposing future directions for improvement.**

---

## 📰 News

**[2025.05]** PhysVideo-Bench is released! Evaluation code and benchmark data are publicly available.

**[2025.03]** Integrated 30+ Video-LLMs with a unified `evaluate_v4.py` interface, including support for GPT-4o, Gemini-2.5, Qwen2/3-VL, InternVL2/3.5, Claude, Kimi-VL, DeepSeek-VL2, and many more.

---

## 🎯 Benchmark Overview

PhysVideo-Bench evaluates whether Video-LLMs can **understand and reason about physical processes** depicted in real-world video clips. Unlike general video QA benchmarks, our tasks target the models' grasp of **physical laws** — gravity, collision, conservation, friction, deformation — and their ability to reason about **causal chains** in dynamic scenes.

### Task Types

| Task | Description | Output | Metric |
|------|-------------|--------|--------|
| **Multiple-Choice QA** | Watch a video and answer a physics question by selecting from 2-6 text options | Single letter (A-F) | Accuracy |
| **Outcome Prediction** | Watch a video, then predict the physical outcome by choosing among image-based options | Single letter | Accuracy |
| **Temporal Frame Sorting** | Given shuffled still frames extracted from a video, reconstruct the correct chronological order | Digit sequence (e.g., `3124`) | Exact Match |

### Ability Dimensions

The benchmark is organized into **hierarchical layers (L1, L2)** , each containing fine-grained **dimensions**:

| Layer | Focus | Example Dimension |
|-------|-------|-------------------|
| **L1 — Perceptual Understanding** | Basic physical property recognition from video | Gravity, Collision, Motion Direction, Deformation |
| **L2 — Reasoning & Prediction** | Multi-step causal reasoning and outcome forecasting | Outcome Prediction, Temporal Ordering, Counterfactual Reasoning |

---

## 🚀 Supported Models

The toolkit provides a **single unified interface** for 30+ Video-LLMs. Add a new model by implementing one evaluator class in `evaluators_v3.py`.

### API-based Models
`GPT-4o` · `GPT-4-Vision` · `Gemini-2.5-Pro` · `Gemini-2.5-Flash` · `Claude-3.5/4` · `Qwen-VL-Plus/Max` · `InternVL2-26B/76B` · `Doubao-Seed` · `Grok` · `DeepSeek-VL2` · `Kimi-VL`

### Locally Deployed Models
`Qwen2-VL (2B/7B/72B)` · `Qwen3-VL` · `InternVL2 (1B-26B)` · `InternVL3.5` · `LLaVA-1.5 (7B/13B)` · `LLaVA-NeXT` · `LLaVA-OneVision` · `LLaVA-NeXT-Video` · `LLaVA-Interleave` · `GLM-4V/4.1V/4.6V` · `Video-LLaVA` · `VideoChatGPT` · `VideoChat-Flash` · `VILA` · `PLLaVA` · `VideoLLaMA3` · `mPLUG-Owl3` · `ShareGPT4Video` · `LongCat` · `BLIP-2` · `InstructBLIP` · `MiniCPM-V` · `Step3VL` · `Ovis (2.5/2.6)`

---

## 🔧 Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-org/PhysVideo-Bench.git
cd PhysVideo-Bench
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `transformers`, `opencv-python`, `pillow`, `decord`, `qwen-vl-utils`

### 3. Data Preparation

Organize data as follows:
```
PhysVideo-Bench/
├── data/
│   ├── L1/                          # Layer 1
│   │   ├── clip_segment_0001.mp4
│   │   ├── clip_segment_0001.json   # One JSON per video
│   │   └── ...
│   ├── L2/                          # Layer 2
│   │   ├── clip_segment_0001.mp4
│   │   ├── clip_segment_0001.json
│   │   └── ...
│   └── options_data/                # Image-based answer choices
│       └── ...
```

Each JSON file structure:
```json
{
  "benchmark_metadata": {
    "video_id": "clip_segment_0001.mp4",
    "layer": "L1"
  },
  "evaluation_items": [
    {
      "id": 1,
      "question": "What will happen to the ball after it is released?",
      "options": {
        "A": "The ball will fall straight down",
        "B": "The ball will move upward"
      },
      "answer": "A",
      "dimension": "Gravity_Understanding"
    }
  ]
}
```

---

## 🏗️ Usage

### Single-GPU (choice QA)
```bash
python evaluate_v4.py \
  --model_name "qwen-vl-plus" \
  --api_key "your-api-key" \
  --json_path "./data/L1" \
  --video_dir "./data/L1" \
  --output_dir "./results"
```

### Multi-GPU distributed (torchrun)
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 evaluate_v4.py \
  --model_name "/path/to/Qwen2-VL-7B-Instruct" \
  --api_key "none" \
  --json_path "./data/L1" \
  --video_dir "./data/L1" \
  --output_dir "./results"
```

### Outcome prediction (image-based options)
```bash
python evaluate_v4.py \
  --model_name "gpt-4o" \
  --api_key "your-api-key" \
  --json_path "./data/L2" \
  --video_dir "./data/L2" \
  --image_root "./data/options_data" \
  --output_dir "./results"
```

### N/A re-inference
```bash
python evaluate_v4.py \
  --model_name "/path/to/model" \
  --api_key "none" \
  --json_path "./results/final_report_xxx.json" \
  --video_dir "./data/L1" \
  --output_dir "./results" \
  --reinfer "true"
```

### Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model_name` | Model name or local path (auto-detects model type) | Required |
| `--api_key` | API key ("none" for local models) | Required |
| `--json_path` | JSON dataset file or directory | Required |
| `--video_dir` | Root directory of video files | Required |
| `--output_dir` | Result output directory | `./results` |
| `--image_root` | Root of image-based answer options (prediction tasks) | `None` |
| `--reinfer` | Enable N/A retry mode (`true`/`false`) | `false` |

---

## 📊 Output

Each run produces a JSON report under `--output_dir`:

```json
{
  "summary": {
    "accuracy_excluding_na": 0.72,
    "total_effective": 450,
    "total_correct": 324,
    "total_na": 15,
    "detailed_stats": {
      "L1": {
        "Gravity_Understanding": {"c": 85, "t": 100, "n": 3, "acc": 0.85},
        "Collision_Analysis":    {"c": 72, "t": 100, "n": 5, "acc": 0.72}
      },
      "L2": {
        "Outcome_Prediction":    {"c": 90, "t": 150, "n": 4, "acc": 0.60},
        "Temporal_Ordering":     {"c": 77, "t": 100, "n": 3, "acc": 0.77}
      }
    }
  },
  "details": [
    {
      "video": "clip_segment_0001.mp4",
      "layer": "L1",
      "dimension": "Gravity_Understanding",
      "question": "...",
      "options": {"A": "...", "B": "..."},
      "gt": "A",
      "pred": "A",
      "is_correct": true,
      "is_na": false,
      "response": "1. Conclusion: A. ...\n2. Analysis: ..."
    }
  ]
}
```

> **Note**: N/A answers (model refusal or extraction failure) are excluded from the accuracy denominator. A separate `pending_na_*.json` file is generated for later re-inference.

---

## 🛠️ Code Architecture

| File | Purpose |
|------|---------|
| `evaluate_v4.py` | **Main entry point** — model dispatch, multi-GPU orchestration, command-line interface |
| `evaluators_v3.py` | Evaluator classes for all 30+ supported models (API + local), video frame extraction, unified `.call(video_path, item)` interface |
| `predict_evaluators.py` | Specialized evaluators for image-option outcome prediction tasks |
| `sort_evaluators.py` | Evaluators for temporal frame sorting tasks |
| `utils.py` | Answer extraction: regex-based parsing for both choice QA and sorting output |
| `utils_1.py` | Core evaluation loop, multi-GPU result merging, N/A retry pipeline, checkpoint/resume logic |
| `utils_predict.py` | Prediction-task utilities: evaluation loop, result aggregation, report generation |

---

## ✨ Features

- **Auto Model Detection**: The `--model_name` string is parsed to automatically select the correct evaluator class (e.g., `qwen` → `QwenLocalEvaluator`, `internvl` → `InternVLLocalEvaluator`, `gemini` → `GeminiEvaluator`)
- **Checkpoint & Resume**: Evaluations survive crashes. Each GPU writes partial results to `tmp_{model}_{dataset}_rank_{N}.json` after every video. Restarting the same command resumes from where it left off.
- **Dynamic GPU Count**: Change `--nproc_per_node` between runs — the system scans all historical temp files and redistributes remaining tasks.
- **N/A Filtering**: Responses the model fails to produce are logged separately and excluded from accuracy metrics. A `--reinfer` mode targets only failed cases.
- **Hierarchical Statistics**: Automatic per-layer and per-dimension accuracy breakdown printed in a formatted table.

---

## 📝 License

PhysVideo-Bench is released under Apache License Version 2.0.

## 🤝 Citation

```bibtex
@article{physvideo-bench2025,
  title={PhysVideo-Bench: A Comprehensive Benchmark for Evaluating Physical Reasoning in Video-based Large Language Models},
  author={...},
  journal={arXiv preprint},
  year={2025}
}
```
