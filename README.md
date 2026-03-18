# 🍊 Orange Problem — ChartQA Multimodal Fine-Tuning with SLMs

**Team:** Langrangers
| Name | USN |
|---|---|
| Aaron Thomas Mathew | PES1UG23AM005 |
| Aman Kumar Mishra | PES1UG23AM040 |
| Preetham VJ | PES1UG23AM913 |

---

## Overview

This repository implements multimodal fine-tuning of a Small Language Model (SLM) on the [ChartQA dataset](https://huggingface.co/datasets/HuggingFaceM4/ChartQA).

**Task:** Chart Question Answering — given a chart image and a natural language question, the model predicts the answer.

**Model:** [Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) fine-tuned with LoRA adapters  
**Fine-tuned Model on HuggingFace:** [aaronmat1905/Qwen2VL-finetuned-chartqa](https://huggingface.co/aaronmat1905/Qwen2VL-finetuned-chartqa)  
**Processed Dataset:** [aaronmat1905/chartqa-processed](https://huggingface.co/datasets/aaronmat1905/chartqa-processed)

---

## Pipeline Overview

```
ChartQA Dataset
      │
      ▼
01_data_exploration_preprocessing.ipynb
  - Load & visualize dataset
  - Resize images to 448×448
  - Format prompt templates
  - Push processed dataset to HuggingFace
      │
      ▼
02_training.ipynb
  - Load Qwen2-VL-2B-Instruct (8-bit quantized)
  - Apply LoRA adapters (rank=16)
  - Train on 28,299 samples (1 epoch)
  - Mid-epoch checkpointing to Google Drive
  - Push adapters to HuggingFace
      │
      ▼
03_evaluation_inference.ipynb
  - Pull model/adapters from HuggingFace
  - Merge LoRA adapters with base model
  - Evaluate on test split (Relaxed Accuracy)
  - Run qualitative inference examples
```

---

## Repository Structure

```
orange-chartqa-slm/
├── README.md
├── 01_data_exploration_preprocessing.ipynb
├── 02_training.ipynb
├── 03_evaluation_inference.ipynb
└── Utils and Reference/
```

---

## Dataset

**Source:** [HuggingFaceM4/ChartQA](https://huggingface.co/datasets/HuggingFaceM4/ChartQA)

| Split | Samples |
|---|---|
| Train | 28,299 |
| Validation | 1,920 |
| Test | 2,500 |

**Columns used:** `image`, `query`, `label`  
(`human_or_machine` column available but not used for training)

**Example:**
```
Image : bar chart showing survey data
Query : "Is the value of Favorable 38 in 2015?"
Label : "Yes"
```

---

## Model

### Why Qwen2-VL-2B-Instruct?
- Native multimodal support (image + text inputs) out of the box
- 2B parameters — small enough to fine-tune on a T4 GPU (15.6 GB VRAM)
- Strong instruction-following base for chart reasoning tasks
- Compatible with LoRA fine-tuning via HuggingFace PEFT

---

## Key Design Decisions

### Image Resolution: 448×448 (MIN: 256 patches, MAX: 512 patches)
Images are resized using Qwen2-VL's dynamic resolution system:
```python
MIN_PIXELS = 256 * 28 * 28   # minimum detail for chart reading
MAX_PIXELS = 512 * 28 * 28   # ~400k pixels — safe for T4 VRAM
```
The 28×28 patch size matches Qwen2-VL's `PatchEmbed` Conv3d kernel. 512 patches is the T4-safe upper limit while retaining enough resolution to read axis labels and data values in charts.

### 8-bit Quantization (BitsAndBytes)
```python
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False
)
```
- Full precision (fp32) Qwen2-VL-2B ≈ 16 GB VRAM — exceeds T4 capacity
- 8-bit quantization ≈ 8 GB VRAM, leaving headroom for gradients and activations
- Accuracy tradeoff is minimal since LoRA only trains adapters, not the quantized base weights

### LoRA Configuration
```python
LORA_RANK    = 16    # rank=8 too small for chart reasoning; rank=32 risks OOM
LORA_ALPHA   = 32    # alpha = 2×rank (standard rule of thumb)
LORA_DROPOUT = 0.05  # light dropout to prevent adapter overfitting
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
```
- Only 4.36M trainable parameters out of 2.21B total (0.20%)
- q/k/v/o projections are the most impactful attention layers for VLMs
- Enables ~10× memory reduction vs full fine-tuning

### Training Hyperparameters
```python
EPOCHS           = 1
BATCH_SIZE       = 1      # reduced from 2 — OOM fix for T4
GRAD_ACCUM_STEPS = 16     # effective batch size = 16 (same as batch=2, accum=8)
LEARNING_RATE    = 2e-4
MAX_LENGTH       = 768    # compromise: 512 too short, 1024 causes OOM
```
Gradient accumulation over 16 steps simulates a larger effective batch size while keeping per-step memory usage within T4 limits. Cosine annealing LR scheduler used over the full epoch.

### Label Masking Strategy
To compute loss only on the answer tokens (not the prompt):
- Sequence structure: `[prompt][answer][EOS][PAD...]`
- Tokenize the answer separately to get `n_answer` tokens
- Find last EOS position in `input_ids`
- Mask everything except `input_ids[answer_end - n_answer : answer_end]` with `-100`

### Prompt Format
```
<|im_start|>user
<image>
{query}<|im_end|>
<|im_start|>assistant
{label}<eos>
```
Matches Qwen2-VL's instruction-tuned chat template, allowing the model to learn answer generation in the format it was pre-trained to expect.

---

## Hardware

- **GPU:** NVIDIA Tesla T4 (15.6 GB VRAM)
- **Platform:** Google Colab / Kaggle
- **Training time:** ~6–7 hours for 1 epoch on 28,299 samples

---

## Installation

```bash
git clone https://github.com/Aman-K-Mishra/orange-chartqa-slm
cd orange-chartqa-slm
pip install transformers peft bitsandbytes accelerate datasets pillow
```

---

## Quick Inference

```python
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel
from PIL import Image
import torch
import requests
from io import BytesIO

BASE_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
ADAPTER_REPO  = "preethamvj/chart-vision-qwen"

# Load base model (8-bit)
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16
)

# Load and merge LoRA adapters
model = PeftModel.from_pretrained(model, ADAPTER_REPO)
model = model.merge_and_unload()  # merges adapters into base weights

processor = AutoProcessor.from_pretrained(BASE_MODEL_ID,
                                          min_pixels=256*28*28,
                                          max_pixels=512*28*28)

# Run inference
image = Image.open("your_chart.png").convert("RGB")
question = "What is the highest value in the chart?"

messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": image},
        {"type": "text",  "text": question}
    ]
}]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=64)

answer = processor.decode(output[0], skip_special_tokens=True)
print("Answer:", answer.split("assistant")[-1].strip())
```

> **Note:** If you encounter memory issues, use `load_in_8bit=True` in BitsAndBytesConfig as shown above.

---

## Notebooks

| Notebook | Description |
|---|---|
| `01_data_exploration_preprocessing.ipynb` | Dataset loading, visualization, image resizing, prompt formatting, push to HF |
| `02_training.ipynb` | 8-bit model loading, LoRA setup, custom Dataset/DataLoader, training loop with checkpointing, HF push |
| `03_evaluation_inference.ipynb` | Load adapters from HF, merge with base, evaluate on test split, qualitative examples |

---

## Results

| Metric | Value |
|---|---|
| Training samples | 28,299 |
| Trainable parameters | 4.36M (0.20%) |
| Epochs | 1 |
| Hardware | Tesla T4 (15.6 GB) |

*(Full evaluation results available in `03_evaluation_inference.ipynb`)*

---

## Acknowledgements

- [Qwen2-VL](https://github.com/QwenLM/Qwen2-VL) by Alibaba Cloud
- [ChartQA Dataset](https://huggingface.co/datasets/HuggingFaceM4/ChartQA) by HuggingFaceM4
- [PEFT](https://github.com/huggingface/peft) by HuggingFace
