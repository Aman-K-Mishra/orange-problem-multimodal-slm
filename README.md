# Multimodal Fine-Tuning with Small Language Models (Orange Problem)

## Overview

This repository contains the implementation for multimodal fine-tuning of a Small Language Model (SLM) using image-text data.

The goal is to train a model that can understand an image and generate a text response.

Dataset used: **[(ChartQA)](https://huggingface.co/datasets/HuggingFaceM4/ChartQA)**

Example tasks:

* Chart question answering
* Image caption generation

---

## Dataset

Dataset Source:
[(ChartQA)](https://huggingface.co/datasets/HuggingFaceM4/ChartQA)

Dataset Columns:

For ChartQA:

* `image`
* `query`
* `label`

For RICO-Screen2Words:

* `image`
* `caption`

---

## Model

Base Model: `[Model Name]`

Example options:

* SmolVLM
* Phi-3 Vision
* LLaVA

Reason for selection:

* Small enough to run on **T4 GPU**
* Supports **multimodal inputs**

---

## Installation

Clone the repository:

```
git clone https://github.com/yourusername/orange-problem-multimodal-slm.git
cd orange-problem-multimodal-slm
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## Training

Run the training script:

```
python train.py
```

Training configuration is stored in:

```
configs/training_config.yaml
```

---

## Model Upload

The trained model or LoRA adapters are uploaded to Hugging Face:

```
HuggingFace Model Link
```

---

## Running Inference

Example:

```python
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image

model_id = "username/model-name"

model = AutoModelForCausalLM.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

image = Image.open("example.png")

inputs = processor(
    text="Describe this image",
    images=image,
    return_tensors="pt"
)

output = model.generate(**inputs)

print(processor.decode(output[0]))
```

---

## Repository Structure

```
train.py            # model fine-tuning
inference.py        # inference script
requirements.txt    # dependencies
configs/            # training configs
docs/               # design decisions
examples/           # inference examples
```

---

## Hardware

Training and inference were designed to run on **NVIDIA T4 GPU**.

Platforms used:

* Google Colab
* Kaggle

---

## Author

Aman Mishra
