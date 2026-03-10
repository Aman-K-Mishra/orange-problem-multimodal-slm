# Multimodal Fine-Tuning with Small Language Models (ChartQA)

## Overview

This repository implements multimodal fine-tuning of a Small Language Model (SLM) on the ChartQA dataset.

The task is **chart question answering**.
Given a chart image and a natural language question, the model predicts the answer.

---

## Dataset

Dataset: https://huggingface.co/datasets/HuggingFaceM4/ChartQA

The dataset contains:

* `image` — chart image
* `query` — question about the chart
* `label` — correct answer

Example task:

Image: bar chart
Query: "What is the highest value?"
Answer: "45"

---

## Model

Base model: `[MODEL_NAME]`

Example options:

* SmolVLM
* LLaVA
* Phi-3 Vision

The model is selected to ensure it can run on **T4 GPU compute**.

---

## Installation

Clone the repository:

```
git clone https://github.com/YOUR_USERNAME/orange-chartqa-slm.git
cd orange-chartqa-slm
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## Training

Run training:

```
python train.py
```

Training parameters are defined in:

```
configs/training_config.yaml
```

---

## Hugging Face Model

Trained model or LoRA adapters:

```
https://huggingface.co/YOUR_USERNAME/YOUR_MODEL
```

---

## Running Inference

Example:

```python
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image

model_id = "YOUR_USERNAME/YOUR_MODEL"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

image = Image.open("example_chart.png")

question = "What is the highest value in the chart?"

inputs = processor(
    text=question,
    images=image,
    return_tensors="pt"
)

output = model.generate(**inputs)

print(processor.decode(output[0]))
```

---

## Repository Structure

```
README.md
train.py
inference.py
requirements.txt
configs/training_config.yaml
docs/decisions.md
```

---

## Hardware

Training and inference are designed to run on **NVIDIA T4 GPU**.

Possible platforms:

* Google Colab
* Kaggle

---

## Author

Aman Mishra
