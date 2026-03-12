# Multimodal Fine-Tuning with Small Language Models (ChartQA)

## Overview

This repository implements multimodal fine-tuning of a Small Language Model (SLM) on the ChartQA dataset.

The task is **chart question answering**.
Given a chart image and a natural language question, the model predicts the answer.

---

## Pipeline Overview

The project follows a three-stage pipeline:

1. **Data Exploration & Preprocessing**
   - Load the ChartQA dataset
   - Visualize chart images and question–answer pairs
   - Resize images and format prompts for multimodal training

2. **Model Fine-Tuning**
   - Load the base vision-language model
   - Apply LoRA adapters for efficient fine-tuning
   - Train on the ChartQA dataset

3. **Evaluation & Inference**
   - Evaluate the model on validation/test splits
   - Run inference on chart images
   - Generate answers to chart-related questions
  
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

Data exploration and preprocessing are performed in:
```
notebooks/01_data_exploration_preprocessing.ipynb
```

This notebook loads the dataset, visualizes chart examples, and prepares the image-text inputs for model training.

---

## Preprocessing Decisions

Several preprocessing steps are applied before training:

**Image Resizing**

Chart images are resized to **448 × 448 pixels** to match the input requirements of the vision-language model.

**Prompt Formatting**

Each dataset example is formatted as:

Question: <query>  
Answer: <label>

This structure aligns with instruction-tuned multimodal models and helps the model learn to generate answers to chart questions.

**Dataset Columns Used**

The dataset includes the following columns:

- image
- query
- label
- human_or_machine

Only **image**, **query**, and **label** are used for training.

---

## Model

Base model: Qwen2-VL-2B-Instruct

This model was selected because it:
- supports multimodal (image + text) inputs
- is small enough to run on a T4 GPU
- works well with LoRA fine-tuning

---

## Installation

Clone the repository:

```
git clone https://github.com/Aman-K-Mishra/orange-chartqa-slm
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
https://huggingface.co/YOUR_USERNAME/chartqa-qwen2vl-lora
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
requirements.txt

train.py
inference.py

notebooks/
  01_data_exploration_preprocessing.ipynb
  02_training.ipynb
  03_evaluation.ipynb

configs/
  training_config.yaml

docs/
  decisions.md
```

---

## Hardware

Training and inference are designed to run on **NVIDIA T4 GPU**.

Possible platforms:

* Google Colab
* Kaggle

---

## Authors

- Aman Kumar Mishra

- Aaron Thomas Mathew

- Preetham VJ
