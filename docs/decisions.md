Model Choice

A small multimodal model was selected to ensure compatibility with T4 GPU compute.

Dataset Choice

ChartQA was selected because it evaluates multimodal reasoning by requiring models
to interpret chart images and answer natural language questions.

Training Strategy

LoRA fine-tuning was used to reduce GPU memory usage and training time.

Batch size was kept small due to memory constraints on T4 GPUs.
