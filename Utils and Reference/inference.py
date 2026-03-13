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
