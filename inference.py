from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image


model_id = "username/model-name"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

image = Image.open("example.png")

inputs = processor(
    text="Describe this image",
    images=image,
    return_tensors="pt"
)

output = model.generate(**inputs)

print(processor.decode(output[0]))
