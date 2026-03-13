from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForCausalLM


def main():

    # Load dataset
    dataset = load_dataset("HuggingFaceM4/ChartQA")

    print(dataset)

    # Example sample
    sample = dataset["train"][0]

    image = sample["image"]
    question = sample["query"]
    answer = sample["label"]

    print("Question:", question)
    print("Answer:", answer)

    # Load base model
    model_name = "MODEL_NAME"

    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    print("Model and processor loaded")

    # TODO
    # preprocessing
    # fine-tuning
    # uploading to HuggingFace


if __name__ == "__main__":
    main()
