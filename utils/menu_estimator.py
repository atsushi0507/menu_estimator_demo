from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch


def est_menu(image, model_name="Salesforce/blip2-opt-2.7b"):
    processor = Blip2Processor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu"
    )

    # question = "この画像に写っている料理は何ですか？"
    question = "What's in the dish in this image?"

    inputs = processor(image, question, return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=50
    )
    answer = processor.decode(generated_ids[0], skip_special_tokens=True)

    return answer