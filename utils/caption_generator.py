from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

def load_image(path):
    image = Image.open(path).convert("RGB")
    return image

def gen_caption(image, model_name="Salesforce/blip-image-captioning-base"):
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)

    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)

    return caption