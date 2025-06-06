def gen_caption(image, processor, model):
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs, min_length=10, max_length=50)
    caption = processor.decode(out[0], skip_special_tokens=True)

    return caption
