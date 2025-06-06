import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from utils.caption_generator import gen_caption
from utils.images import load_image

st.title("メニュー推定アプリ")

uploaded_file = st.file_uploader(
    "Choose image",
    accept_multiple_files=False,
    type=["jpg", "png", "jpeg"]
)

model_name = "Salesforce/blip-image-captioning-base"
processor = BlipProcessor.from_pretrained(model_name)
model = BlipForConditionalGeneration.from_pretrained(model_name)

if uploaded_file:
    image = load_image(uploaded_file)
    st.image(image, caption="アップロードされた画像", use_container_width=True)

    with st.spinner("キャプション生成中", show_time=True):
        caption = gen_caption(image, processor, model)

    st.markdown("### 入力画像のキャプション:")
    st.write(caption)
