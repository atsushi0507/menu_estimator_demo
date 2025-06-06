import streamlit as st
# from transformers import BlipProcessor, BlipForConditionalGeneration
from utils.caption_generator import load_image, gen_caption
import torch

st.title("メニュー推定アプリ")

uploaded_file = st.file_uploader(
    "Choose image",
    accept_multiple_files=False,
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:
    image = load_image(uploaded_file)
    st.image(image, caption="アップロードされた画像", use_container_width=True)

    with st.spinner("キャプション生成中", show_time=True):
        caption = gen_caption(image)

    st.markdown("### 推定メニュー:")
    st.write(caption)