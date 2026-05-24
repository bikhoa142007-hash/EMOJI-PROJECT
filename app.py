import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import streamlit_drawable_canvas as drawable_canvas

@st.cache_resource
def load_trained_model():
    return tf.keras.models.load_model("emoji_dense_model.keras")

model = load_trained_model()
Emoji_name = [
    "Grinning Face",
    "Grinning Face With Big Eyes",
    "Grinning Face With Smiling Eyes",
    "Beaming Face With Smiling Eyes",
    "Grinning Squinting Face",
    "Grinning Face With Sweat",
    "Rolling On The Floor Laughing",
    "Face With Tears Of Joy",
    "Slightly Smiling Face",
    "Upside-down Face",
    "Melting Face",
    "Winking Face",
    "Smiling Face With Smiling Eyes",
    "Smiling Face With Halo",
    "Smiling Face With Hearts",
    "Smiling Face With Heart-eyes",
    "Star-struck",
    "Face Blowing A Kiss",
    "Kissing Face",
    "Smiling Face",
    "Kissing Face With Closed Eyes",
    "Kissing Face With Smiling Eyes",
    "Smiling Face With Tear",
    "Face Savoring Food",
    "Face With Tongue",
    "Winking Face With Tongue",
    "Zany Face",
    "Squinting Face With Tongue",
    "Money-mouth Face",
    "Smiling Face With Open Hands"
]
def app():
    st.title("App Emoji")
    tab1, tab2, tab3 = st.tabs(["Bảng vẽ", "Ô nhập người dùng", "Bảng kết quả"])
    with tab1:
        st.header("Bảng vẽ")
        st.write("Đây là nơi bạn có thể vẽ hình ảnh của mình.")
        # Art tool
        canvas_result = drawable_canvas.st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=2,
            stroke_color="#000000",
            background_color="#ffffff",
            height=400,
            width=400,
            drawing_mode="freedraw",
            key="canvas"
        )
    with tab2:
        st.header("Ô nhập người dùng")
        st.write("Nhập thông tin của bạn vào đây.")
        name = st.text_input("Tên của bạn")
        age = st.number_input("Tuổi của bạn", min_value=0, max_value=120)
        button = st.button("Dự đoán hình vẽ")
    with tab3:
        st.header("Kết quả")
def main():
    if __name__ == "__main__":
        app()