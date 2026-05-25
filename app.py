from keras.src.ops import Imag
from keras.utils import img_to_array
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
        if button:
            if canvas_result.image_data is not None and np.any(canvas_result.image_data[:, :, :3] < 255):
                if model is not None:
                    img_data = canvas_result.image_data
                    img = Image.fromarray(img_data.astype('uint8'))
                    img = img.convert('L')
                    img = img.resize((32, 32))
                    img_array = np.array(img)/ 255.0
                    img_array = img_array.reshape(1, 1024)
                    prediction = model.predict(img_array)
                    predicted_class = np.argmax(prediction)
                    if predicted_class < len(Emoji_name):
                        st.session_state.prediction_result = Emoji_name[predicted_class]
                    else:
                        st.session_state.prediction_result = "Không xác định được emoji"
                else:
                    st.error("Model không được tải thành công")
                st.session_state.user_info = {"Name": name, "Age": age}
            else:
                st.warning("Vui lòng vẽ gì đó trước khi dự đoán")
    with tab3:
        st.header("Kết quả")
        if st.session_state.prediction_result and st.session_state.user_info:
            st.write(f"**Xin chào:** {st.session_state.user_info['name']} ({st.session_state.user_info['age']} tuổi)")
            st.metric(label="Emoji được dự đoán là:", value=st.session_state.prediction_result)
        else:
            st.info("Vui lòng nhập thông tin và nhấn 'Dự đoán hình vẽ' ở Tab 'Ô nhập người dùng'.")
if __name__ == "__main__":
    app()