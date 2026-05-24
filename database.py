import random 
from keras.src.ops import softmax
from sklearn.model_selection import train_test_split
import streamlit as st
import pandas as pd
import json
import streamlit_drawable_canvas as drawable_canvas
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.utils import load_img, img_to_array
from PIL import Image
import os
import numpy as np
# Data processing
def get_data(images_folder, file_csv, num_classes=30, variants_class=50):
    rows=[]
    print("Đang lấy dữ liệu dể train")
    for i in range(1, num_classes + 1):
        img_name = f"{i}.png"
        img_path = os.path.join(images_folder, img_name)
        if not os.path.exists(img_path):
            print("Không đủ dữ liệu để train")
            continue
        try:
            img_original = Image.open(img_path).convert('L').resize((32, 32))
            label = i - 1
            rows.append([label] + list(np.array(img_original).flatten()))
            for u in range(variants_class - 1):
                img_var = img_original.copy()
                img_var = img_var.rotate(random.randint(-12, 12), fillcolor=255)
                scale_percent = random.randint(85,115)
                new_size = int(32*scale_percent/100)
                img_scaled = img_original.resize((new_size, new_size))
                bg = Image.new('L', (32, 32), 255)
                if scale_percent < 100:
                    offset = (new_size - 32) // 2
                    bg.paste(img_scaled, (offset, offset))
                    img_var = bg
                else:
                    start = (new_size - 32) // 2
                    img_var = img_scaled.crop((start, start, start + 32, start + 32))
                rows.append([label] + list(np.array(img_var).flatten()))
        except Exception as e:
            print("Lỗi xử lý dữ liệu")
    columns = ['label'] + [f"pixel_{idx}" for idx in range(32 * 32)]
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(file_csv, index=False)
# Train function
class EmojiModel():
    def __init__(self, model):
        self.model = None
    def build_model(self):
        self.model = Sequential()
        self.model.add(Dense(512,activation='relu'))
        self.model.add(Dense(256,activation='relu'))
        self.model.add(Dense(30,activation='softmax'))
        self.model.summary()
    def train_model(self, x_train, y_train):
        if self.model is None:
            self.build_model()
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(x_train, y_train, epochs=30, batch_size=64)

# Main page
def app():
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
        if button:
            if name.strip() != 0 and age > 0:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.subheader("Tên")
                    st.write(name)
                with col2:
                    st.subheader("Tuổi")
                    st.write(age)
                with col3:
                    #
                    st.write("Đã xử lý hình ảnh")
            else:
                st.warning("Vui lòng nhập tên tuổi trước khi dự đoán hình ảnh")
        else:
            st.info("Vui lòng vẽ và nhập đầy đủ thông tin trước khi dự đoán hình ảnh")
