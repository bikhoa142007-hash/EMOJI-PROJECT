import random 
from keras.src.ops import softmax
import streamlit as st
import pandas as pd
import json
import streamlit_drawable_canvas as drawable_canvas
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout
from keras.utils import to_categorical
from keras.utils import load_img, img_to_array
from PIL import Image
import os
import numpy as np

def get_data(images_folder, file_csv, num_classes=30, variants_class=50):
    rows=[]
    print("Đang lấy dữ liệu để train...")
    for i in range(1, num_classes + 1):
        img_name = f"{i}.png"
        img_path = os.path.join(images_folder, img_name)
        if not os.path.exists(img_path):
            print(f"Không tìm thấy file: {img_path}")
            continue
        try:
            img_original = Image.open(img_path).convert('L').resize((32, 32))
            label = i - 1
            rows.append([label] + list(np.array(img_original).flatten()))
            for u in range(variants_class - 1):
                img_var = img_original.copy()
                img_var = img_var.rotate(random.randint(-12, 12), fillcolor=255)
                scale_percent = random.randint(85, 115)
                new_size = int(32 * scale_percent / 100)
                img_scaled = img_original.resize((new_size, new_size))
                bg = Image.new('L', (32, 32), 255)
                if scale_percent < 100:
                    offset = (32 - new_size) // 2
                    bg.paste(img_scaled, (offset, offset))
                    img_var = bg
                else:
                    start = (new_size - 32) // 2
                    img_var = img_scaled.crop((start, start, start + 32, start + 32))
                rows.append([label] + list(np.array(img_var).flatten()))
        except Exception as e:
            print(f"Lỗi xử lý file {img_name}: {e}")
            
    if len(rows) == 0:
        print("CẢNH BÁO: Không tìm thấy ảnh nào! Hãy kiểm tra lại thư mục chứa ảnh.")
        return
        
    columns = ['label'] + [f"pixel_{idx}" for idx in range(32 * 32)]
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(file_csv, index=False)
    print(f"Đã tạo xong file CSV với {len(rows)} dòng dữ liệu.")

class EmojiModel():
    def __init__(self):
        self.model = None
    def build_model(self):
        self.model = Sequential()
        self.model.add(Input(shape=(1024,)))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(30, activation='softmax'))
        self.model.summary()
    def train_model(self, x_train, y_train):
        if self.model is None:
            self.build_model()
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(x_train, y_train, epochs=30, batch_size=64)

if __name__ == "__main__":
    # Thay vì dùng "./emoji", bạn điền thẳng đường dẫn tuyệt đối từ ổ C của máy bạn vào đây:
    emoji_folder = r"C:\Users\Khoa Bi\source\repos\EMOJI PROJECT\emoji"
    
    # Tương tự với file CSV, lưu thẳng vào thư mục dự án:
    File_csv = r"C:\Users\Khoa Bi\source\repos\EMOJI PROJECT\file_csv.csv"
    
    # Gọi hàm quét ảnh và tạo biến thể dữ liệu
    get_data(images_folder=emoji_folder, file_csv=File_csv, num_classes=30, variants_class=50)
    
    # Kiểm tra nếu file CSV đã được tạo và có dữ liệu thì tiến hành train
    if os.path.exists(File_csv) and os.path.getsize(File_csv) > 100:
        df = pd.read_csv(File_csv)
        y_train = df['label'].values
        x_train = df.drop(columns=['label']).values
        
        x_train_scaled = x_train.astype('float32') / 255.0
        y_train_scaled = to_categorical(y_train, num_classes=30)
        
        my_emoji = EmojiModel()
        my_emoji.train_model(x_train_scaled, y_train_scaled)
        
        # Lưu file model .keras cứng vào thư mục dự án
        model_save_path = r"C:\Users\Khoa Bi\source\repos\EMOJI PROJECT\emoji_dense_model.keras"
        my_emoji.model.save(model_save_path)
        print(f"Đã lưu model thành công tại: {model_save_path}")