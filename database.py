import random
import streamlit as st
import pandas as pd
import os
import numpy as np
from PIL import Image, ImageEnhance
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout, BatchNormalization
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Data proccesing
def get_data(images_folder, file_csv, num_classes=30, variants_class=200):
    rows = []
    print("Đang lấy dữ liệu để train...")
    for i in range(1, num_classes + 1):
        img_name = f"{i}.png.png"
        img_path = os.path.join(images_folder, img_name)
        if not os.path.exists(img_path):
            print(f"Không tìm thấy file: {img_path}")
            continue
        try:
            img_original = Image.open(img_path).convert('RGB').resize((32, 32))
            label = i - 1
            rows.append([label] + list(np.array(img_original).flatten()))
            for _ in range(variants_class - 1):
                img_var = img_original.copy()
                angle = random.randint(-15, 15)
                img_var = img_var.rotate(angle, fillcolor=(255, 255, 255))
                scale = random.uniform(0.85, 1.15)
                new_size = int(32 * scale)
                img_scaled = img_var.resize((new_size, new_size))
                bg = Image.new('RGB', (32, 32), (255, 255, 255))
                if scale < 1.0:
                    offset = (32 - new_size) // 2
                    bg.paste(img_scaled, (offset, offset))
                else:
                    start = (new_size - 32) // 2
                    img_cropped = img_scaled.crop((start, start, start + 32, start + 32))
                    bg.paste(img_cropped, (0, 0))
                img_var = bg
                enhancer = ImageEnhance.Brightness(img_var)
                img_var = enhancer.enhance(random.uniform(0.85, 1.15))

                rows.append([label] + list(np.array(img_var).flatten()))
        except Exception as e:
            print(f"Lỗi xử lý file {img_name}: {e}")
    if len(rows) == 0:
        print("Không tìm thấy ảnh nào!")
        return
    columns = ['label'] + [f"pixel_{idx}" for idx in range(32 * 32 * 3)]
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(file_csv, index=False)
# Train model
class EmojiModel():
    def __init__(self):
        self.model = None
    def build_model(self):
        self.model = Sequential([
            Input(shape=(3072,)),          
            Dense(1024, activation='relu'),
            BatchNormalization(),         
            Dropout(0.4),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(256, activation='relu'),
            Dropout(0.2),
            Dense(30, activation='softmax')
        ])
        self.model.summary()
    def train_model(self, x_train, y_train):
        if self.model is None:
            self.build_model()
        self.model.compile(
            optimizer='adam',           
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5)  
        ]
        self.model.fit(
            x_train, y_train,
            epochs=100,                    
            batch_size=32,
            validation_split=0.1,
            callbacks=callbacks
        )
# Input
if __name__ == "__main__":
    emoji_folder = r"C:\Users\Khoa Bi\source\repos\EMOJI PROJECT\emoji"
    File_csv = r"C:\Users\Khoa Bi\source\repos\EMOJI PROJECT\file_csv.csv"
    get_data(images_folder=emoji_folder, file_csv=File_csv, num_classes=30, variants_class=200)
    if os.path.exists(File_csv) and os.path.getsize(File_csv) > 100:
        df = pd.read_csv(File_csv)
        y_train = df['label'].values
        x_train = df.drop(columns=['label']).values
        x_train_scaled = x_train.astype('float32') / 255.0
        y_train_scaled = to_categorical(y_train, num_classes=30)
        my_emoji = EmojiModel()
        my_emoji.train_model(x_train_scaled, y_train_scaled)
        model_save_path = r"C:\Users\Khoa Bi\source\repos\EMOJI PROJECT\emoji_dense_model.keras"
        my_emoji.model.save(model_save_path)
        print(f"Đã lưu model: {model_save_path}")