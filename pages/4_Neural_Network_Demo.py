import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# โหลดโมเดลที่ฝึกไว้
model = tf.keras.models.load_model("weather_classification_model.h5")

# รายการ labels (อัปเดตให้ตรงกับ dataset ของคุณ)
labels = ["dew", "fogsmog", "frost", "glaze", "hail", "lightning", "rain", "rainbow", "rime", "sandstorm", "snow"]

# ตั้งค่าหน้าเว็บ
st.title("🌤️ Weather Classification App")
st.write("อัปโหลดรูปภาพเพื่อทำนายสภาพอากาศ")

# อัปโหลดรูปภาพ
uploaded_file = st.file_uploader("📤 เลือกไฟล์ภาพ", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # แสดงภาพที่อัปโหลด
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 Uploaded Image", use_container_width=True)

    # แปลงภาพเป็นข้อมูลที่ใช้กับโมเดล
    img = image.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ทำการทำนาย
    prediction = model.predict(img_array)
    predicted_label = labels[np.argmax(prediction)]

    # แสดงผลลัพธ์
    st.subheader(f"Prediction: **{predicted_label}**")
