import streamlit as st
import joblib
import numpy as np

# โหลดโมเดลที่บันทึกไว้
model = joblib.load("xgboost_house_price_model.pkl")
scaler = joblib.load("scaler.pkl")

# ส่วนติดต่อของเว็บแอป (UI)
st.title("🏠 พยากรณ์ราคาบ้านด้วย XGBoost")

# รับค่าคุณสมบัติของบ้านจากผู้ใช้
area = st.number_input("📏 ขนาดพื้นที่บ้าน (ตร.ม.)", min_value=10, max_value=10000, value=100)
bedrooms = st.slider("🛏️ จำนวนห้องนอน", 1, 10, 3)
bathrooms = st.slider("🚿 จำนวนห้องน้ำ", 1, 5, 2)
stories = st.slider("🏢 จำนวนชั้น", 1, 3, 1)
parking = st.slider("🚗 จำนวนที่จอดรถ", 0, 5, 1)

# ต้องเพิ่มข้อมูลที่หายไปเช่น mainroad, guestroom, basement, hotwaterheating, airconditioning, prefarea
mainroad = st.selectbox("🚗 อยู่ติดถนนหลักหรือไม่?", ["yes", "no"])
guestroom = st.selectbox("🏠 มีห้องรับแขกไหม?", ["yes", "no"])
basement = st.selectbox("🏡 มีห้องใต้ดินหรือไม่?", ["yes", "no"])
hotwaterheating = st.selectbox("🚿 มีระบบทำน้ำร้อนหรือไม่?", ["yes", "no"])
airconditioning = st.selectbox("❄️ มีเครื่องปรับอากาศหรือไม่?", ["yes", "no"])
prefarea = st.selectbox("🌍 อยู่ในทำเลพิเศษหรือไม่?", ["yes", "no"])

# ฟีเจอร์การตกแต่ง
furnishingstatus = st.selectbox("📦 สถานะเฟอร์นิเจอร์", ["furnished", "semi-furnished", "unfurnished"])

# การแปลงข้อมูลประเภทข้อความเป็นตัวเลข
input_data = np.array([[area, bedrooms, bathrooms, stories, parking,
                        1 if mainroad == "yes" else 0,
                        1 if guestroom == "yes" else 0,
                        1 if basement == "yes" else 0,
                        1 if hotwaterheating == "yes" else 0,
                        1 if airconditioning == "yes" else 0,
                        1 if prefarea == "yes" else 0,
                        1 if furnishingstatus == "semi-furnished" else 0,
                        1 if furnishingstatus == "unfurnished" else 0]])

# ปรับข้อมูลด้วย scaler
input_data_scaled = scaler.transform(input_data)

# ทำนายราคาบ้าน
predicted_price = model.predict(input_data_scaled)[0]

# แสดงผลลัพธ์
st.success(f"💰 ราคาที่คาดการณ์: ฿{predicted_price:,.2f}")
