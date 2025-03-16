import streamlit as st
import joblib
import pandas as pd
import numpy as np

# โหลดโมเดลที่บันทึกไว้
lr_model = joblib.load("linear_regression.pkl")
rf_model = joblib.load("random_forest.pkl")

# โหลดตัวอย่างข้อมูลเพื่อดึงชื่อฟีเจอร์ที่ถูกต้อง
sample_data = pd.read_csv("Housing.csv") 
sample_data = sample_data.drop(columns=["price"])  
feature_columns = sample_data.columns 

# ตั้งค่าหน้า
st.set_page_config(
    page_title="🏡 House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

st.markdown("""
    <style>
        .header {
            text-align: center;
            color: #2C3E50;
            font-size: 32px;
            font-weight: bold;
            margin-top: 20px;
        }
        .subheader {
            font-size: 20px;
            color: #34495E;
            margin-bottom: 10px;
        }
        .input-section {
            padding: 20px;
            background-color: #f4f7f6;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .model-selection {
            background-color: #E0F7FA;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .button-style {
            background-color: #4CAF50; 
            color: white;
            padding: 12px 24px;
            font-size: 16px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        .button-style:hover {
            background-color: #45a049;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🏡 House Price Prediction")

# ส่วนเลือกโมเดล
st.markdown('<p class="header">Select the model you want to use.    เลือกโมเดลที่ต้องการใช้:</p>', unsafe_allow_html=True)

with st.expander("Choose a model    เลือกโมเดล", expanded=True):
    model_choice = st.selectbox("Select the model you want to use.", ["Linear Regression", "Random Forest"], key="model")

# ส่วนกรอกข้อมูล
st.markdown('<p class="header">Fill in the information to predict house prices.     กรอกข้อมูลเพื่อทำนายราคาบ้าน:</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    area = st.number_input("Area (sq.m.)    พื้นที่ (ตร.ม.)", min_value=1, max_value=10000, value=200)
    bedrooms = st.number_input("Number of bedrooms.    จำนวนห้องนอน", min_value=1, max_value=10, value=3)
    bathrooms = st.number_input("Number of bathrooms.    จำนวนห้องน้ำ", min_value=1, max_value=10, value=2)
    stories = st.number_input("Number of floors.     จำนวนชั้น", min_value=1, max_value=5, value=1)
    parking = st.number_input("Parking.     ที่จอดรถ", min_value=0, max_value=5, value=1)

with col2:
    mainroad = st.selectbox("Next to the main road.      ติดถนนใหญ่?", ["yes", "no"])
    guestroom = st.selectbox("There is a living room.     มีห้องรับแขก?", ["yes", "no"])
    basement = st.selectbox("There is a basement.    มีห้องใต้ดิน?", ["yes", "no"])
    hotwaterheating = st.selectbox("There is a water heater.     มีเครื่องทำน้ำอุ่น?", ["yes", "no"])
    airconditioning = st.selectbox("There is air conditioning.      มีเครื่องปรับอากาศ?", ["yes", "no"])

# เพิ่มเติม
prefarea = st.selectbox("Located in a special location.      ตั้งอยู่ในทำเลพิเศษ?", ["yes", "no"])
furnishingstatus = st.selectbox("decoration     การตกแต่ง", ["furnished", "semi-furnished", "unfurnished"])

# ปุ่มทำนาย
if st.button("Price prediction      ทำนายราคา"):
    # **สร้าง DataFrame สำหรับอินพุต**
    input_data = pd.DataFrame([[area, bedrooms, bathrooms, stories, mainroad, guestroom, basement,
                                hotwaterheating, airconditioning, parking, prefarea, furnishingstatus]],
                              columns=["area", "bedrooms", "bathrooms", "stories", "mainroad", "guestroom",
                                       "basement", "hotwaterheating", "airconditioning", "parking",
                                       "prefarea", "furnishingstatus"])

    # **แปลงค่าหมวดหมู่เป็นตัวเลข (Yes/No → 1/0)**
    binary_mapping = {"yes": 1, "no": 0}
    for col in ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]:
        input_data[col] = input_data[col].map(binary_mapping)

    # **One-hot encoding สำหรับ `furnishingstatus`**
    input_data = pd.get_dummies(input_data, columns=["furnishingstatus"])

    # **ให้แน่ใจว่าฟีเจอร์ที่โมเดลต้องการมีครบ**
    missing_cols = set(lr_model.feature_names_in_) - set(input_data.columns)
    for col in missing_cols:
        input_data[col] = 0  # ใส่ค่า 0 ให้ฟีเจอร์ที่ขาดไป

    # **จัดเรียงคอลัมน์ให้ตรงกับตอนเทรนโมเดล**
    input_data = input_data[lr_model.feature_names_in_]

    # **พยากรณ์ด้วยโมเดลที่เลือก**
    if model_choice == "Linear Regression":
        prediction = lr_model.predict(input_data)
    else:
        prediction = rf_model.predict(input_data)

    # **แสดงผลลัพธ์**
    st.success(f"📢 ราคาบ้านที่ทำนายได้: {prediction[0]:,.2f} บาท")
