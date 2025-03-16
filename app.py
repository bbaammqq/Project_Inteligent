import streamlit as st

# ตั้งค่าหน้า
st.set_page_config(
    page_title="🏡 House Price Prediction",
    page_icon="🏠",
    layout="wide"  # ทำให้เลย์เอาท์กว้างขึ้น
)

# ใช้ CSS เพื่อจัดการข้อความที่แสดงผล
st.markdown("""
    <style>
        .big-font {
            font-size: 35px !important;
            color: #4CAF50;
        }

        .header {
            text-align: center;
            color: #2C3E50;
        }

        .description {
            font-size: 20px;
            color: #34495E;
        }

        .main {
            padding-top: 10px;
            text-align: center;
        }

        /* การตั้งค่าขนาดตัวอักษรที่ปรับเปลี่ยนตามขนาดหน้าจอ */
        .responsive-text {
            font-size: 2vw;  /* ขนาดตัวอักษรจะปรับตามความกว้างของหน้าจอ */
        }

        /* ใช้ media queries เพื่อปรับขนาดตัวอักษรสำหรับหน้าจอเล็ก */
        @media only screen and (max-width: 600px) {
            .responsive-text {
                font-size: 5vw;
            }
        }

        /* ใช้ media queries เพื่อปรับขนาดตัวอักษรสำหรับหน้าจอที่ใหญ่ขึ้น */
        @media only screen and (min-width: 1200px) {
            .responsive-text {
                font-size: 1.5vw;
            }
        }

    </style>
""", unsafe_allow_html=True)

# หัวข้อหลัก
st.title("House Price Prediction in USA & Weather Forecast")
st.markdown('<p class="header responsive-text">Welcome to the Web App for Predicting House Prices and Weather Forecasts!</p>', unsafe_allow_html=True)

# เนื้อหาภายใน
st.markdown('<p class="description responsive-text">📌 This web app helps you learn and try out **Machine Learning & Neural Networks** models.</p>', unsafe_allow_html=True)
st.markdown("""
    🏡 Predict House Prices in the USA  
    📊 Compare the performance of models in **Machine Learning**  
    ⛈️ Predict weather from images in **Neural Network**  
""", unsafe_allow_html=True)

