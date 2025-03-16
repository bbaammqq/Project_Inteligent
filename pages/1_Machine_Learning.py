import streamlit as st
import pandas as pd

st.set_page_config(
    layout="wide"
)

st.title("Machine Learning")

st.write("""
## พัฒนาโมเดล Machine Learning สำหรับการพยากรณ์ราคาบ้านใน USA
[คลิกเพื่อไปยัง dataset ที่ใช้](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset)

### Feature ที่มีใน dataset
- price: ราคาของบ้าน (ตัวแปรเป้าหมาย)
- area: พื้นที่ของบ้าน (ในหน่วยตารางเมตร)
- bedrooms: จำนวนห้องนอน
- bathrooms: จำนวนห้องน้ำ
- stories: จำนวนชั้นของบ้าน
- mainroad: ติดถนนใหญ่หรือไม่ (ค่าที่เป็น yes หรือ no)
- guestroom: มีห้องรับแขกหรือไม่ (ค่าที่เป็น yes หรือ no)
- basement: มีห้องใต้ดินหรือไม่ (ค่าที่เป็น yes หรือ no)
- hotwaterheating: มีระบบทำน้ำร้อนหรือไม่ (ค่าที่เป็น yes หรือ no)
- airconditioning: มีเครื่องปรับอากาศหรือไม่ (ค่าที่เป็น yes หรือ no)
- parking: จำนวนที่จอดรถ
- prefarea: อยู่ในทำเลพิเศษหรือไม่ (ค่าที่เป็น yes หรือ no)
- furnishingstatus: สภาพการตกแต่งของบ้าน (ค่าที่เป็น furnished, semi-furnished, หรือ unfurnished)
""")

st.write("### อธิบาย Algorithm ที่ใช้พัฒนา")

st.write("### Linear Regression")
st.markdown("""
    - ใช้สำหรับทำนายค่าต่อเนื่อง (เช่น ราคาบ้าน)
    - คำนวณจากสมการเส้นตรงระหว่าง ฟีเจอร์ และ เป้าหมาย
    - ข้อดี: เข้าใจง่าย, คำนวณเร็ว, ใช้ได้ดีเมื่อข้อมูลมีความสัมพันธ์เชิงเส้น
    - ข้อเสีย: ไม่ดีสำหรับข้อมูลที่ไม่มีความสัมพันธ์เชิงเส้น
""", unsafe_allow_html=True)

st.write("### Random Forest Regressor")
st.markdown("""
    - ใช้ หลายต้นไม้การตัดสินใจ (Decision Trees) เพื่อทำนาย
    - ช่วยลดปัญหาการ Overfitting และจับความซับซ้อนของข้อมูล
    - ข้อดี: ใช้ได้กับข้อมูลที่ซับซ้อนและไม่เชิงเส้น, ทำนายได้แม่นยำ
    - ข้อเสีย: คำนวณช้าเมื่อมีข้อมูลมาก, ยากต่อการอธิบาย
""", unsafe_allow_html=True)

# โหลดข้อมูล
df = pd.read_csv("Housing.csv")

# แสดงข้อมูลในตาราง
st.write("### จัดการข้อมูล")  
st.dataframe(df)

# แสดงข้อความเกี่ยวกับการเตรียมข้อมูล
st.markdown("""
    <div style="background-color: #f0f0f0; padding: 10px; border-radius: 10px; font-family: monospace;">
        print(df.isnull().sum())<br>
        print(f"พบข้อมูลซ้ำทั้งหมด: {df.duplicated().sum()} แถว")
    </div>
    <br>
    - df.isnull().sum(): ตรวจสอบค่าที่ขาดหายไปในแต่ละคอลัมน์<br>
    - df.duplicated().sum(): ตรวจสอบข้อมูลที่ซ้ำกันใน DataFrame
    <br><br>
""", unsafe_allow_html=True)

st.write("### เนื่องจาก dataset ที่นำมาใช้มีความสมบูรณ์ จึงต้องทำให้ dataset นั้นเกิดความไม่สมบูรณ์ขึ้น ฉันจึงทำดังต่อไปนี้") 
st.markdown("""
    <div style="background-color: #f0f0f0; padding: 10px; border-radius: 10px; font-family: monospace;">
        f_incomplete = df.copy() #สร้างสำเนาของ dataframe เพื่อแก้ไข<br>
        np.random.seed(42)<br>
        missing_indices = np.random.choice(df_incomplete.index, size=50, replace=False)<br>
        df_incomplete.loc[missing_indices, ['bedrooms', 'bathrooms', 'parking']] = np.nan      
    </div>
    <br>
    - np.random.seed(42): ตั้งค่า seed เพื่อให้การสุ่มค่ามีผลลัพธ์เหมือนเดิมทุกครั้งที่รัน<br>
    - np.random.choice(df_incomplete.index, size=50, replace=False): สุ่มเลือก 50 แถวจาก DataFrame ที่จะเป็นค่าที่ขาดหายไป <br>
    - df_incomplete.loc[missing_indices, ['bedrooms', 'bathrooms', 'parking']] = np.nan: ตั้งค่า NaN (ค่าที่ขาดหายไป) สำหรับคอลัมน์ bedrooms, bathrooms, และ parking ในแถวที่ถูกสุ่ม
    <br><br> 

    <div style="background-color: #f0f0f0; padding: 10px; border-radius: 10px; font-family: monospace;">
        df_incomplete = pd.concat([df_incomplete, df_incomplete.iloc[:10]], ignore_index=True)
    </div>
    <br>
    - df_incomplete.iloc[:10]: เลือก 10 แถวแรกจาก df_incomplete<br>
    - pd.concat([...]): รวมข้อมูลเดิมและข้อมูลที่ซ้ำกันเข้าไปใน DataFrame ใหม่ ซึ่งจะเพิ่มข้อมูลซ้ำเข้ามาใน DataFrame
    <br><br>

    <div style="background-color: #f0f0f0; padding: 10px; border-radius: 10px; font-family: monospace;">
        outlier_indices = np.random.choice(df_incomplete.index, size=5, replace=False)<br>
        df_incomplete.loc[outlier_indices, 'price'] = [-9999999, 1000000000, -5000000, 999999999, -1000000]    
    </div>
    <br>
    - np.random.choice(df_incomplete.index, size=5, replace=False): สุ่มเลือก 5 แถวจาก df_incomplete<br>
    - df_incomplete.loc[outlier_indices, 'price'] = [...]: ตั้งค่าราคาบ้าน (price) ให้เป็นค่าผิดปกติ (Outliers) เช่น ราคาติดลบหรือราคามหาศาล
    <br><br>

    <div style="background-color: #f0f0f0; padding: 10px; border-radius: 10px; font-family: monospace;">
        print(df_incomplete.isnull().sum())<br>
        print(f"พบข้อมูลซ้ำทั้งหมด: {df_incomplete.duplicated().sum()} แถว")<br><br>
        print("ตรวจสอบ Outliers (ราคาบ้านติดลบหรือสูงผิดปกติ):")<br>
        print(df_incomplete[df_incomplete['price'] < 0])<br>
        print(df_incomplete[df_incomplete['price'] > df['price'].max() * 10])
    </div>
    <br>
    ตรวจสอบค่าที่หายไปอีกครั้ง เพื่อดูว่าเราสร้าง dataset ที่ไม่สมบูรณ์ได้แล้ว<br><br>
    - df_incomplete[df_incomplete['price'] < 0]: ตรวจสอบแถวที่มีราคาบ้านติดลบ <br>
    - df_incomplete[df_incomplete['price'] > df['price'].max() * 10]: ตรวจสอบแถวที่มีราคาบ้านสูงเกินกว่าค่าราคามากที่สุดใน df ถึง 10 เท่า
    <br><br>
""", unsafe_allow_html=True)


st.write("### การเตรียมข้อมูล") 
st.markdown("""
    <div style="background-color: #f0f0f0; padding: 10px; border-radius: 10px; font-family: monospace;">
        df_clean = df_incomplete.copy()<br>
        df_clean['bedrooms'].fillna(df_clean['bedrooms'].median(), inplace=True)<br>
        df_clean['bathrooms'].fillna(df_clean['bathrooms'].median(), inplace=True)<br>
        df_clean['parking'].fillna(df_clean['parking'].median(), inplace=True)<br>      
    </div>
    <br>
    - df_clean = df_incomplete.copy(): สร้างสำเนาของ DataFrame df_incomplete เพื่อทำการแก้ไขข้อมูลโดยไม่กระทบกับข้อมูลเดิมใน df_incomplete <br>
    - df_clean['bedrooms'].fillna(df_clean['bedrooms'].median(), inplace=True): ใช้ค่า มัธยฐาน (median) ของคอลัมน์ bedrooms มาเติมค่า NaN<br>
    - inplace=True: ทำการเปลี่ยนแปลงใน DataFrame df_clean โดยตรง <br>
    - การเติมค่าที่ขาดหายไปในคอลัมน์ bathrooms และ parking ทำในลักษณะเดียวกันโดยใช้ค่า มัธยฐาน (median) สำหรับทั้งสองคอลัมน์
    <br><br> 

    <div style="background-color: #f0f0f0; padding: 10px; border-radius: 10px; font-family: monospace;">
        df_clean.drop_duplicates(inplace=True)
    </div>
    <br>
    - drop_duplicates(): ใช้สำหรับลบข้อมูลที่ซ้ำกันใน DataFrame <br>
    - inplace=True: ทำการลบข้อมูลซ้ำใน DataFrame df_clean โดยตรง <br>
    - ข้อมูลที่ซ้ำกันในที่นี้จะถูกลบออก (หากแถวไหนมีค่าทั้งหมดเหมือนกับแถวอื่น ๆ)
    <br><br>

    <div style="background-color: #f0f0f0; padding: 10px; border-radius: 10px; font-family: monospace;">
        df_clean = df_clean[(df_clean['price'] > 0) & (df_clean['price'] < df['price'].max() * 10)]    
    </div>
    <br>
    - df['price'].max(): ค่าราคาบ้านสูงสุดใน DataFrame เดิม <br>
    - df_clean['price'] > 0: กำจัดค่าที่ราคาต่ำกว่าศูนย์ <br>
    - df_clean['price'] < df['price'].max() * 10: กำจัดค่าที่ราคาสูงผิดปกติ (เช่น ถ้าราคาบ้านสูงเกินกว่า 10 เท่าของราคาบ้านสูงสุดใน DataFrame)
    <br><br>

    <div style="background-color: #f0f0f0; padding: 10px; border-radius: 10px; font-family: monospace;">
        print(df_clean.isnull().sum())<br>
        print(f"พบข้อมูลซ้ำทั้งหมด: {df_clean.duplicated().sum()} แถว")
    </div>
    <br>
    - df_clean.isnull().sum(): ตรวจสอบว่ามีค่าที่ขาดหายไป (NaN) ใน DataFrame หลังจากการทำความสะอาด <br>
    - df_clean.duplicated().sum(): ตรวจสอบจำนวนข้อมูลที่ซ้ำกันใน DataFrame หลังจากการลบข้อมูลซ้ำ
    <br><br>
            
    <div style="background-color: #f0f0f0; padding: 10px; border-radius: 10px; font-family: monospace;">
        binary_cols = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]<br>
        df_clean[binary_cols] = df_clean[binary_cols].apply(lambda x: x.map({'yes': 1, 'no': 0}))
    </div>
    <br>
        แปลงค่าหมวดหมู่ที่เป็น Yes/No เป็น 0/1
    <br><br>
            
    <div style="background-color: #f0f0f0; padding: 10px; border-radius: 10px; font-family: monospace;">
        df_clean = pd.get_dummies(df_clean, columns=["furnishingstatus"], drop_first=True)
    </div>
    <br>
        แปลงค่าหมวดหมู่ที่มีมากกว่าสองค่าเป็นตัวแปร Dummy
    <br><br>
            
    <div style="background-color: #f0f0f0; padding: 10px; border-radius: 10px; font-family: monospace;">
        print(df_clean.head())
    </div>
    <br>
        ดูข้อมูลที่แปลงแล้ว
    <br><br>
""", unsafe_allow_html=True)

st.write("### การ Train ข้อมูล") 
st.markdown("""
    <div style="background-color: #f0f0f0; padding: 10px; border-radius: 10px; font-family: monospace;">
        def train_and_evaluate(model, name): <br>
                model.fit(X_train, y_train)<br>
                y_pred = model.predict(X_test)<br>
                mae = mean_absolute_error(y_test, y_pred)<br>
                mse = mean_squared_error(y_test, y_pred)<br>
                r2 = r2_score(y_test, y_pred)
    </div>
    <br>
        - สร้างฟังก์ชัน train_and_evaluate() เพื่อฝึกโมเดลและประเมินผลโมเดล คือ LinearRegression และ RandomForestRegressor<br>
        - model.predict(X_test): ใช้โมเดลที่ฝึกเสร็จแล้วเพื่อทำนายผลลัพธ์จากชุดทดสอบ (test data)<br>
        - mean_absolute_error(y_test, y_pred): คำนวณ Mean Absolute Error (MAE) ซึ่งคือค่าเฉลี่ยของค่าความผิดพลาดแบบสัมบูรณ์ (absolute error)<br>
        - mean_squared_error(y_test, y_pred): คำนวณ Mean Squared Error (MSE) ซึ่งคือค่าเฉลี่ยของค่าความผิดพลาดกำลังสอง (squared error)<br>
        - r2_score(y_test, y_pred): คำนวณ R² Score ซึ่งวัดประสิทธิภาพของโมเดล (ค่าที่ใกล้ 1 แสดงว่าโมเดลมีความแม่นยำสูง) <br>   
    <br><br>
            
    <div style="background-color: #f0f0f0; padding: 10px; border-radius: 10px; font-family: monospace;">
        print(f"ผลลัพธ์ของโมเดล {name}:")<br>
        print(f"MAE: {mae:.2f}")<br>
        print(f"MSE: {mse:.2f}")<br>
        print(f"R² Score: {r2:.2f}")<br>
        print("-" * 40)
    </div>
    <br>
        - print(): พิมพ์ผลลัพธ์ของการประเมินผล: <br>
        - name: แสดงชื่อของโมเดลที่กำลังประเมินผล <br>
    <br><br>
            
    <div style="background-color: #f0f0f0; padding: 10px; border-radius: 10px; font-family: monospace;">
        lr_model = LinearRegression()<br>
        train_and_evaluate(lr_model, "Linear Regression")
    </div>
    <br>
        การสร้างและทดสอบโมเดล Linear Regression
    <br><br>
            
    <div style="background-color: #f0f0f0; padding: 10px; border-radius: 10px; font-family: monospace;">
        rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)<br>
        train_and_evaluate(rf_model, "Random Forest")
    </div>
    <br>
        การสร้างและทดสอบโมเดล Random Forest
    <br><br>
""", unsafe_allow_html=True)
