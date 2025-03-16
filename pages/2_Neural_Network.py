import streamlit as st
import pandas as pd

st.set_page_config(
    layout="wide"
)

st.title("Neural Network")

st.write("""
## พัฒนาโมเดล Neural Network สำหรับการพยากรณ์อากาศจากรูปภาพ
[คลิกเพื่อไปยัง dataset ที่ใช้](https://www.kaggle.com/datasets/jehanbhathena/weather-dataset)

### Feature ที่มีใน dataset
- fogsmog: หมอกและมลพิษ
- rime: น้ำค้างแข็งแบบขาวขุ่นที่เกาะตัวแน่น
- frost: น้ำค้างแข็งแบบผลึกบาง
- snow: หิมะ
- rain: ฝน
- lightning: ฟ้าผ่า
- dew: น้ำค้าง
- hail: ลูกเห็บ
- glaze: น้ำแข็ง
- rainbow: รุ้ง
- sandstorm: พายุทราย
""")

st.write("### อธิบาย Algorithm ที่ใช้พัฒนา")

st.write("### Convolutional Neural Network (CNN)")
st.markdown("""
        CNN มีความสามารถในการจับลักษณะสำคัญจากภาพและสามารถจำแนกประเภทของภาพได้ดี โดยใช้การเรียนรู้ลักษณะของภาพผ่านการคำนวณในชั้นต่างๆ
    - Convolutional Layers: ใช้ฟิลเตอร์ (kernels) สำหรับดึงลักษณะสำคัญจากภาพ เช่น ขอบ, รูปทรง หรือพื้นผิว<br>
    - Activation Function (ReLU): ใช้ ReLU เพื่อเปิดใช้งานค่าที่ผ่านการคำนวณจาก convolution และช่วยให้โมเดลเรียนรู้ความสัมพันธ์ที่ไม่เป็นเชิงเส้น<br>
    - Pooling Layers: ใช้ Max Pooling เพื่อย่อขนาดภาพ (downsampling) โดยเลือกค่าที่ใหญ่ที่สุดจากแต่ละกลุ่มพิกเซล<br>
    - Fully Connected Layers: ทำการตัดสินใจสุดท้ายในการจำแนกประเภทของภาพหลังจากที่ข้อมูลผ่านการ convolution และ pooling <br>
""", unsafe_allow_html=True)

# โหลดข้อมูล
df = pd.read_csv("Housing.csv")

# แสดงข้อมูลในตาราง
st.write("### จัดการข้อมูล")

# แสดงข้อความเกี่ยวกับการเตรียมข้อมูล
st.markdown("""
    <div style="background-color: #f0f0f0; padding: 10px; border-radius: 10px; font-family: monospace;">
        dataset_path = os.path.join(data_path, "dataset") <br>
        print(os.listdir(dataset_path))
    </div>
    <br>
        ตรวจสอบว่าในโฟลเดอร์ dataset มีอะไรอยู่ข้างใน
    <br><br>
            
    <div style="background-color: #f0f0f0; padding: 10px; border-radius: 10px; font-family: monospace;">
        sample_folder = os.path.join(dataset_path, "fogsmog")<br>
        print(os.listdir(sample_folder)[:5])
    </div>
    <br>
        เลือกโฟลเดอร์แรกมาทดสอบ ตรวจสอบว่ามีไฟล์ในโฟลเดอร์ "fogsmog" หรือไม่ โดยการแสดงชื่อไฟล์ 5 ไฟล์แรก
    <br><br>
            
    <div style="background-color: #f0f0f0; padding: 10px; border-radius: 10px; font-family: monospace;">
        image_paths = []<br>
        labels = []<br>
        dataset_path = "weather_data/dataset" 
    </div>
    <br>
        สร้าง lists เก็บ path ของรูปภาพ และ labels จากนั้นปรับ path ให้ตรงกับโฟลเดอร์
    <br><br>
            
    <div style="background-color: #f0f0f0; padding: 10px; border-radius: 10px; font-family: monospace;">
        for category in os.listdir(dataset_path): <br>
            category_path = os.path.join(dataset_path, category)
    </div>
    <br>
        วนลูปอ่านภาพจากทุกโฟลเดอร์
    <br><br>
            
    <div style="background-color: #f0f0f0; padding: 10px; border-radius: 10px; font-family: monospace;">
        if os.path.isdir(category_path): <br>
        for img_file in glob.glob(os.path.join(category_path, "*.jpg")):  # "*.jpg" หรือ "*.png" <br>
            image_paths.append(img_file)<br>
            labels.append(category)
    </div>
    <br>
        ตรวจสอบว่าเป็นโฟลเดอร์ (ไม่ใช่ไฟล์) จากนั้นใช้ชื่อโฟลเดอร์เป็น label 
    <br><br>
            
    <div style="background-color: #f0f0f0; padding: 10px; border-radius: 10px; font-family: monospace;">
        print(f"จำนวนรูปภาพทั้งหมด: {len(image_paths)}")<br>
        print(f"ตัวอย่าง labels: {labels[:5]}")<br>
    </div>
    <br>
        ตรวจสอบว่าข้อมูลถูกโหลดมาหรือไม่ 
    <br><br>
""", unsafe_allow_html=True)

st.write("### เนื่องจาก dataset ที่นำมาใช้มีความสมบูรณ์ จึงต้องทำให้ dataset นั้นเกิดความไม่สมบูรณ์ขึ้น ฉันจึงทำดังต่อไปนี้") 
st.markdown("""
    <div style="background-color: #f0f0f0; padding: 10px; border-radius: 10px; font-family: monospace;">
        num_remove_images = 100  <br> 
        num_missing_labels = 50  <br>
        num_corrupt_images = 30  <br>    
    </div>
    <br>
        - num_remove_images: กำหนดจำนวน 100 ภาพ ที่จะลบออกจาก dataset <br>
        - num_missing_labels: กำหนดจำนวน 50 labels ที่จะทำให้หายไป (ค่า label จะกลายเป็น None) <br>
        - num_corrupt_images: กำหนดจำนวน 30 ภาพ ที่จะทำให้เสียหาย (จะถูกเขียนทับด้วยภาพดำ) <br>
    <br><br> 

    <div style="background-color: #f0f0f0; padding: 10px; border-radius: 10px; font-family: monospace;">
        remove_indices = random.sample(range(len(image_paths)), num_remove_images) <br>
        image_paths = [img for i, img in enumerate(image_paths) if i not in remove_indices] <br>
        labels = [label for i, label in enumerate(labels) if i not in remove_indices]
    </div>
    <br>
        ลบภาพบางส่วนออกจาก dataset: สุ่มเลือก 100 ดัชนีจาก image_paths ซึ่งจะเป็นภาพที่เราจะลบออกจาก dataset
    <br><br>

    <div style="background-color: #f0f0f0; padding: 10px; border-radius: 10px; font-family: monospace;">
        missing_label_indices = random.sample(range(len(labels)), num_missing_labels) <br>
        for i in missing_label_indices: <br>
        labels[i] = None  # ตั้งค่าให้ label เป็น None (Missing Data)
    </div>
    <br>
        สุ่มเลือก 50 labels ที่จะทำให้หายไป จากนั้นตั้งค่า label ของตำแหน่งนั้นเป็น None ซึ่งหมายถึงข้อมูลที่ขาดหายไป
    <br><br>

    <div style="background-color: #f0f0f0; padding: 10px; border-radius: 10px; font-family: monospace;">
        corrupt_indices = random.sample(range(len(image_paths)), num_corrupt_images) <br>
        for i in corrupt_indices:<br>
            img = np.zeros((128, 128, 3), dtype=np.uint8)  # สร้างภาพดำ <br>
            cv2.imwrite(image_paths[i], img)  # เขียนภาพใหม่ทับไฟล์เดิม <br>
    </div>
    <br>
        สุ่มเลือก 30 ภาพ ที่จะทำให้เสียหาย(เปลี่ยนเป็นภาพสีดำ)
    <br><br>
            
    <div style="background-color: #f0f0f0; padding: 10px; border-radius: 10px; font-family: monospace;">
        print(f"รูปภาพคงเหลือ: {len(image_paths)}") <br>
        missing_labels_count = sum(1 for label in labels if label is None) <br>
        print(f"Labels ที่หายไปตอนนี้: {missing_labels_count}")
    </div>
    <br>
        ตรวจสอบจำนวนภาพที่เหลืออยู่ และตรวจสอบจำนวน Labels ที่หายไป
    <br><br>
            
    <div style="background-color: #f0f0f0; padding: 10px; border-radius: 10px; font-family: monospace;">
        if len(image_paths) == len(labels): <br>
            print("ข้อมูลยังคงสอดคล้องกัน (Image = Label)") <br>
        else: <br>
            print("จำนวน Image และ Labels ไม่ตรงกัน!")
    </div>
    <br>
        ตรวจสอบว่า จำนวนภาพ (ใน image_paths) และ จำนวน labels (ใน labels) ตรงกันหรือไม่
    <br><br>
            
    <div style="background-color: #f0f0f0; padding: 10px; border-radius: 10px; font-family: monospace;">
        from collections import Counter <br>
        category_counts = Counter(labels) <br>
        print("จำนวนภาพในแต่ละหมวดหมู่:") <br>
        for category, count in category_counts.items(): <br>
            print(f"{category}: {count}")
    </div>
    <br>
        ตรวจสอบหมวดหมู่ของข้อมูลที่เหลืออยู่
    <br><br>
""", unsafe_allow_html=True)


st.write("### การเตรียมข้อมูล") 
st.markdown("""
    <div style="background-color: #f0f0f0; padding: 10px; border-radius: 10px; font-family: monospace;">
        clean_image_paths = [img for img, label in zip(image_paths, labels) if label is not None] <br>
        clean_labels = [label for label in labels if label is not None] <br>
        print(f"ข้อมูลหลังทำความสะอาด: เหลือรูปภาพ {len(clean_image_paths)} รูป")
    </div>
    <br>
        ลบข้อมูลที่ไม่มี Label ใช้ list comprehension เพื่อกรองเฉพาะ labels ที่ไม่เป็น None จะเก็บเฉพาะ labels ที่มีข้อมูล (ไม่เป็น None)
        จากนั้นพิมพ์จำนวนข้อมูลที่เหลือ
    <br><br> 

    <div style="background-color: #f0f0f0; padding: 10px; border-radius: 10px; font-family: monospace;">
        IMG_SIZE = (128, 128)  # ปรับขนาดรูปให้เท่ากัน
    </div>
    <br>
        กำหนดขนาดของภาพที่ต้องการจะ resize ให้อยู่ที่ 128x128 พิกเซล
    <br><br>

    <div style="background-color: #f0f0f0; padding: 10px; border-radius: 10px; font-family: monospace;">
        def preprocess_image(img_path): <br>
            img = load_img(img_path, target_size=IMG_SIZE) <br>
            img_array = img_to_array(img) / 255.0  <br>
            return img_array
    </div>
    <br>
        - ใช้ฟังก์ชัน load_img ในการโหลดภาพจาก img_path และ ปรับขนาด ให้เป็นขนาด 128x128 พิกเซล ตามที่กำหนดใน IMG_SIZE <br>
        - ใช้ฟังก์ชัน img_to_array ในการแปลงภาพที่โหลดมาจากรูปแบบของไฟล์เป็น อาร์เรย์ (array) ซึ่งจะสามารถนำไปใช้ในการฝึกโมเดล <br>
        - normalize ค่าของพิกเซลในภาพให้เป็นช่วงระหว่าง 0-1 (จากเดิมที่ค่าพิกเซลจะอยู่ระหว่าง 0-255)<br>
            ฟังก์ชันนี้จะ return ภาพที่ถูกแปลงเป็น อาร์เรย์ ของตัวเลขที่ มีขนาด 128x128 และ ค่าพิกเซลระหว่าง 0-1
    <br><br>

    <div style="background-color: #f0f0f0; padding: 10px; border-radius: 10px; font-family: monospace;">
        image_data = np.array([preprocess_image(img) for img in clean_image_paths])
    </div>
    <br>
        เก็บอาร์เรย์ของข้อมูลภาพทั้งหมด ที่ถูกแปลงแล้ว
    <br><br>
            
    <div style="background-color: #f0f0f0; padding: 10px; border-radius: 10px; font-family: monospace;">
        print(f"Shape ของ dataset: {image_data.shape}")
    </div>
    <br>
        ตรวจสอบขนาดของ dataset ที่ได้หลังจากแปลงภาพทั้งหมดเป็นอาร์เรย์
    <br><br>
            
    <div style="background-color: #f0f0f0; padding: 10px; border-radius: 10px; font-family: monospace;">
        label_encoder = LabelEncoder() <br>
        encoded_labels = label_encoder.fit_transform(clean_labels) 
    </div>
    <br>
        LabelEncoder object ซึ่งจะใช้ในการแปลง labels เป็นตัวเลข ผลที่ได้จะเป็น array ที่มีค่าตัวเลขแทน labels แต่ละตัว
    <br><br>
            
    <div style="background-color: #f0f0f0; padding: 10px; border-radius: 10px; font-family: monospace;">
        label_mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_)))) <br>
        print("Label Mapping:", label_mapping)
    </div>
    <br>
        - ใช้ฟังก์ชัน range() สร้าง ช่วงตัวเลข (จาก 0 ถึงจำนวนหมวดหมู่ที่มี) เพื่อนำมา Map กับหมวดหมู่ที่พบ <br>
        จากนั้นแสดงผลการจับคู่ที่เกิดขึ้นระหว่างหมวดหมู่และตัวเลข ในรูปแบบ dictionary
    <br><br>
            
    <div style="background-color: #f0f0f0; padding: 10px; border-radius: 10px; font-family: monospace;">
        X_train, X_test, y_train, y_test = train_test_split(image_data, encoded_labels, test_size=0.2, random_state=42)
    </div>
    <br>
        แบ่งข้อมูลเป็น Training Data และ Testing Data <br>
        - test_size=0.2: กำหนดว่า 20% ของข้อมูลทั้งหมดจะถูกใช้เป็นข้อมูลทดสอบ (test data) และ 80% จะใช้เป็นข้อมูลฝึก (training data)
    <br><br>
            
    <div style="background-color: #f0f0f0; padding: 10px; border-radius: 10px; font-family: monospace;">
        print(f"Training Data: {X_train.shape}") <br>
        print(f"Testing Data: {X_test.shape}")
    </div>
    <br>
        แสดงขนาดของ Training Data และ Testing Data
    <br><br>
""", unsafe_allow_html=True)

st.write("### การ Train ข้อมูล") 
st.markdown("""
    <div style="background-color: #f0f0f0; padding: 10px; border-radius: 10px; font-family: monospace;">
        num_classes = len(label_mapping) 
    </div>
    <br>
        กำหนดจำนวนประเภทของสภาพอากาศที่ต้องทำนาย (จำนวน classes) ซึ่งได้มาจาก label_mapping (จำนวนหมวดหมู่ที่มีใน dataset)
    <br><br>
            
    <div style="background-color: #f0f0f0; padding: 10px; border-radius: 10px; font-family: monospace;">
        model = Sequential([
            Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)), <br>
            MaxPooling2D(pool_size=(2,2)), <br>
            Conv2D(64, (3,3), activation='relu'), <br>
            MaxPooling2D(pool_size=(2,2)), <br>
            Flatten(), <br>
            Dense(128, activation='relu'), <br>
            Dense(num_classes, activation='softmax')  # ใช้ softmax เพราะเป็น classification <br>
        ])
    </div>
    <br>
        สร้างโมเดลในรูปแบบของ Sequential ซึ่งแต่ละเลเยอร์จะทำงานต่อเนื่องกัน <br>
        - Conv2D: เลเยอร์ Convolutional ที่ใช้กรองภาพ (ใช้ตัวกรองขนาด 3x3) เพื่อดึงลักษณะจากภาพ มี 32 filters <br>
        - activation='relu': ใช้ฟังก์ชัน ReLU (Rectified Linear Unit) สำหรับการคำนวณค่า activation <br>
        - MaxPooling2D: เลเยอร์ Max Pooling ใช้เพื่อย่อขนาดภาพ <br>
        - Conv2D(64, (3,3), activation='relu') เป็นเลเยอร์ Convolutional อีกชุดหนึ่ง มี 64 filters <br>
        - Flatten(): ใช้แปลงข้อมูลจากรูปแบบหลายมิติ (2D) ที่ได้จากเลเยอร์ก่อนหน้านี้ให้เป็นข้อมูลแบบ 1D เพื่อป้อนเข้าไปใน Dense layer <br>
        - Dense(128, activation='relu'): เลเยอร์ Fully connected ที่มี 128 neurons และใช้ ReLU เป็นฟังก์ชัน activation <br>
        - Dense(num_classes, activation='softmax'): เลเยอร์ Fully connected ที่มี num_classes neurons (ตามจำนวนประเภทของสภาพอากาศ)
    <br><br>
            
    <div style="background-color: #f0f0f0; padding: 10px; border-radius: 10px; font-family: monospace;">
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    </div>
    <br>
        - optimizer='adam': ใช้ Adam optimizer ปรับค่าพารามิเตอร์ของโมเดลให้เหมาะสม
        - loss='sparse_categorical_crossentropy': ใช้ Sparse Categorical Crossentropy เป็นฟังก์ชันสูญเสีย (loss function) สำหรับการจำแนกประเภทที่มีหลายประเภท <br>
        - metrics=['accuracy']: ใช้ accuracy เป็นเมตริกในการวัดผลการทำนาย (บอกเปอร์เซ็นต์ของการทำนายที่ถูกต้อง)
    <br><br>
            
    <div style="background-color: #f0f0f0; padding: 10px; border-radius: 10px; font-family: monospace;">
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)
    </div>
    <br>
        - ฝึกโมเดลเป็นจำนวน 10 รอบ (epochs) บนข้อมูลฝึก <br>
        - ใช้ 32 ตัวอย่างในแต่ละ batch ในระหว่างการฝึก
    <br><br>
            
    <div style="background-color: #f0f0f0; padding: 10px; border-radius: 10px; font-family: monospace;">
        test_loss, test_acc = model.evaluate(X_test, y_test) <br>
        print(f"ความแม่นยำบน Test Set: {test_acc * 100:.2f}%")
    </div>
    <br>
        ทดสอบความแม่นยำบน Test Set จากนั้นแสดงความแม่นยำ
    <br><br>
            
    <div style="background-color: #f0f0f0; padding: 10px; border-radius: 10px; font-family: monospace;">
        train_acc = history.history['accuracy'][-1] <br>
        val_acc = history.history['val_accuracy'][-1] <br>
        print(f"ความแม่นยำบน Training Set: {train_acc * 100:.2f}%") <br>
        print(f"ความแม่นยำบน Validation Set: {val_acc * 100:.2f}%")
    </div>
    <br>
        ดูค่าความแม่นยำของโมเดลใน Training Set และ Validation Set จากนั้นแสดงความแม่นยำ
    <br><br>
""", unsafe_allow_html=True)
