# Student Dropout Risk Prediction

พยากรณ์ความเสี่ยงการลาออกกลางคันของนักเรียนด้วย AI และ Machine Learning
(Student Dropout Risk Prediction using AI & Machine Learning)

โครงงาน Mini-Project วิชา AI & ML

## ภาพรวม

ระบบนี้ใช้ Machine Learning เรียนรู้จากข้อมูลย้อนหลังเพื่อให้ **คะแนนความเสี่ยงการลาออก** ของนักเรียนแต่ละคน แล้วจัดเป็นระดับ **ต่ำ / ปานกลาง / สูง (Low / Medium / High)** พร้อม Dashboard ต้นแบบที่ช่วยให้ครูเห็นกลุ่มเสี่ยงและจัดลำดับการช่วยเหลือได้เร็วขึ้น

โครงงานประกอบด้วย 3 ส่วน

1. **การวิเคราะห์ข้อมูล (EDA)** เพื่อหาปัจจัยที่สัมพันธ์กับการลาออก
2. **โมเดล Machine Learning** เปรียบเทียบ Logistic Regression, Random Forest และ SVM (ทั้งแบบไม่ใช้และใช้ SMOTE)
3. **Dashboard (Streamlit)** แสดงรายชื่อ ระดับความเสี่ยง การแจ้งเตือน และผลรายบุคคล

## ผลลัพธ์สำคัญ

โมเดลหลักคือ **SVM + SMOTE** (เลือกจากชุด Validation แล้วประเมินบนชุด Test เพียงครั้งเดียว)

| เมตริก (กลุ่ม Dropout) | ชุด Test |
|---|---|
| F1-score | 0.834 |
| Recall | 0.859 |
| Precision | 0.810 |
| ROC-AUC | 0.938 |
| Accuracy | 0.890 |

อัตราการลาออกจริงในแต่ละระดับความเสี่ยง (ชุด Test 664 คน)

| ระดับ | เกณฑ์คะแนน | นักเรียน | อัตราลาออกจริง |
|---|---|---|---|
| Low | < 0.22 | 340 | 4.1% |
| Medium | 0.22 – 0.68 | 129 | 23.3% |
| High | ≥ 0.68 | 195 | 86.7% |

ปัจจัยที่สำคัญที่สุดคือ จำนวนวิชาที่ผ่านและเกรดในภาคเรียนที่ 1–2 และสถานะการชำระค่าเทอม

## Dashboard
ความสามารถหลัก

- การ์ดสรุปจำนวนนักเรียนในแต่ละระดับความเสี่ยง
- กล่องแจ้งเตือนเมื่อพบนักเรียนที่คะแนนเกินเกณฑ์ (ปรับได้) พร้อมดาวน์โหลดรายชื่อเป็น CSV
- รายชื่อนักเรียนเรียงตามความเสี่ยง กรองระดับและค้นหารหัสนักเรียนได้
- หน้ารายบุคคล เปรียบเทียบตัวชี้วัดกับค่ากลางของทั้งชุด
- ทดลองกรอกข้อมูลนักเรียนแล้วดูคะแนนความเสี่ยงที่โมเดลพยากรณ์

> **หมายเหตุ:** Dashboard เป็นต้นแบบ แสดงข้อมูลชุด Test จากไฟล์ การแจ้งเตือนเป็นการจำลอง (ยังไม่ส่งข้อความจริง) และรหัสนักเรียน `ST-xxxx` สร้างจากลำดับแถวเพื่อสาธิต

## Dataset

- **ชื่อ:** Predict Students' Dropout and Academic Success
- **แหล่งที่มา:** [UCI Machine Learning Repository (ID 697)](https://archive.ics.uci.edu/dataset/697)
- **ขนาด:** 4,424 แถว 36 ฟีเจอร์ (Target: Dropout / Enrolled / Graduate)
- **สัญญาอนุญาต:** CC BY 4.0
- **อ้างอิง:** Realinho, V., Machado, J., Baptista, L., & Martins, M. V. (2022). Predicting Student Dropout and Academic Success. *Data*, 7(11), 146. https://doi.org/10.3390/data7110146

ข้อมูลเป็นนักศึกษาระดับอุดมศึกษาของ Instituto Politécnico de Portalegre (ประเทศโปรตุเกส) โครงงานนี้ใช้คำว่า "นักเรียน" ตามข้อเสนอโครงงาน และแปลง Target เป็น 2 คลาส (Dropout = 1, อื่น ๆ = 0)

## ขั้นตอนการทำงาน (Workflow)

1. โหลดข้อมูลจาก UCI และสำรวจข้อมูล (EDA)
2. สร้าง Label 2 คลาส และแบ่งข้อมูล Train / Validation / Test = 70 / 15 / 15 (Stratified)
3. เตรียมข้อมูล: One-Hot Encoding, StandardScaler (fit บน Train เท่านั้น) และ SMOTE (เฉพาะ Train)
4. ฝึกและปรับจูน 3 โมเดล เลือกจาก F1 ของกลุ่ม Dropout บน Validation
5. ประเมินบนชุด Test ครั้งเดียว (Confusion Matrix, ROC, Feature Importance)
6. ฝึก SVM ซ้ำด้วย `probability=True` เพื่อให้ได้คะแนนความเสี่ยง และกำหนดเกณฑ์ Low / Medium / High จาก Validation
7. พัฒนา Dashboard ด้วย Streamlit

## โครงสร้างโปรเจกต์

```
student-dropout-prediction/
├── app/app.py                         # Dashboard (Streamlit)
├── models/
│   ├── final_model_svm_prob.joblib    # โมเดล SVM (probability=True)
│   ├── preprocessor.joblib            # ColumnTransformer (Scaler + One-Hot)
│   └── config.json                    # เกณฑ์ระดับความเสี่ยง และรายการฟีเจอร์
├── data/
│   ├── test_raw.csv                   # ข้อมูลดิบชุด Test (664 คน)
│   └── feature_importance.csv         # ความสำคัญของฟีเจอร์ (Random Forest)
├── results/                           # ตารางผลการทดลอง (.csv)
├── notebooks/01_eda_and_modeling.ipynb
├── docs/images/                       # ภาพหน้าจอ Dashboard
├── requirements.txt
└── README.md
```

## วิธีติดตั้งและรัน Dashboard

ต้องใช้ Python 3.10 ขึ้นไป และต้องติดตั้ง **scikit-learn เวอร์ชันเดียวกับที่ใช้เทรนโมเดล** (ระบุไว้ใน `requirements.txt`) เพราะไฟล์ `.joblib` ผูกกับเวอร์ชันของไลบรารี

```bash
https://github.com/TalayPsu/student-dropout-prediction.git
cd student-dropout-prediction

python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # Mac / Linux

pip install -r requirements.txt
streamlit run app/app.py
```

เบราว์เซอร์จะเปิดที่ `http://localhost:8501` เมื่อรันสำเร็จ การ์ดด้านบนควรแสดงนักเรียน 664 คน แยกเป็น High 195 / Medium 129 / Low 340

หากพบข้อความ "ไม่พบไฟล์ที่จำเป็น" ให้ตรวจว่าไฟล์อยู่ในโฟลเดอร์ `models/` และ `data/` ตามโครงสร้างด้านบน และรันคำสั่งจากโฟลเดอร์หลักของโปรเจกต์

## วิธีทำซ้ำผลการทดลอง

1. เปิด `notebooks/01_eda_and_modeling.ipynb` ใน Google Colab
2. เชื่อมต่อ Google Drive แล้วรันเซลล์ตามลำดับ (Runtime → Run all)
3. Notebook จะบันทึกโมเดลและไฟล์ผลลัพธ์ลงโฟลเดอร์ `dropout-project` ใน Google Drive ซึ่งนำมาวางใน `models/`, `data/` และ `results/` ได้

## ข้อจำกัด

- ข้อมูลเป็นนักศึกษาในโปรตุเกส และไม่มีฟีเจอร์การเข้าเรียน การส่งงาน และวินัย ตามที่เสนอไว้เดิม
- โมเดลพึ่งข้อมูลภาคเรียนที่ 2 เป็นสำคัญ จึงยังไม่ใช่การเตือนล่วงหน้าที่เร็วที่สุด
- คะแนนความเสี่ยงใช้จัดอันดับ ไม่ใช่ความน่าจะเป็นจริง (โมเดลฝึกบนข้อมูลที่ปรับสมดุลด้วย SMOTE)
- เกณฑ์ระดับความเสี่ยงเป็นค่าที่ตั้งเพื่อทดลอง และประเมินด้วยการแบ่งข้อมูลครั้งเดียว
- ผลลัพธ์ควรใช้ประกอบดุลยพินิจของครู ไม่ควรใช้ตัดสินนักเรียนโดยอัตโนมัติ

## ผู้จัดทำ

สิปปกร กองบก (6710110440) Section 1
