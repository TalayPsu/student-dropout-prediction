"""
Student Dropout Risk Dashboard (Streamlit prototype)

Run from the repo root:   streamlit run app/app.py

Files it needs (it searches app/, models/, data/ and the repo root):
  models/final_model_svm_prob.joblib   SVM trained with probability=True
  models/preprocessor.joblib           ColumnTransformer fitted on the train set
  models/config.json                   thresholds + feature lists
  data/test_raw.csv                    raw (un-encoded) test features + Dropout
  data/feature_importance.csv          optional, shown on the overview tab
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Dropout Risk Dashboard", page_icon="🎓", layout="wide")

LEVELS = ["Low", "Medium", "High"]
COLORS = {"Low": "#4CAF82", "Medium": "#F0B44C", "High": "#E3522B"}
ICON = {"Low": "🟢 Low", "Medium": "🟡 Medium", "High": "🔴 High"}

KEY_FEATURES = {
    "Curricular units 1st sem (grade)": "เกรดภาคเรียนที่ 1",
    "Curricular units 1st sem (approved)": "วิชาที่ผ่านภาคเรียนที่ 1",
    "Curricular units 2nd sem (grade)": "เกรดภาคเรียนที่ 2",
    "Curricular units 2nd sem (approved)": "วิชาที่ผ่านภาคเรียนที่ 2",
    "Tuition fees up to date": "ค่าเทอมเป็นปัจจุบัน (1=ใช่)",
    "Scholarship holder": "ได้รับทุน (1=ใช่)",
    "Debtor": "มีหนี้ (1=ใช่)",
    "Age at enrollment": "อายุตอนเข้าเรียน",
}

REQUIRED = ["final_model_svm_prob.joblib", "preprocessor.joblib", "config.json", "test_raw.csv"]

try:
    HERE = Path(__file__).resolve().parent
except NameError:
    HERE = Path.cwd()

SEARCH_DIRS = [
    HERE, HERE.parent, HERE / "models", HERE / "data",
    HERE.parent / "models", HERE.parent / "data", HERE.parent / "data" / "processed",
    Path.cwd(), Path.cwd() / "models", Path.cwd() / "data",
    Path("/content/drive/MyDrive/dropout-project"),
]


def find_file(name):
    for d in SEARCH_DIRS:
        p = d / name
        if p.exists():
            return p
    return None


def to_band(scores, t_low, t_high):
    """Same rule as the notebook: Low < t_low <= Medium < t_high <= High."""
    return pd.cut(np.atleast_1d(scores), bins=[-0.001, t_low, t_high, 1.001],
                  labels=LEVELS, right=False)


@st.cache_resource(show_spinner="กำลังโหลดโมเดลและข้อมูล...")
def load_bundle():
    paths = {n: find_file(n) for n in REQUIRED}
    missing = [n for n, p in paths.items() if p is None]
    if missing:
        return {"missing": missing}

    model = joblib.load(paths["final_model_svm_prob.joblib"])
    prep = joblib.load(paths["preprocessor.joblib"])
    cfg = json.loads(paths["config.json"].read_text(encoding="utf-8"))
    raw = pd.read_csv(paths["test_raw.csv"], index_col=0)
    scores = model.predict_proba(prep.transform(raw[cfg["feature_columns"]]))[:, 1]

    imp = None
    imp_path = find_file("feature_importance.csv")
    if imp_path is not None:
        imp = pd.read_csv(imp_path, index_col=0).iloc[:, 0]

    return {"model": model, "prep": prep, "cfg": cfg, "raw": raw, "scores": scores, "imp": imp}


bundle = load_bundle()
if "missing" in bundle:
    st.error("ไม่พบไฟล์ที่จำเป็น: " + ", ".join(bundle["missing"]))
    st.info("วางไฟล์โมเดล, preprocessor และ config.json ไว้ในโฟลเดอร์ `models/` "
            "และวาง `test_raw.csv` ไว้ในโฟลเดอร์ `data/` แล้วรันใหม่")
    st.stop()

model, prep, cfg = bundle["model"], bundle["prep"], bundle["cfg"]
raw, scores, imp = bundle["raw"], bundle["scores"], bundle["imp"]
t_low, t_high = float(cfg["t_low"]), float(cfg["t_high"])
feats = cfg["feature_columns"]

df = raw.copy()
df.insert(0, "Student ID", ["ST-" + str(i).zfill(4) for i in df.index])
df["Risk score"] = scores
df["Risk level"] = to_band(scores, t_low, t_high).astype(str)
key_cols = [c for c in KEY_FEATURES if c in df.columns]

# ---------- Sidebar ----------
st.sidebar.header("ตัวกรอง")
levels = st.sidebar.multiselect("ระดับความเสี่ยง", LEVELS, default=LEVELS)
query = st.sidebar.text_input("ค้นหารหัสนักเรียน", placeholder="เช่น ST-2781")
alert_th = st.sidebar.slider("เกณฑ์แจ้งเตือน (คะแนนความเสี่ยง ≥)", 0.0, 1.0, t_high, 0.01)
show_actual = st.sidebar.checkbox("แสดงผลจริงของข้อมูลทดสอบ", value=False,
                                  disabled="Dropout" not in df.columns)
st.sidebar.caption(f"เกณฑ์ระดับความเสี่ยง: Low < {t_low:.2f} ≤ Medium < {t_high:.2f} ≤ High")

# ---------- Header and summary ----------
st.title("🎓 ระบบพยากรณ์ความเสี่ยงการลาออกกลางคันของนักเรียน")
st.caption("ต้นแบบ Dashboard · โมเดล SVM + SMOTE · ข้อมูลทดสอบจาก UCI "
           "Predict Students' Dropout and Academic Success")

n = len(df)
counts = df["Risk level"].value_counts().reindex(LEVELS).fillna(0).astype(int)
k1, k2, k3, k4 = st.columns(4)
k1.metric("นักเรียนทั้งหมด", f"{n:,}")
k2.metric("🔴 ความเสี่ยงสูง (High)", f"{counts['High']:,}", f"{counts['High'] / n:.1%}", delta_color="off")
k3.metric("🟡 ปานกลาง (Medium)", f"{counts['Medium']:,}", f"{counts['Medium'] / n:.1%}", delta_color="off")
k4.metric("🟢 ต่ำ (Low)", f"{counts['Low']:,}", f"{counts['Low'] / n:.1%}", delta_color="off")


def display_table(frame, with_actual=False):
    cols = ["Student ID"] + key_cols + ["Risk score", "Risk level"]
    out = frame[cols].copy()
    out["Risk score"] = out["Risk score"].round(3)
    out["Risk level"] = out["Risk level"].map(ICON)
    out = out.rename(columns={**KEY_FEATURES, "Student ID": "รหัสนักเรียน",
                              "Risk score": "คะแนนความเสี่ยง", "Risk level": "ระดับความเสี่ยง"})
    if with_actual and "Dropout" in frame.columns:
        out["ผลจริง"] = frame["Dropout"].map({1: "ลาออก", 0: "ไม่ลาออก"}).to_numpy()
    return out


# ---------- Alert ----------
alert_df = df[df["Risk score"] >= alert_th].sort_values("Risk score", ascending=False)
if len(alert_df):
    st.error(f"🚨 แจ้งเตือน: พบนักเรียน {len(alert_df):,} คน ที่มีคะแนนความเสี่ยง ≥ {alert_th:.2f}")
    a1, a2, _ = st.columns([1, 1, 2])
    a1.download_button("ดาวน์โหลดรายชื่อ (CSV)",
                       data=display_table(alert_df).to_csv(index=False).encode("utf-8-sig"),
                       file_name="alert_list.csv", mime="text/csv")
    if a2.button("ส่งแจ้งเตือนครูประจำชั้น (จำลอง)"):
        st.success(f"บันทึกการแจ้งเตือน {len(alert_df):,} รายการแล้ว "
                   "(ต้นแบบนี้ยังไม่ได้ส่งข้อความจริง)")
else:
    st.success(f"ไม่พบนักเรียนที่มีคะแนนความเสี่ยง ≥ {alert_th:.2f}")

tab1, tab2, tab3, tab4 = st.tabs(["ภาพรวม", "รายชื่อนักเรียน", "รายบุคคล", "ทดลองกรอกข้อมูล"])

# ---------- Tab 1: overview ----------
with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("จำนวนนักเรียนตามระดับความเสี่ยง")
        fig, ax = plt.subplots(figsize=(5, 3.6))
        bars = ax.bar(LEVELS, [counts[lv] for lv in LEVELS], color=[COLORS[lv] for lv in LEVELS])
        ax.bar_label(bars)
        ax.set_ylabel("Number of students")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)
        plt.close(fig)
    with c2:
        st.subheader("ปัจจัยที่มีอิทธิพลต่อการพยากรณ์")
        if imp is not None:
            top = imp.sort_values(ascending=False).head(10).sort_values()
            fig, ax = plt.subplots(figsize=(5, 3.6))
            ax.barh(top.index, top.values, color="steelblue")
            ax.set_xlabel("Relative importance (Random Forest)")
            ax.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("ยังไม่พบไฟล์ feature_importance.csv (วางไว้ในโฟลเดอร์ data/ เพื่อแสดงกราฟนี้)")

    st.subheader("การกระจายของคะแนนความเสี่ยง")
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.hist(df["Risk score"], bins=30, color="steelblue", edgecolor="white")
    ax.axvline(t_low, color="black", ls="--")
    ax.axvline(t_high, color="black", ls="--")
    ax.set_xlabel("Risk score")
    ax.set_ylabel("Students")
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig)
    plt.close(fig)

# ---------- Tab 2: student list ----------
with tab2:
    view = df[df["Risk level"].isin(levels)]
    if query.strip():
        view = view[view["Student ID"].str.contains(query.strip(), case=False, regex=False)]
    view = view.sort_values("Risk score", ascending=False)
    st.subheader(f"รายชื่อนักเรียน ({len(view):,} คน)")
    if view.empty:
        st.info("ไม่พบนักเรียนตามตัวกรองที่เลือก")
    else:
        st.dataframe(
            display_table(view, show_actual), hide_index=True,
            column_config={"คะแนนความเสี่ยง": st.column_config.ProgressColumn(
                "คะแนนความเสี่ยง", min_value=0.0, max_value=1.0, format="%.2f")})

# ---------- Tab 3: one student ----------
with tab3:
    ordered = df.sort_values("Risk score", ascending=False)["Student ID"].tolist()
    sid = st.selectbox("เลือกรหัสนักเรียน (เรียงจากความเสี่ยงสูงไปต่ำ)", ordered)
    row = df.loc[df["Student ID"] == sid].iloc[0]
    left, right = st.columns([1, 2])
    left.metric("คะแนนความเสี่ยง", f"{row['Risk score']:.3f}")
    left.markdown(f"**ระดับความเสี่ยง:** {ICON[row['Risk level']]}")
    left.progress(float(min(max(row["Risk score"], 0.0), 1.0)))
    if show_actual and "Dropout" in df.columns:
        left.caption("ผลจริง: " + ("ลาออก" if row["Dropout"] == 1 else "ไม่ลาออก"))
    compare = pd.DataFrame({
        "ตัวชี้วัด": [KEY_FEATURES[c] for c in key_cols],
        "นักเรียนคนนี้": [round(float(row[c]), 2) for c in key_cols],
        "ค่ากลางของทั้งชุด": [round(float(df[c].median()), 2) for c in key_cols],
    })
    right.dataframe(compare, hide_index=True)

# ---------- Tab 4: what-if ----------
with tab4:
    st.subheader("ทดลองกรอกข้อมูลนักเรียน")
    st.caption("ฟีเจอร์ที่ไม่ได้ให้กรอกจะใช้ค่ากลางของข้อมูลทดสอบ (ตัวเลข) หรือค่าที่พบบ่อยที่สุด "
               "(หมวดหมู่) ใช้เพื่อสาธิตการทำงานของโมเดลเท่านั้น")

    template = {}
    for c in feats:
        if c in cfg["cat_cols"] + cfg["bin_cols"]:
            template[c] = raw[c].mode().iloc[0]
        else:
            template[c] = raw[c].median()

    inputs = {}
    num_specs = [c for c in ["Curricular units 1st sem (grade)", "Curricular units 2nd sem (grade)",
                             "Curricular units 1st sem (approved)", "Curricular units 2nd sem (approved)",
                             "Age at enrollment"] if c in feats]
    bin_specs = [c for c in ["Tuition fees up to date", "Scholarship holder", "Debtor"] if c in feats]

    form_l, form_r = st.columns(2)
    for i, c in enumerate(num_specs):
        box = form_l if i % 2 == 0 else form_r
        label = KEY_FEATURES.get(c, c)
        if "(grade)" in c:
            inputs[c] = box.slider(label, 0.0, 20.0, float(round(template[c], 1)), 0.1, key=f"wi_{c}")
        else:
            lo = 17 if c == "Age at enrollment" else 0
            hi = int(raw[c].max())
            inputs[c] = box.slider(label, lo, hi, int(np.clip(round(template[c]), lo, hi)), 1, key=f"wi_{c}")
    for i, c in enumerate(bin_specs):
        box = form_l if i % 2 == 0 else form_r
        inputs[c] = box.radio(KEY_FEATURES.get(c, c), [0, 1], index=1 if template[c] == 1 else 0,
                              format_func=lambda v: "ใช่" if v == 1 else "ไม่", horizontal=True,
                              key=f"wi_{c}")

    one = pd.DataFrame([{**template, **inputs}])[feats]
    s1 = float(model.predict_proba(prep.transform(one))[:, 1][0])
    lvl = str(to_band(s1, t_low, t_high)[0])

    st.divider()
    m1, m2 = st.columns(2)
    m1.metric("คะแนนความเสี่ยงที่พยากรณ์", f"{s1:.3f}")
    m2.markdown(f"**ระดับความเสี่ยง:** {ICON[lvl]}")
    st.progress(min(max(s1, 0.0), 1.0))

st.divider()
st.caption("หมายเหตุ: คะแนนความเสี่ยงใช้จัดอันดับ ไม่ใช่ความน่าจะเป็นจริง (โมเดลฝึกบนข้อมูลที่ปรับสมดุลด้วย SMOTE) "
           "ควรใช้ประกอบดุลยพินิจของครู ไม่ใช้ตัดสินนักเรียนโดยอัตโนมัติ · "
           "รหัสนักเรียนสร้างจากลำดับแถวของข้อมูลเพื่อสาธิต")
