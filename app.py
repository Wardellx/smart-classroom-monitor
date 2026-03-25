import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd

st.set_page_config(page_title="Smart Classroom Monitor", layout="wide")

st.title("🎓 Smart Classroom Monitoring System")
st.markdown("---")

# Store results
if "data" not in st.session_state:
    st.session_state.data = []

# Classroom selector
classroom = st.selectbox("🏫 Select Classroom", ["A", "B", "C", "D", "E"])

uploaded_file = st.file_uploader("📤 Upload Classroom Image", type=["jpg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.image(image, caption="📷 Uploaded Image", use_column_width=True)

    if st.button("🚀 Analyze Classroom"):

        model = YOLO("yolov8n.pt")
        results = model(image)

        annotated_frame = results[0].plot()
        st.image(annotated_frame, caption="🧠 Detection Output", use_column_width=True)

        # Count objects
        count_person = 0
        count_phone = 0
        count_bottle = 0
        count_book = 0
        count_chair = 0

        for r in results:
            for cls in r.boxes.cls:
                cls = int(cls)

                if cls == 0:
                    count_person += 1
                elif cls == 67:
                    count_phone += 1
                elif cls == 39:
                    count_bottle += 1
                elif cls == 73:
                    count_book += 1
                elif cls == 56:  # chair
                    count_chair += 1

        # ---------------------------
        # 🔥 CLEANLINESS (TEMP LOGIC)
        # ---------------------------
        if count_bottle + count_phone > 5:
            cleanliness = "Dirty"
        else:
            cleanliness = "Clean"

        # 👉 LATER replace with:
        # cleanliness = clean_model.predict(image)

        # ---------------------------
        # 🔥 ALIGNMENT (TEMP LOGIC)
        # ---------------------------
        if count_chair > 0 and count_person <= count_chair:
            alignment = "Aligned"
        else:
            alignment = "Not Aligned"

        # 👉 LATER replace with:
        # alignment = alignment_model.predict(image)

        # ---------------------------
        # 🔥 SMART SCORING
        # ---------------------------
        students = max(count_person, 1)

        phone_ratio = count_phone / students
        bottle_ratio = count_bottle / students

        score = 100
        score -= phone_ratio * 40
        score -= bottle_ratio * 30
        score += min(count_book, students) * 0.5

        if cleanliness == "Dirty":
            score -= 20

        if alignment == "Not Aligned":
            score -= 15

        score = max(0, min(100, int(score)))

        # Status
        if score > 80:
            status = "Excellent"
            color = "green"
        elif score > 60:
            status = "Good"
            color = "orange"
        else:
            status = "Poor"
            color = "red"

        # Save data
        st.session_state.data.append({
            "Classroom": classroom,
            "Students": count_person,
            "Phones": count_phone,
            "Bottles": count_bottle,
            "Books": count_book,
            "Cleanliness": cleanliness,
            "Alignment": alignment,
            "Score": score
        })

        # ---------------------------
        # 📊 DISPLAY
        # ---------------------------
        st.markdown("---")
        st.subheader("📊 Classroom Analysis")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("👨‍🎓 Students", count_person)
        col2.metric("📱 Phones", count_phone)
        col3.metric("🧴 Bottles", count_bottle)
        col4.metric("📚 Books", count_book)

        st.markdown("---")

        col5, col6, col7 = st.columns(3)

        col5.metric("🧹 Cleanliness", cleanliness)
        col6.metric("🪑 Alignment", alignment)
        col7.metric("⭐ Score", score)

        # Color status
        if color == "green":
            st.success(f"✅ Status: {status} Classroom")
        elif color == "orange":
            st.warning(f"⚠️ Status: {status} Classroom")
        else:
            st.error(f"❌ Status: {status} Classroom")

# ---------------------------
# 📊 DASHBOARD
# ---------------------------
st.markdown("---")
st.header("📊 Classroom Comparison Dashboard")

if len(st.session_state.data) > 0:

    df = pd.DataFrame(st.session_state.data)

    st.subheader("📋 Data Table")
    st.dataframe(df)

    st.subheader("🏆 Ranking")
    ranking = df.sort_values(by="Score", ascending=False)

    for i, row in ranking.iterrows():
        st.write(f"**Classroom {row['Classroom']} → Score: {row['Score']}**")

else:
    st.info("Upload and analyze classrooms to see dashboard.")