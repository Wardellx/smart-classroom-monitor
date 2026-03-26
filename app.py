import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
import gdown
import os

# ---------------------------
st.set_page_config(page_title="Smart Classroom Monitor", layout="wide")

st.markdown("""
<h1 style='text-align:center; color:#4CAF50;'>🎓 Smart Classroom Monitoring System</h1>
<p style='text-align:center; color:gray;'>AI-powered cleanliness, behavior & classroom analytics</p>
<hr>
""", unsafe_allow_html=True)

# ---------------------------
@st.cache_resource
def load_models():
    model_main = YOLO("yolov8m.pt")

    # ✅ Download best.pt from Google Drive if not present
    if not os.path.exists("best.pt"):
        url = "https://drive.google.com/uc?id=1nPBbqYVZiI-VTM14ejHMiz-7qJlsSeit"
        gdown.download(url, "best.pt", quiet=False)

    model_trash = YOLO("best.pt")

    return model_main, model_trash

model_main, model_trash = load_models()

# ---------------------------
def remove_duplicates(boxes, dist_threshold=40):
    filtered = []
    for box in boxes:
        keep = True
        for f in filtered:
            b1 = box["coords"]
            b2 = f["coords"]

            cx1 = (b1[0] + b1[2]) / 2
            cy1 = (b1[1] + b1[3]) / 2
            cx2 = (b2[0] + b2[2]) / 2
            cy2 = (b2[1] + b2[3]) / 2

            dist = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)

            if dist < dist_threshold:
                if box["conf"] > f["conf"]:
                    f.update(box)
                keep = False
                break

        if keep:
            filtered.append(box)

    return filtered

# ---------------------------
def draw_boxes(image, boxes):
    for b in boxes:
        x1, y1, x2, y2 = map(int, b["coords"])

        label = str(b["cls"])
        if label == "0":
            label = "Student"; color = (0,255,0)
        elif label == "56":
            label = "Chair"; color = (255,0,0)
        elif label == "39":
            label = "Bottle"; color = (0,0,255)
        elif label == "67":
            label = "Phone"; color = (255,255,0)
        elif label == "73":
            label = "Book"; color = (255,0,255)
        elif label == "trash":
            label = "Trash"; color = (0,255,255)
        else:
            color = (200,200,200)

        cv2.rectangle(image, (x1,y1), (x2,y2), color, 2)
        cv2.putText(image, label, (x1,y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image

# ---------------------------
def calculate_score(count_person, count_phone, count_bottle,
                    count_book, cleanliness, alignment, trash_count):

    students = max(count_person, 1)
    score = 100

    score -= min(count_phone*8, 40)
    score -= min(count_bottle*4, 20)
    score += min(count_book/students,1.0)*15
    score -= min(trash_count*5,20)

    if cleanliness == "Dirty":
        score -= 15
    if alignment == "Not Aligned":
        score -= 10

    return max(0, min(100, int(score)))

# ---------------------------
def check_alignment(chairs):
    if len(chairs) < 2:
        return "Not Enough Chairs"

    centers = [(float((c["coords"][0]+c["coords"][2])/2),
                float((c["coords"][1]+c["coords"][3])/2)) for c in chairs]

    centers.sort(key=lambda p: p[1])

    rows, current = [], [centers[0]]

    for cx, cy in centers[1:]:
        if abs(cy - current[-1][1]) < 60:
            current.append((cx, cy))
        else:
            rows.append(current)
            current = [(cx, cy)]
    rows.append(current)

    aligned_rows = sum(1 for r in rows if len(r) >= 2)

    return "Aligned" if aligned_rows/len(rows) >= 0.6 else "Not Aligned"

# ---------------------------
def process_frame(image, conf_main, conf_trash):

    results_main = model_main(image, conf=conf_main, imgsz=960,
                              classes=[0,39,56,67,73])

    all_boxes = []

    for r in results_main:
        if r.boxes is None:
            continue
        for box in r.boxes:
            all_boxes.append({
                "coords": box.xyxy[0].cpu().numpy(),
                "cls": str(int(box.cls)),
                "conf": float(box.conf)
            })

    results_trash = model_trash(image, conf=conf_trash, imgsz=960)

    for r in results_trash:
        if r.boxes is None:
            continue
        for box in r.boxes:
            all_boxes.append({
                "coords": box.xyxy[0].cpu().numpy(),
                "cls": "trash",
                "conf": float(box.conf)
            })

    all_boxes = remove_duplicates(all_boxes, 50)

    count_person = sum(1 for b in all_boxes if b["cls"]=="0")
    count_chair  = sum(1 for b in all_boxes if b["cls"]=="56")
    count_bottle = sum(1 for b in all_boxes if b["cls"]=="39")
    count_phone  = sum(1 for b in all_boxes if b["cls"]=="67")
    count_book   = sum(1 for b in all_boxes if b["cls"]=="73")
    trash_count  = sum(1 for b in all_boxes if b["cls"]=="trash")

    chairs = [b for b in all_boxes if b["cls"]=="56"]
    alignment = check_alignment(chairs)

    cleanliness = "Dirty" if trash_count > 2 else "Clean"

    score = calculate_score(
        count_person,count_phone,count_bottle,
        count_book,cleanliness,alignment,trash_count
    )

    annotated = draw_boxes(image.copy(), all_boxes)

    return annotated, count_person, count_chair, count_bottle, count_phone, count_book, trash_count, score, cleanliness, alignment

# ---------------------------
# SIDEBAR
st.sidebar.header("⚙️ Controls")

mode = st.sidebar.selectbox("Mode", ["Image Upload", "Live Webcam"])
conf_main = st.sidebar.slider("Main Confidence", 0.1, 1.0, 0.25)
conf_trash = st.sidebar.slider("Trash Confidence", 0.1, 1.0, 0.3)

if "data" not in st.session_state:
    st.session_state.data = []

# ---------------------------
# IMAGE MODE
if mode == "Image Upload":

    classroom = st.selectbox("🏫 Select Classroom", ["A","B","C","D","E"])
    uploaded_file = st.file_uploader("Upload Image", type=["jpg","png"])

    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        if st.button("🚀 Analyze Classroom"):
            annotated, *metrics = process_frame(image, conf_main, conf_trash)

            (count_person,count_chair,count_bottle,
             count_phone,count_book,trash_count,
             score,cleanliness,alignment) = metrics

            st.session_state.data.append({
                "Classroom": classroom,
                "Students": count_person,
                "Chairs": count_chair,
                "Phones": count_phone,
                "Bottles": count_bottle,
                "Books": count_book,
                "AI Trash": trash_count,
                "Cleanliness": cleanliness,
                "Alignment": alignment,
                "Score": score
            })

            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

            cols = st.columns(7)
            cols[0].metric("Students", count_person)
            cols[1].metric("Chairs", count_chair)
            cols[2].metric("Bottles", count_bottle)
            cols[3].metric("Phones", count_phone)
            cols[4].metric("Books", count_book)
            cols[5].metric("Trash", trash_count)
            cols[6].metric("Score", score)

# ---------------------------
# WEBCAM MODE
if mode == "Live Webcam":

    run = st.checkbox("Start Webcam")
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, *_ = process_frame(frame, conf_main, conf_trash)
        FRAME_WINDOW.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

    cap.release()

# ---------------------------
# DASHBOARD
st.markdown("---")
st.header("📊 Dashboard")

if st.session_state.data:
    df = pd.DataFrame(st.session_state.data)
    st.dataframe(df)

    st.subheader("🏆 Ranking")
    for _,row in df.sort_values("Score",ascending=False).iterrows():
        st.write(f"{row['Classroom']} → {row['Score']}")
else:
    st.info("No data yet")