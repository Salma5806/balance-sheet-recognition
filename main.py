import streamlit as st
from pdf2image import convert_from_bytes
from ultralytics import YOLO
import tempfile
import cv2
import os
from PIL import Image

# Set up the Streamlit app
st.set_page_config(page_title="Table Detection App", layout="wide")
st.sidebar.title("📁 Navigation")
page = st.sidebar.radio("Go to:", ["Main", "App", "Financial Dashboard"])

# Output directory for extracted tables
OUTPUT_DIR = "output_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# YOLO table detection function
def extract_tables(image_path, output_dir=OUTPUT_DIR, page_index=0):
    model = YOLO('best.pt')  # Replace with the relative path if needed
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")
    h, w = img.shape[:2]
    results = model(img)
    extracted_paths = []

    for i, result in enumerate(results):
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        for j, (box, conf) in enumerate(zip(boxes, confidences)):
            if conf > 0.7:
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1 - 10), max(0, y1 - 10)
                x2, y2 = min(w, x2 + 10), min(h, y2 + 10)
                roi = img[y1:y2, x1:x2]
                output_path = os.path.join(output_dir, f"page_{page_index}_table_{i}_{j}.png")
                if cv2.imwrite(output_path, roi):
                    extracted_paths.append(output_path)
    return extracted_paths

# PAGE 1: MAIN
if page == "Main":
    st.title("📄 Balance Sheet Detection from PDF Report")

    uploaded_pdf = st.file_uploader("📎 Upload your annual report (PDF)", type=["pdf"])
    if uploaded_pdf:
        st.success("✅ PDF successfully uploaded!")
        try:
            images = convert_from_bytes(uploaded_pdf.read(), first_page=1, last_page=4)
            st.subheader("🖼️ Extracted Pages")
            cols = st.columns(min(len(images), 5))
            for i, img in enumerate(images):
                with cols[i % len(cols)]:
                    st.image(img, caption=f"Page {i+1}", width=160)

            if st.button("🚀 Run YOLOv11 Table Detection"):
                st.subheader("📍 Detection Results")
                for i, img in enumerate(images):
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                        img.save(tmp.name, format="PNG")
                        tmp_path = tmp.name

                    try:
                        extracted = extract_tables(tmp_path, page_index=i)
                        if extracted:
                            st.markdown(f"**📑 Page {i+1} - Extracted Tables:**")
                            cols_tables = st.columns(min(len(extracted), 4))
                            for k, table_path in enumerate(extracted):
                                with cols_tables[k % len(cols_tables)]:
                                    st.image(table_path, caption=f"Table {k+1}", width=280)
                        else:
                            st.warning(f"No table detected on page {i+1}.")
                    except Exception as e:
                        st.error(f"Error on page {i+1}: {e}")
                    finally:
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)
        except Exception as e:
            st.error(f"Error processing the PDF: {e}")

# PAGE 2: APP 


elif page == "App":
    app_main()

elif page == "Financial Dashboard":
    app_financial()
