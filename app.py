# app.py
import streamlit as st
from predict import predict_labels
from copy import deepcopy
from PIL import ImageDraw, ImageFont
import os
from PIL import Image

def app_main():
    st.set_page_config(page_title="LayoutLMv3 Prediction", layout="centered")
    st.title("🔍 Predict Labels with LayoutLMv3")

    OUTPUT_DIR = "output_images"
    image_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".png")]

    if not image_files:
        st.error("❌ No images detected. Please run detection first in the main app.")
    else:
        selected_image = st.selectbox("🖼️ Select an image to predict:", image_files)
        image_path = os.path.join(OUTPUT_DIR, selected_image)
        image = Image.open(image_path).convert("RGB")
        st.image(image, caption="🖼️ Detected Image", use_container_width=True)

        if st.button("🎯 Predict Labels"):
            results = predict_labels(image_path)

            label2color = {
                'key': '#0000FF',
                'value': '#008000',
                'name': '#FFA500',
                'reference': '#FF0000',
                'year': '#FFFF00',
                'other': '#808080'
            }

            image_copy = deepcopy(results["image"])
            draw = ImageDraw.Draw(image_copy)
            font = ImageFont.load_default()

            for pred, box in zip(results["true_predictions"], results["true_boxes"]):
                color = label2color.get(pred.lower(), "gray")
                draw.rectangle(box, outline=color, width=2)
                draw.text((box[0] + 5, box[1] - 10), pred.lower(), fill=color, font=font)

            st.image(image_copy, caption="📌 Prediction Result", use_container_width=True)
            st.success("✅ Prediction complete!")

if __name__ == "__main__":
    app_main()
