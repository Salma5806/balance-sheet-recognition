import streamlit as st
import pandas as pd
import os
import re
import logging
from PIL import Image
import matplotlib.pyplot as plt
from ratio import clean_number, calculate_ratios, select_key_for_year
from predict import predict_labels

# Directory containing images
OUTPUT_DIR = "output_images"

def is_valid_image(file_path):
    try:
        Image.open(file_path).verify()
        return True
    except Exception:
        return False

def load_image(image_file):
    return os.path.join(OUTPUT_DIR, image_file)

def process_all_images():
    all_dfs = []

    try:
        image_files = [
            f for f in os.listdir(OUTPUT_DIR)
            if f.lower().endswith((".png", ".jpg", ".jpeg")) and is_valid_image(os.path.join(OUTPUT_DIR, f))
        ]

        if not image_files:
            st.warning(f"No valid images found in {OUTPUT_DIR}")
            return None

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, image_file in enumerate(image_files):
            try:
                status_text.text(f"Processing {image_file}... ({i+1}/{len(image_files)})")
                progress_bar.progress((i + 1) / len(image_files))

                image_path = load_image(image_file)
                result = predict_labels(image_path)

                if result and "df" in result:
                    df = result["df"]
                    df["source_image"] = image_file

                    numeric_columns = [col for col in df.columns if col not in ['key', 'source_image']]
                    for col in numeric_columns:
                        df[col] = df[col].apply(clean_number)

                    all_dfs.append(df)

            except Exception as e:
                logging.error(f"Error with {image_file}: {str(e)}")
                st.error(f"Erreur avec {image_file}: {str(e)}")
                continue

        if all_dfs:
            return pd.concat(all_dfs, ignore_index=True)
        return None

    except FileNotFoundError as e:
        logging.error(f"Directory {OUTPUT_DIR} not found: {str(e)}")
        st.error(f"Directory {OUTPUT_DIR} not found: {str(e)}")
        return None

def app_financial():
    st.title("📊 Financial Dashboard")
    st.sidebar.header("Settings")

    if st.sidebar.button("🔄 Process All Images"):
        if "df" not in st.session_state:
            with st.spinner("Extracting data..."):
                final_df = process_all_images()
                if final_df is not None:
                    st.session_state.df = final_df
                    st.success(f"{len(image_files)} images processed successfully!")

    if st.sidebar.button("🗑 Réinitialiser"):
        st.session_state.clear()
        st.success("Session reset. Please reload the data.")

    if "df" in st.session_state:
        df = st.session_state["df"]
        key_column = st.sidebar.selectbox(
            "Select key column",
            options=list(df.columns),
            index=list(df.columns).index('key') if 'key' in df.columns else 0
        )

        try:
            ratios = calculate_ratios(df, key_column=key_column)
            if ratios:
                st.session_state.ratios = ratios
        except (ValueError, KeyError) as e:
            st.warning(f"Error calculating ratios: {str(e)}")

        year_columns = [
            col for col in df.columns
            if re.match(r'^\d{4}$|^FY\d{4}$', str(col).strip())
            and col not in [key_column, 'source_image']
        ]

        if year_columns:
            selected_year = st.sidebar.selectbox(
                "Select year",
                options=sorted(year_columns, reverse=True),
                index=0
            )

            if "ratios" in st.session_state:
                display_metrics(st.session_state.ratios, selected_year)

            display_capital_pie_chart(df, selected_year, key_column)

            st.subheader("Raw Data")
            st.dataframe(df)
        else:
            st.warning("No year columns found in the data")

if __name__ == "__main__":
    app_financial()
