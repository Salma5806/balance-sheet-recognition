import streamlit as st

st.set_page_config(page_title="Financial Dashboard", layout="centered")

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
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Validate image files
def is_valid_image(file_path):
    try:
        Image.open(file_path).verify()
        return True
    except Exception:
        return False

# List valid image files
image_files = [
    f for f in os.listdir(OUTPUT_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg")) and is_valid_image(os.path.join(OUTPUT_DIR, f))
]

def load_image(image_file):
    return os.path.join(OUTPUT_DIR, image_file)

def process_all_images():
    all_dfs = []

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

def display_metrics(ratios, selected_year):
    if not ratios or selected_year not in ratios:
        st.warning(f"No data available for year {selected_year}")
        return

    year_ratios = ratios[selected_year]
    cols = st.columns(4)

    metrics_config = {
        "Ratio liquidité générale": {"format": ".2f"},
        "Ratio de liquidité immédiate": {"format": ".2f"},
        "Marge nette": {"format": ".2%"},
        "Ratio de solvabilité": {"format": ".2f"},
    }

    for i, (metric, config) in enumerate(metrics_config.items()):
        if metric in year_ratios and year_ratios[metric] is not None:
            value = year_ratios[metric]
            cols[i].metric(label=metric, value=format(value, config["format"]))
        else:
            cols[i].warning(f"{metric} - No data available")

def display_capital_pie_chart(df, selected_year, key_column):
    st.subheader("Répartition des Capitaux Propres")

    results = select_key_for_year(df, selected_year, key_column=key_column)

    capital_data = {
        'Capital social': results.get('capital_social'),
        'Réserves': results.get('reserves'),
        'Résultat reporté': results.get('resultat_reporte'),
        'Autres capitaux propres': results.get('autres_capitaux_propres'),
        'Actions propres': results.get('actions_propres')
    }

    filtered_data = {k: abs(v) for k, v in capital_data.items() if v not in [None, 0]}

    if not filtered_data:
        st.warning("No valid data available for equity composition")
        return

    labels = list(filtered_data.keys())
    sizes = list(filtered_data.values())

    fig, ax = plt.subplots()
    try:
        ax.pie(
            sizes,
            labels=labels,
            autopct='%1.1f%%',
            startangle=90,
            colors=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0']
        )
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        fig.gca().add_artist(centre_circle)
        ax.axis('equal')
        plt.title(f"Composition des capitaux propres ({selected_year})")
        st.pyplot(fig)

    except ValueError as e:
        st.error(f"Erreur lors de la création du graphique : {str(e)}")

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

# This makes sure it runs if executed directly
if __name__ == "__main__":
    app_financial()
