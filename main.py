from operator import index
import streamlit as st
import plotly.express as px
from pycaret.regression import setup, compare_models, pull, save_model, load_model
import pandas_profiling
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import os
from pathlib import Path
from PIL import Image


# Set the background color for the entire web app

current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
css_file = current_dir / "styles" / "main.css"
# profile_pic = current_dir / "assets" / "3.png"
if os.path.exists('./dataset.csv'):
    df = pd.read_csv('dataset.csv', index_col=None)

with st.sidebar:
    st.image("data:image/jpeg;base64,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")
    st.title("Automating the task")
    choice = st.radio("Navigation", ["Credit","Upload", "Profiling", "Modelling","ML" ,"Download"])
    st.info("This project application helps you build and explore your data.")

if choice == "Credit":
    st.markdown(
    """
    <style>
        body {
            background-color: #88d8c0 !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

    current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
    css_file = current_dir / "styles" / "main.css"
    profile_pic_url = "https://drive.google.com/file/d/10zRHkviIl9c-Tg8WtCdu9zdYsLIC129A/view?usp=drive_link"
    NAME = "Sangratna Gaikwad"
    DESCRIPTION = """M.Tech student at IIT Kharagpur | Aspiring Data Scientist"""
    SOCIAL_MEDIA = {
        "Email": "mailto:gsangratna21@gmail.com",
        "Linkedln": "https://linkedin.com/in/sangratna-gaikwad-395376134",
        "GitHub": "https://github.com/sangratna",
    }

    with open(css_file) as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

    try:
        col1, col2 = st.columns(2)

        with col1:
            # Displaying image directly with st.image
            st.image(profile_pic_url, width=190)

        with col2:
            st.markdown("<h2 style='margin:0; font-size:1.5em;'>Sangratna Gaikwad</h2>", unsafe_allow_html=True)
            st.write(DESCRIPTION)

            ICON_MAPPING = {
                "Email": "✉️",
                "Linkedln": "🌐",
                "GitHub": "🐙"
            }
            social_media_line = " ".join(
                f"{ICON_MAPPING[platform]} [{platform}]({link})" for platform, link in SOCIAL_MEDIA.items())
            st.markdown(social_media_line)

    except FileNotFoundError as e:
        st.error(f"Error: {e}. Make sure the image file is present at the specified location.")
if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file: 
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)
        
if choice == "Profiling": 
    st.title("Exploratory Data Analysis")
    profile_df = df.profile_report()
    st_profile_report(profile_df)

    
if choice == "Modelling": 
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    if st.button('Run Modelling'): 
        setup(df, target=chosen_target)
        setup_df = pull()
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.dataframe(compare_df)
        save_model(best_model, 'best_model')

if choice == "ML":
    st.title("Model Training")
    target = st.selectbox("Select your target variable",df.columns)
    setup(df,target = target)
    setup_df = pull()
    st.info("This is ML Experiment Setting Created by SBG")
    st.dataframe(setup_df)
    best_model = compare_models()
    compare_df = pull()
    st.info("This is the best model")
    st.dataframe(compare_df)
    best_model
    
if choice == "Download": 
    with open('best_model.pkl', 'rb') as f: 
        st.download_button('Download Model', f, file_name="best_model.pkl")