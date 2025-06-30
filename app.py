import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import json
import pickle
import os
from PIL import Image
import joblib
import io
import base64

st.set_page_config(page_title="📈 DeepMarket", layout="wide")

# ---- Custom Sidebar Font Size ----
st.markdown("""
    <style>
        .sidebar .sidebar-content {
            font-size: 60px !important;
        }
    </style>
""", unsafe_allow_html=True)

# ---- Load data ----
@st.cache_data
def load_vis_data():
    df = pd.read_csv("cleaned_sorted_dse_data_3.csv", parse_dates=["DATE"])
    df["MONTH"] = df["DATE"].dt.month
    df["YEAR_MONTH"] = df["DATE"].dt.to_period("M").astype(str)
    return df

df_vis = load_vis_data()

# Load encoding and model
with open("company_encoding.json", "r") as f:
    enc_dict = json.load(f)

model = joblib.load("lgbm_model.pkl")

# ---- Feedback function ----
def feedback():
    st.markdown("<h2 style='text-align:center;'>:mailbox: Please Give your Feedback</h2>", unsafe_allow_html=True)

    con_form = """
    <form action="https://formsubmit.co/choowdhuryfahim03@gmail.com" method="POST">
        <input type="hidden" name="_captcha" value="false">
        <input type="text" name="name" placeholder="Your Name" required>
        <input type="text" name="email" placeholder="Your Email" required>
        <textarea name="message" placeholder="Give your Feedback"></textarea>
        <button type="submit">Send</button>
    </form>
    """
    st.markdown(con_form, unsafe_allow_html=True)

    # Optional: add custom CSS if you have a style file
    # def css(fl):
    #     with open(fl) as f:
    #         st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    # css("style/style.css")


# Sidebar
st.sidebar.title("📂 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Visualization", "📌 Prediction", "🚀 Project Journey", "✉️ Feedback"])

# ---- Page routing ----
if page == "🏠 Home":
    # Your existing Home page code here...

    st.markdown("""
        <div style="text-align: center;">
            <h1 style='color:black; font-size: 70px;'>DeepMarket</h1>
            <h3 style='color:#1b1f3a; font-size: 28px;'>Dhaka Stock Market Analysis and Price Prediction</h3>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""<div style='height:40px;'></div>
        <div style="text-align: center; max-width: 900px; margin: 0 auto; color:#241717; font-size: 18px; line-height: 1.6;">
            <h2>🌐 Description</h2>
            <p>
                Explore trends, visualize insights, and predict future movement of stocks from Dhaka Stock Exchange using interactive tools...
            </p>
        </div>
        <div style='height:60px;'></div>
    """, unsafe_allow_html=True)

    # Team members code here...

elif page == "📊 Visualization":
    # Your visualization page code...

elif page == "📌 Prediction":
    # Your prediction page code...

elif page == "🚀 Project Journey":
    # Your project journey page code...

elif page == "✉️ Feedback":
    feedback()
