import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import json
import pickle
import os
from PIL import Image
import joblib

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

# Sidebar
st.sidebar.title("📂 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Visualization", "📌 Prediction", "🚀 Project Journey"])

# ---- Home Page ----
if page == "🏠 Home":
    st.markdown("""
        <div style="text-align: center;">
            <h1 style='color:black; font-size: 70px;'>DeepMarket</h1>
            <h3 style='color:#1b1f3a; font-size: 28px;'>Dhaka Stock Market Analysis and Price Prediction</h3>
            <p style='font-size:20px; color:#241717;'>
                Explore trends, visualize insights, and predict future movement of stocks from Dhaka Stock Exchange using interactive tools.
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Add vertical spacing after the intro block
    st.markdown("<div style='height:50px'></div>", unsafe_allow_html=True)



    # Circle for Team QuantumTalk right after heading
    circle_style = """
        width: 160px;
        height: 160px;
        background: radial-gradient(circle at center, #12333A 0%, #0a2127 70%);
        border-radius: 50%;
        color: #E7D2CC;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 22px;
        margin: 0 auto 40px auto;
        box-shadow: 0 2px 5px rgba(0,0,0,0.4);
        text-align: center;
        padding: 10px;
        line-height: 1.2;
    """
    st.markdown(
        f"<div style='max-width: 160px; margin: 0 auto;'>"
        f"<div style='{circle_style}'>Team QuantumTalk</div>"
        f"</div>",
        unsafe_allow_html=True
    )

    # Team members list
    team_members = [
        {"name": "Atkia Mona Rahi", "email": "atkiamona.rahi2003@gmail.com"},
        {"name": "Abu Zafor Mohammad Saleh", "email": "abuzaforsaleh11@gmail.com"},
        {"name": "Chowdhury Manjurul Hasan", "email": "cmhfahim@gmail.com"},
        {"name": "Pijush Das", "email": "pijushdas123@gmail.com"},
        {"name": "Shafayat Hossain Ornob", "email": "ornobhossain121@gmail.com"},
    ]

    # Smaller circles for members
    member_circle_style = circle_style.replace("160px", "120px").replace("22px", "18px").replace("margin: 0 auto 40px auto;", "margin: 0 auto 10px auto;")

    email_style = """
        color: #241717;
        font-size: 14px;
        text-align: center;
        margin-bottom: 30px;
        padding: 6px 12px;
        background-color: #e7d2cc;
        border-radius: 8px;
        display: inline-block;
        max-width: 100%;
        word-wrap: break-word;
    """

    col1, col2 = st.columns(2)

    for i, member in enumerate(team_members[:4]):
        with (col1 if i % 2 == 0 else col2):
            st.markdown(
                f"""
                <div style="text-align:center; margin-bottom: 40px;">
                    <div style="{member_circle_style}">{member['name']}</div>
                    <div style="{email_style}">📧 {member['email']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # Fifth member centered below columns
    st.markdown(
        f"""
        <div style="max-width: 120px; margin: 0 auto 40px auto; text-align:center;">
            <div style="{member_circle_style}">{team_members[4]['name']}</div>
            <div style="{email_style}">📧 {team_members[4]['email']}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Bottom text (optional)
    st.markdown(
        "<p style='text-align:center; margin-top:50px; color:black;'>💡 Built by <strong>Team QuantumTalk</strong></p>",
        unsafe_allow_html=True
    )

# ---- Visualization Page ----
elif page == "📊 Visualization":
    st.markdown("## 📊 Data Visualization")

    selected_company = st.selectbox("Select a company", sorted(df_vis["TRADING CODE"].unique()))
    company_df = df_vis[df_vis["TRADING CODE"] == selected_company]

    st.subheader("📄 Raw Data")
    st.dataframe(company_df, use_container_width=True)

    st.markdown("---")

    st.subheader("📈 Close Price Over Time")
    fig1 = px.area(company_df, x="DATE", y="CLOSEP*", title=f"{selected_company} – Close Price Trend", color_discrete_sequence=["#4B8BBE"])
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("📦 Volume by Date")
    fig2 = px.bar(company_df, x="DATE", y="VOLUME", title=f"{selected_company} – Trading Volume", color_discrete_sequence=["#ff7f0e"])
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("🥧 Target Distribution")
    pie_data = company_df["TARGET"].value_counts().reindex([1, 0, -1], fill_value=0)
    pie_labels = ["1 = Up", "0 = No Change", "-1 = Down"]
    fig3 = px.pie(values=pie_data.values, names=pie_labels, color_discrete_sequence=["#2ecc71", "#f1c40f", "#e74c3c"])
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("📅 Monthly Target Histogram")
    fig4 = px.histogram(company_df, x="MONTH", color="TARGET",
                        category_orders={"MONTH": list(range(1, 13))},
                        color_discrete_map={1: "#2ecc71", 0: "#f1c40f", -1: "#e74c3c"},
                        title="Target by Month")
    st.plotly_chart(fig4, use_container_width=True)

# ---- Prediction Page ----
elif page == "📌 Prediction":
    st.markdown("## 🔮 Prediction")

    st.markdown("Enter the feature values below:")

    company_name = st.selectbox("Select company", sorted(enc_dict.keys()))
    company_id = enc_dict[company_name]

    month = st.selectbox("Month", list(range(1, 13)))
    openp = st.number_input("OPENP*", min_value=0.0, value=100.0)
    high = st.number_input("HIGH", min_value=0.0, value=105.0)
    low = st.number_input("LOW", min_value=0.0, value=95.0)
    closep = st.number_input("CLOSEP*", min_value=0.0, value=102.0)
    trade = st.number_input("TRADE", min_value=0, value=500)
    volume = st.number_input("VOLUME", min_value=0, value=10000)

    if st.button("📊 Predict"):
        input_df = pd.DataFrame([{
            "COMPANY_ID": company_id,
            "MONTH": month,
            "OPENP*": openp,
            "HIGH": high,
            "LOW": low,
            "CLOSEP*": closep,
            "TRADE": trade,
            "VOLUME": volume
        }])

        prediction = model.predict(input_df)[0]
        label_map = {1: "📈 Price Up", 0: "➖ No Change", -1: "📉 Price Down"}

        st.metric("Prediction", label_map[prediction])
        st.success(f"📊 Model predicts: **{label_map[prediction]}** for {company_name}")

        st.markdown("""
        ---
        ⚠️ **Disclaimer**:  
        This prediction is for **research purposes only**.  
        Investment decisions should be made independently.  
        The development team is **not responsible** for any outcomes.
        """)

# ---- Project Journey Page ----
elif page == "🚀 Project Journey":
    import io
    import base64

    st.markdown("## 🛤️ Project Journey")

    def image_to_base64(img):
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    image_dir = r"project_pic"
    valid_exts = (".jpg", ".jpeg", ".png")

    image_files = sorted(
        [f for f in os.listdir(image_dir) if f.lower().endswith(valid_exts)],
        key=lambda x: int(os.path.splitext(x)[0])
    )

    if not image_files:
        st.warning("⚠️ No JPG/PNG images found in the folder.")
    else:
        if "img_index" not in st.session_state:
            st.session_state.img_index = 0
        if "full_size" not in st.session_state:
            st.session_state.full_size = False

        cols = st.columns(5)
        with cols[1]:
            if st.button("⬅️ Previous"):
                st.session_state.img_index = max(0, st.session_state.img_index - 1)
        with cols[2]:
            toggle_label = "Exit Full Size" if st.session_state.full_size else "Full Size"
            if st.button(toggle_label):
                st.session_state.full_size = not st.session_state.full_size
        with cols[3]:
            if st.button("Next ➡️"):
                st.session_state.img_index = min(len(image_files) - 1, st.session_state.img_index + 1)

        img_path = os.path.join(image_dir, image_files[st.session_state.img_index])
        img = Image.open(img_path)

        max_size = (1200, 900) if st.session_state.full_size else (800, 600)
        img.thumbnail(max_size)

        st.markdown(
            f'<div style="display:flex; justify-content:center;">'
            f'<img src="data:image/png;base64,{image_to_base64(img)}" style="max-width:100%; height:auto;">'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown(f"<p style='text-align:center; margin-top:10px;'>Step {st.session_state.img_index + 1} of {len(image_files)}</p>", unsafe_allow_html=True)
