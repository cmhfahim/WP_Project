import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import json
import pickle
import os
from PIL import Image
import joblib


st.set_page_config(page_title="📈 Dhaka Stock Market", layout="wide")

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
            <h1 style='color:black;'>📈 Dhaka Stock Market Analysis</h1>
            <p style='font-size:18px; color:black;'>
                Explore trends, visualize insights, and predict future movement of stocks from Dhaka Stock Exchange using interactive tools.
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 👨‍💻 Team Members")

    col1, col2 = st.columns(2)
    with col1:
        st.success("**Alice Rahman**\n\nData Engineer\n\n📧 alice@example.com")
        st.success("**Babar Hossain**\n\nWeb Developer\n\n📧 babar@example.com")
    with col2:
        st.success("**Chandni Akter**\n\nAnalyst\n\n📧 chandni@example.com")
        st.success("**Danish Khan**\n\nML Engineer\n\n📧 danish@example.com")
        st.success("**Eliza Sultana**\n\nUI Designer\n\n📧 eliza@example.com")

    st.markdown("<p style='text-align:center; margin-top:50px; color:black;'>💡 Built by <strong>Team QuantumStock</strong></p>", unsafe_allow_html=True)

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

    # Get user inputs
    company_name = st.selectbox("Select company", sorted(enc_dict.keys()))
    company_id = enc_dict[company_name]  # map to encoded value

    month = st.selectbox("Month", list(range(1, 13)))
    openp = st.number_input("OPENP*", min_value=0.0, value=100.0)
    high = st.number_input("HIGH", min_value=0.0, value=105.0)
    low = st.number_input("LOW", min_value=0.0, value=95.0)
    closep = st.number_input("CLOSEP*", min_value=0.0, value=102.0)
    trade = st.number_input("TRADE", min_value=0, value=500)
    volume = st.number_input("VOLUME", min_value=0, value=10000)

    # When button is clicked, make prediction
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

        # Predict
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
    import streamlit as st
    from PIL import Image
    import os
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

        # --- Buttons centered at top ---
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

        # --- Show image centered ---
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

