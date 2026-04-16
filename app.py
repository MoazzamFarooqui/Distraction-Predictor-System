import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Page config
st.set_page_config(page_title="Distraction Predictor", page_icon="🧘", layout="centered")

# Load model
model = joblib.load("model.pkl")

# Styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;700&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
        background: radial-gradient(circle at 50% 50%, #1a1c2c 0%, #0e1117 100%);
        color: #e0e0e0;
    }

    .main {
        padding: 2rem;
    }

    /* Glassmorphism Container */
    .stContainer {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 2.5rem;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.4);
        margin-bottom: 2rem;
    }

    /* ASCII Header with Glow */
    .ascii-art {
        font-family: 'JetBrains Mono', monospace;
        white-space: pre;
        line-height: 1.1;
        background: rgba(0, 0, 0, 0.3);
        padding: 30px;
        border-radius: 15px;
        color: #ffffff;
        font-size: 11px;
        text-align: center;
        margin-bottom: 40px;
        border: 1px solid rgba(108, 92, 231, 0.2);
        box-shadow: 0 0 20px rgba(108, 92, 231, 0.1), inset 0 0 15px rgba(108, 92, 231, 0.05);
        text-shadow: 0 0 10px rgba(255, 255, 255, 0.3);
    }

    /* Custom Button */
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 4em;
        background: linear-gradient(135deg, #6c5ce7 0%, #a29bfe 100%);
        color: white;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        border: none;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        margin-top: 1rem;
    }

    .stButton>button:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(108, 92, 231, 0.4);
        color: #ffffff;
    }

    .stButton>button:active {
        transform: translateY(0);
    }

    /* Slider Styling */
    .stSlider [data-baseweb="slider"] {
        padding-top: 30px;
        padding-bottom: 10px;
    }

    .stSlider [data-testid="stMarkdownContainer"] p {
        font-weight: 600;
        color: #a29bfe;
        font-size: 0.9rem;
    }

    /* Results */
    .result-container {
        animation: fadeIn 0.8s ease-out;
        margin-top: 2rem;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .focused-card {
        background: linear-gradient(135deg, rgba(46, 213, 115, 0.1) 0%, rgba(46, 213, 115, 0.02) 100%);
        border: 1px solid rgba(46, 213, 115, 0.2);
        padding: 30px;
        border-radius: 15px;
        text-align: center;
    }

    .distracted-card {
        background: linear-gradient(135deg, rgba(255, 71, 87, 0.1) 0%, rgba(255, 71, 87, 0.02) 100%);
        border: 1px solid rgba(255, 71, 87, 0.2);
        padding: 30px;
        border-radius: 15px;
        text-align: center;
    }

    h3 {
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }

    .stDivider {
        margin: 3rem 0;
        opacity: 0.1;
    }
    </style>
    """, unsafe_allow_html=True)

# ASCII Art Header
st.markdown(r"""
<div class="ascii-art">
   ___ ___ ___ _____ ___   _   ___ _____ ___ ___  _  _ 
  |   \_ _/ __|_   _| _ \ /_\ / __|_   _|_ _/ _ \| \| |
  | |) | |\__ \ | | |   / _ \ (__  | |  | | (_) | .  |
  |___/___|___/ |_| |_|_\_/ \_\___| |_| |___\___/|_|\_|
   ___ ___ ___ ___ ___ ___ _____ ___  ___ 
  | _ \ _ \ __|   \_ _/ __|_   _/ _ \| _ \
  |  _/   / _|| |) || | (__  | || (_) |   /
  |_| |_|_\___|___/|___\___| |_| \___/|_|_\
   ___ _   _ ___ _____ ___ __  __ 
  / __| \ / / __|_   _| __|  \/  |
  \__ \\ V /\__ \ | | | _|| |\/| |
  |___/ |_| |___/ |_| |___|_|  |_|
</div>
""", unsafe_allow_html=True)

st.markdown("<p style='text-align: center; color: #a29bfe; font-weight: 400; font-size: 1.1rem; margin-top: -20px; margin-bottom: 40px;'>Predict your productivity through the lens of machine learning.</p>", unsafe_allow_html=True)

with st.container():
    st.markdown("<h3 style='margin-bottom: 1.5rem;'>Your Daily Metrics</h3>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        study_hours = st.slider("STUDY HOURS", 0, 12, 5)
    with col2:
        sleep_hours = st.slider("SLEEP HOURS", 0, 12, 7)
    with col3:
        phone_usage = st.slider("PHONE USAGE", 0, 12, 3)

    st.markdown("<div style='margin-top: 1.5rem;'>", unsafe_allow_html=True)
    if st.button("Analyze My Focus"):
        # Create a DataFrame with feature names to avoid warnings
        input_df = pd.DataFrame([[study_hours, sleep_hours, phone_usage]], 
                                 columns=["study_hours", "sleep_hours", "phone_usage"])
        
        prediction = model.predict(input_df)[0]
        
        st.markdown("</div>", unsafe_allow_html=True)
        st.divider()
        
        st.markdown("<div class='result-container'>", unsafe_allow_html=True)
        if prediction == 1:
            st.markdown("""
                <div class="focused-card">
                    <h1 style="font-size: 4rem; margin-bottom: 0;">🎯</h1>
                    <h2 style="color: #2ed573; margin-top: 0;">You are Focused!</h2>
                    <p style="color: #e0e0e0; font-size: 1.1rem; max-width: 500px; margin: 0 auto;">Excellent! Your current habits support high productivity. Keep maintaining this balance to reach your peak potential.</p>
                </div>
            """, unsafe_allow_html=True)
            st.balloons()
        else:
            st.markdown("""
                <div class="distracted-card">
                    <h1 style="font-size: 4rem; margin-bottom: 0;">⚠️</h1>
                    <h2 style="color: #ff4757; margin-top: 0;">You are Distracted!</h2>
                    <p style="color: #e0e0e0; font-size: 1.1rem; max-width: 500px; margin: 0 auto;">It seems your current routine might be hindering your focus. Consider reducing phone time or optimizing your sleep schedule.</p>
                </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)