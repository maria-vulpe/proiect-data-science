import streamlit as st
import matplotlib.pyplot as plt

from streamlit_option_menu import option_menu

from home import render as render_home
from clinician import render as render_clinician
from data_scientist import render as render_data_scientist
from data_scientist_models import render as render_data_scientist_models
from cardiac_angiography import render as render_cardiac_angiography
from heart_reconstruction_diagnosis import render as render_heart
plt.style.use("dark_background")
plt.rcParams.update({
    "figure.facecolor": "#2C3E50",
    "axes.facecolor": "#2C3E50",
    "axes.edgecolor": "#ECF0F1",
    "axes.labelcolor": "#ECF0F1",
    "text.color": "#ECF0F1",
    "xtick.color": "#BDC3C7",
    "ytick.color": "#BDC3C7",
    "legend.facecolor": "#2C3E50",
    "legend.edgecolor": "#1ABC9C"
})

st.set_page_config(page_title="HealthTrack AI", page_icon="❤️", layout="wide")

st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Global CSS
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;500;700&display=swap" rel="stylesheet">
<style>
body, .stApp { font-family: 'Inter', sans-serif; background: #1E1E2F; color: #ECF0F1; }
.main .block-container { padding: 1rem 2rem; background: #2C3E50; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.5); }
h1, h2, h3, h4 { color: #1ABC9C; font-weight: 700; }
.nav-link-icon { font-size: 1.2rem; margin-right: 8px; }
.nav-link:hover { background-color: #189070 !important; }
.nav-link-selected { background-color: #1ABC9C !important; }
.feature-card {
  background: #273241; border-radius: 12px; padding: 1.5rem; text-align: center;
  transition: transform .2s ease, box-shadow .2s ease;
  box-shadow: 0 4px 12px rgba(0,0,0,0.4);
}
.feature-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 8px 24px rgba(0,0,0,0.6);
}
.feature-card h4 { margin-bottom: 0.5rem; color: #1ABC9C; }
.feature-card p { font-size: 0.9rem; color: #BDC3C7; }

details > div.preview {
  background: #1E1E2F;
  padding: 0.75rem 1rem;
  border-radius: 0 0 12px 12px;
  box-shadow: inset 0 4px 12px rgba(0,0,0,0.2);
  max-height: 300px;
  overflow-y: auto;
} 
        [data-testid="stSidebar"] {
            display: none !important;
        }
        [data-testid="stSidebarNav"] {
            display: none !important;
        }
        [data-testid="collapsedControl"] {
            display: none !important;
        }
        [data-testid="stPageNav"] {
            display: none !important;
        }
        div[role="navigation"] {
            display: none !important;
        }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.nav-link:hover {
    background-color: #1ABC9C !important;
    color: #1E1E2F !important;
}
</style>
""", unsafe_allow_html=True)

# Navigation menu
st.markdown('<div class="nav-container">', unsafe_allow_html=True)
selected = option_menu(
    menu_title=None,
    options=["Home", "Clinician", "Data Scientist", "DS Models", "Cardiac Angiography", "Heart Reconstruction"],
    icons=["house", "person-badge", "gear", "bar-chart", "heart-pulse", "activity"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding":"0!important","background-color":"#2C3E50","border-radius":"8px","box-shadow":"0 4px 12px rgba(0,0,0,0.4)"},
        "nav-link": {"font-size":"16px","color":"#BDC3C7","padding":"8px 16px","margin":"0px 8px","border-radius":"6px","background-color":"transparent"},
        "nav-link-selected": {"background-color":"#1ABC9C","color":"#1E1E2F","border-radius":"6px"},
        "icon": {"color":"#1ABC9C", "font-size":"18px"}
    }
)
st.markdown('</div>', unsafe_allow_html=True)



if selected == "Home":
    render_home()
elif selected == "Clinician":
    render_clinician()
elif selected == "DS Models":
    render_data_scientist_models()
elif selected == "Cardiac Angiography":
    render_cardiac_angiography()
elif selected == "Heart Reconstruction":
    render_heart()
else:
    render_data_scientist()