import streamlit as st

from home import home
from blazepose import blazepose
from openpose import openpose
from PIL import Image

icon = Image.open("images/icon.jpg")
st.set_page_config(
    page_title="REAL TIME HUMAN POSE ESTIMATION:",
    page_icon=icon,
    layout="wide",
    initial_sidebar_state="auto",
)

st.sidebar.markdown("<h1 style='color:Red ;'>Navigation</h1>", unsafe_allow_html=True)
page = st.sidebar.radio(" ",("Home","BlazePose","OpenPose"))

if page == "BlazePose":
    blazepose()
elif page == "OpenPose":
    openpose()
else:
    home()
    