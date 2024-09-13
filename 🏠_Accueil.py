import os
# from st_pages import show_pages_from_config

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage



st.set_page_config(
        page_title="La boussole",
)

#from streamlit_extras.app_logo import add_logo
#add_logo("images/icon_accidents.png", height=10)

# show_pages_from_config()

st.title("Accueil")
