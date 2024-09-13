import os
from st_pages import show_pages_from_config

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage



st.set_page_config(
        page_title="La boussole",
)

#from streamlit_extras.app_logo import add_logo
#add_logo("images/icon_accidents.png", height=10)

show_pages_from_config()

st.title("Score d'immunité")

def get_response(user_query):
    # Call API with query and sessionId
    a = ""

def clear_history():
    st.session_state.chat_history = []
    get_response("reset")

st.button("Vider l'historique", on_click=clear_history())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)


user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        response = st.write_stream(get_response(user_query))
        st.session_state.chat_history.append(AIMessage(content=response))
