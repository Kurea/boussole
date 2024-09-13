import os
import json
from requests import get
from utils import _get_session
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from streamlit.runtime.scriptrunner import get_script_run_ctx

def get_session_id():
    ctx = get_script_run_ctx()
    if ctx is None:
        return None

    return ctx.session_id

st.set_page_config(
        page_title="La boussole",
)

st.title("Score d'immunité")

def get_response(user_query):
    # Call API with query and sessionId
    response = get("https://n8n.cloudron.interrest.fr/webhook/6ca7fc09-98a2-4996-8293-22026d70d14d?chatInput="+user_query+"&sessionId="+get_session_id())
    return response.json()

def clear_history():
    st.session_state.chat_history = []
    get_response("reset")
score = st.container()
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

    response = get_response(user_query)
    score.write("### score : "+response["score"])
    with st.chat_message("AI"):
        if "question" in response:
            st.write(response["question"])
        if "recommandations" in response:
            st.write(response["recommandations"])
            st.balloons()
        if "response" in response:
            st.write(response["response"])
        if "suggestions" in response:
            cols = st.columns(len(response["suggestions"]))
            for i, sug in enumerate(response["suggestions"]):
                cols[i].button(sug, on_click=clear_history())
        st.session_state.chat_history.append(AIMessage(content=response["question"]))
else:
    score.write("### score : 50")