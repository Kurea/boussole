import streamlit as st
from utils import get_llm_config, _get_session
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from QA_LLMGB import QA_RAG, clear_chat_history
from langchain_community.graphs import Neo4jGraph


st.title("Interrogation de la boussole")

client, chatmodel, embeddings = get_llm_config()

def get_response(user_query, chat_history):

    template = """
    You are an AI-powered question-answering agent. Your task is to provide accurate and concise responses to user queries based on the given context, chat history, and available resources.

    ### Response Guidelines:
    1. **Direct Answers**: Provide straightforward answers to the user's queries without headers unless requested. Avoid speculative responses.
    2. **Utilize History and Context**: Leverage relevant information from previous interactions, the current user input, and the context provided below.
    3. **No Greetings in Follow-ups**: Start with a greeting in initial interactions. Avoid greetings in subsequent responses unless there's a significant break or the chat restarts.
    4. **Admit Unknowns**: Clearly state if an answer is unknown. Avoid making unsupported statements.
    5. **Avoid Hallucination**: Only provide information based on the context provided. Do not invent information.
    6. **Response Length**: Keep responses concise and relevant. Aim for clarity and completeness within 2-3 sentences unless more detail is requested.
    7. **Tone and Style**: Maintain a professional and informative tone. Be friendly and approachable.
    8. **Error Handling**: If a query is ambiguous or unclear, ask for clarification rather than providing a potentially incorrect answer.
    9. **Fallback Options**: If the required information is not available in the provided context, provide a polite and helpful response. Example: "I don't have that information right now." or "I'm sorry, but I don't have that information. Is there something else I can help with?"

    Answer the following questions considering the history of the conversation:

    Chat history: {chat_history}

    User question: {user_question}
    """

    prompt = ChatPromptTemplate.from_template(template)

        
    chain = prompt | chatmodel | StrOutputParser()
    
    return chain.stream({
        "chat_history": chat_history,
        "user_question": user_query,
    })

url = st.secrets["AAA_URI"]
username = st.secrets["AAA_USERNAME"]
password = st.secrets["AAA_PASSWORD"]
graph = Neo4jGraph(url=url, username=username, password=password, refresh_schema=False, sanitize=True)

def clear_history():
    st.session_state.chat_history = [AIMessage(content="Hello, I am a bot. How can I help you?")]
    clear_chat_history(graph,_get_session())

st.button("Vider l'historique", on_click=clear_history())


if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="Hello, I am a bot. How can I help you?")]

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
        #response = st.write_stream(get_response(user_query, st.session_state.chat_history))
        response = st.write_stream(QA_RAG(graph, user_query, _get_session()))
    st.session_state.chat_history.append(AIMessage(content=response))
