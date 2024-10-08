#import altair as alt
import numpy as np
#import pandas as pd
import streamlit as st
#import neo4j as neo4j
#import googlesearch as googlesearch
import langchain
import langchain_community
import os
from utils import get_llm_config


st.set_page_config(
        page_title="La boussole",
)

#add_logo("images/icon_accidents.png", height=10)

"""
# Observatoire des accidents
"""

#st.image('images/chatgpt_accident.png', caption='Accidents', width=500)

# openai_api_key = st.secrets["OPENAI_KEY"]

from langchain_community.graphs import Neo4jGraph
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Neo4jVector


llm, chatmodel, embeddings = get_llm_config()
url = st.secrets["AAA_URI"]
username = st.secrets["AAA_USERNAME"]
password = st.secrets["AAA_PASSWORD"]
graph = Neo4jGraph(
    url=url,
    username=username,
    password=password
)

vectorstore = Neo4jVector.from_existing_graph(
    embeddings,
    url=url,
    username=username,
    password=password,
    index_name='articledescription',
    node_label="Article",
    text_node_properties=['titre', 'description', 'texte'],
    embedding_node_property='embedding',
)

vector_qa = RetrievalQA.from_chain_type(
    llm=chatmodel, chain_type="stuff", retriever=vectorstore.as_retriever())

contextualize_query = """
match (node)-[:DOCUMENTE]->(e:Evenement)
WITH node AS a, e, score, {} as metadata limit 1
OPTIONAL MATCH (e)<-[:EXPLIQUE]-(f:Facteur)-[:EXPLIQUE]->(e2:Evenement)
WITH a, e, score, metadata, apoc.text.join(collect(e2.description), ",") AS autres_evenements
RETURN "Evenement : "+ e.description + " autres événements expliqués par les mêmes facteurs : " + coalesce(autres_evenements, "") +"\n" as text, score, metadata
"""

contextualize_query1 = """
match (node)-[:DOCUMENTE]->(e:Evenement)
WITH node AS a, e, score, {} as metadata limit 1
OPTIONAL MATCH (e)<-[:EXPLIQUE]-(:Facteur)
WITH a, e, i, f, score, metadata
RETURN "Titre Article: "+ a.titre + " description: "+ a.description + " facteur: "+ coalesce(f.name, "")+ "\n" as text, score, metadata
"""

contextualized_vectorstore = Neo4jVector.from_existing_index(
    embeddings,
    url=url,
    username=username,
    password=password,
    index_name="articledescription",
    retrieval_query=contextualize_query,
)

vector_plus_context_qa = RetrievalQA.from_chain_type(
    llm=chatmodel, chain_type="stuff", retriever=contextualized_vectorstore.as_retriever())

# Streamlit layout with tabs
container = st.container()
question = container.text_input("**:blue[Question:]**", "")

if question:
    tab1, tab2, tab3 = st.tabs(["No-RAG", "Basic RAG", "Augmented RAG"])
    with tab1:
        st.markdown("**:blue[No-RAG.] LLM seulement. Réponse générée par l'IA générative seule:**")
        st.write(llm(question))
    with tab2:
        st.markdown("**:blue[Basic RAG.] Réponse par recherche vectorielle:**")
        st.write(vector_qa.run(question))
    with tab3:
        st.markdown("**:blue[Augmented RAG.] Réponse par recherche vectorielle ET par augmentation de contexte:**")
        st.write(vector_plus_context_qa.run(question))
