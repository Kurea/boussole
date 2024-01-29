import streamlit as st
import neo4j
from neo4j import GraphDatabase
# Recherche de base : chercher Google sur Google...
from googlesearch import search

st.title("📝 Créer un nouvel observatoire sur les accidents")

question = st.text_input(
    "Renseigner les termes de recherche",
    placeholder="accident+voiture",
)

#term = "accident+moto+grievement"

results = search(question, lang="fr", num_results=50, advanced=True)


# Neo4j connection details
url = st.secrets["AAA_URI"]
username = st.secrets["AAA_USERNAME"]
password = st.secrets["AAA_PASSWORD"]

# Function to add an event to the Neo4j database
def add_event(session, url, description):
    session.run("MERGE (e:Event {url: $url}) ON CREATE SET e.description = $description", url=url, description=description)

# Create a driver instance
driver = GraphDatabase.driver(uri, auth=(username, password))

# Insert data from the DataFrame
with driver.session() as session:
    for result in results:
        add_event(session, result.url, result.description)

# Close the driver
driver.close()
