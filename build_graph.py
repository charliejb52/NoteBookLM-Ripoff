import os
model_name = "gpt-4.1-2025-04-14"
os.environ["OPENAI_API_KEY"] = "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"
import re
import ast
import spacy
from typing import List
from langchain_openai import ChatOpenAI
from langchain_community.graphs import Neo4jGraph
from langchain.prompts import PromptTemplate



# Config

NEO4J_URL='bolt://localhost:7687'
NEO4J_USERNAME = 'neo4j'
NEO4J_PASSWORD = '553-I6WqkRtLhbxyloL6n8ZJth3G8xADg1cUhZ91Hgg'

# Setup

nlp = spacy.load("en_core_web_trf")
llm = ChatOpenAI(model=model_name, temperature=0)
graph = Neo4jGraph(
    url=NEO4J_URL,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
)

def extract_entities(text: str) -> List[str]:
    """
    Extract entities from text using spaCy.
    """
    doc = nlp(text)
    entities = set()
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PERSON", "GPE", "LOC", "PRODUCT", "EVENT", "LAW", "LANGUAGE", "DATE", "TIME", "MONEY", "PERCENT", "QUANTITY", "ORDINAL", "CARDINAL"]:
            entities.add(ent.text)
    return list(entities)


def extract_triples(text: str, entities: List[str]) -> List[str]:
    prompt = PromptTemplate.from_template(
        "given the following entities: {entities}\n"
        "Extract every factual (subject, relationship, object) triple connecting them in the text below for a knowledge graph. "
        "Include cross sentence and implied relations. Do not invent. "
        "Output as a python list of tuples.\n"
        "Text:\n{text}\n\nTriples:"
    )

    chain = prompt | llm
    results = chain.invoke({"entities": ", ".join(entities), "text": text}).content.strip()
    try:
        start = results.find("[")
        end = results.find("]")
        triples_list_str = results[start:end+1]
        triples = ast.literal_eval(triples_list_str)
        return [", ".join([str(i).strip() for i in triple]) for triple in triples if len(triple) == 3]
    except Exception as e:
        print(f"Error extracting triples: {e}")
        print(results)
        print(f"Error type: {type(e)}")
        return []

    
def safe_var(name: str) -> str:
    name = name.strip().replace("-", "_").replace("'", "").replace("`", "")
    name = name.re.sub(r"[^a-zA-Z0-9_]", "_", name)
    name = name.re.sub(r"^[^a-zA-Z_]+", "", name)
    if not name:
        name = "Entity"
    return name

def cypher_escape(s: str) -> str:
    return s.replace("'", "\\'")


# create triples into cypher queries that we can send to Neo4j
def triples_to_cypher(triples: List[str]) -> List[str]:
    cyphers = []
    for triple in triples:
        parts = [p.strip() for p in triple.split(",")]
        if len(parts) == 3 and all(parts):
            subject, relation, object = parts
            subject_var = safe_var(subject)
            relation_var = safe_var(relation)
            object_var = safe_var(object)
            subject_escaped = cypher_escape(subject_var)
            object_escaped = cypher_escape(object_var)
            cypher = (
                f"MERGE ({subject_var}:Entity {{name: '{subject_escaped}'}})\n"
                f"MERGE ({object_var}:Entity {{name: '{object_escaped}'}})\n"
                f"MERGE ({subject_var})-[:{relation_var}]->({object_var})"
            )
            cyphers.append(cypher)
    return cyphers

# add triples to the graph
def add_triples_to_graph(triples: List[str]) -> None:
    cyphers = triples_to_cypher(triples)
    for cypher in cyphers:
        graph.query(cypher)


def ingest_text(text: str) -> None:
    entities = extract_entities(text)
    triples = extract_triples(text, entities)
    add_triples_to_graph(triples)


