"""
Chat Script for Local RAG System

This script demonstrates how to retrieve relevant information from the vector database
using similarity search. It connects to the existing Chroma database and retrieves
the most relevant chunks for a given query.
"""

from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
import os


def load_vector_store(persist_directory="chroma_db", embedding_model="nomic-embed-text"):
    """
    Load an existing Chroma vector store from disk.
    
    Args:
        persist_directory (str): Directory path where the vector database is stored
        embedding_model (str): Name of the Ollama embedding model (must match the one used during ingestion)
        
    Returns:
        Chroma: Vector store instance
    """
    # Check if the database directory exists
    if not os.path.exists(persist_directory):
        raise FileNotFoundError(
            f"Vector database not found at '{persist_directory}'. "
            f"Please run ingest.py first to create the database."
        )
    
    # Initialize the same embedding model used during ingestion
    # This is crucial - the embedding model must match, otherwise the embeddings
    # won't be compatible for similarity search
    print(f"Loading embedding model: {embedding_model}...")
    embeddings = OllamaEmbeddings(model=embedding_model)
    
    # Load the existing vector store from disk
    # Chroma will automatically load all the stored embeddings and metadata
    print(f"Loading vector store from: {persist_directory}...")
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    
    return vector_store


def retrieve_relevant_chunks(vector_store, query, k=3):
    """
    Retrieve the most relevant chunks for a given query using similarity search.
    
    Args:
        vector_store (Chroma): The vector store instance
        query (str): The user's question or query
        k (int): Number of top results to retrieve (default: 3)
        
    Returns:
        list: List of Document objects containing the most relevant chunks
    """
    # Use similarity_search to find the most relevant chunks
    # This converts the query into an embedding and finds the chunks with
    # the most similar embeddings using cosine similarity
    results = vector_store.similarity_search(query, k=k)
    
    return results


if __name__ == "__main__":
    try:
        # Load the existing vector store
        print("=" * 60)
        print("Loading Vector Database")
        print("=" * 60)
        vector_store = load_vector_store(
            persist_directory="chroma_db",
            embedding_model="nomic-embed-text"
        )
        
        # Get the number of documents in the database
        # Note: We can't easily get the count from Chroma, but we know it exists
        print("✓ Vector store loaded successfully!")
        print()
        
        # Test query
        query = "What are the course prerequisites?"
        
        print("=" * 60)
        print("Testing Retrieval Mechanism")
        print("=" * 60)
        print(f"Query: '{query}'")
        print(f"Retrieving top 3 most relevant chunks...")
        print()
        
        # Retrieve the most relevant chunks
        relevant_chunks = retrieve_relevant_chunks(vector_store, query, k=3)
        
        # Display the results
        print(f"Found {len(relevant_chunks)} relevant chunks:")
        print()
        
        for i, chunk in enumerate(relevant_chunks, 1):
            print("-" * 60)
            print(f"Chunk {i}:")
            print(f"  Length: {len(chunk.page_content)} characters")
            print(f"  Metadata: {chunk.metadata}")
            print(f"  Content:")
            print(f"  {chunk.page_content}")
            print()
        
        print("=" * 60)
        print("✓ Retrieval mechanism is working correctly!")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nTo fix this:")
        print("  1. Make sure you've run ingest.py first to create the vector database")
        print("  2. Check that the 'chroma_db' folder exists in the current directory")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

