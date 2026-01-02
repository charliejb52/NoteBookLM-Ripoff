"""
Debug Script to View All Chunks in Chroma Database

This script loads the vector database and displays all stored chunks
for debugging purposes.
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
    print(f"Loading embedding model: {embedding_model}...")
    embeddings = OllamaEmbeddings(model=embedding_model)
    
    # Load the existing vector store from disk
    print(f"Loading vector store from: {persist_directory}...")
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    
    return vector_store


def display_all_chunks(vector_store):
    """
    Retrieve and display all chunks from the vector store.
    
    Args:
        vector_store (Chroma): The vector store instance
    """
    # Get all documents from the vector store
    # We can use a very broad query or access the collection directly
    print("=" * 80)
    print("Retrieving all chunks from vector database...")
    print("=" * 80)
    
    # Use a very generic query to get all chunks, or we can access the collection
    # For Chroma, we can use get() to retrieve all documents
    try:
        # Try to get all documents using the collection
        all_docs = vector_store.get()
        
        if all_docs and 'documents' in all_docs:
            total_chunks = len(all_docs['documents'])
            print(f"\nTotal chunks in database: {total_chunks}\n")
            
            # Display each chunk
            for i, (doc_id, doc_content, metadata) in enumerate(zip(
                all_docs.get('ids', []),
                all_docs['documents'],
                all_docs.get('metadatas', [{}] * len(all_docs['documents']))
            ), 1):
                print("=" * 80)
                print(f"CHUNK {i} of {total_chunks}")
                print("=" * 80)
                print(f"ID: {doc_id}")
                print(f"Length: {len(doc_content)} characters")
                print(f"Metadata: {metadata}")
                print(f"\nContent:")
                print("-" * 80)
                print(doc_content)
                print("-" * 80)
                print()
        else:
            print("No documents found in the database.")
            
    except Exception as e:
        print(f"Error retrieving documents: {e}")
        print("\nTrying alternative method using similarity search...")
        # Fallback: use a very generic query to get many results
        try:
            # Use a very generic query and request many results
            results = vector_store.similarity_search("", k=100)  # Empty query, get many results
            print(f"\nRetrieved {len(results)} chunks using similarity search:\n")
            
            for i, chunk in enumerate(results, 1):
                print("=" * 80)
                print(f"CHUNK {i} of {len(results)}")
                print("=" * 80)
                print(f"Length: {len(chunk.page_content)} characters")
                print(f"Metadata: {chunk.metadata}")
                print(f"\nContent:")
                print("-" * 80)
                print(chunk.page_content)
                print("-" * 80)
                print()
        except Exception as e2:
            print(f"Alternative method also failed: {e2}")


if __name__ == "__main__":
    try:
        # Load the vector store
        print("Loading Vector Database...")
        vector_store = load_vector_store(
            persist_directory="chroma_db",
            embedding_model="nomic-embed-text"
        )
        print("✓ Vector store loaded successfully!\n")
        
        # Display all chunks
        display_all_chunks(vector_store)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nTo fix this:")
        print("  1. Make sure you've run ingest.py first to create the vector database")
        print("  2. Check that the 'chroma_db' folder exists in the current directory")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

