"""
Document Ingestion Script for Local RAG System

This script demonstrates how to load PDF documents using LangChain's PyPDFLoader.
Document Loaders are a crucial component in the LLM workflow for RAG systems.
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
import os


# In the RAG pipeline:
#   Raw Files → Document Loader → Text Splitter → Embeddings → Vector Store → Retrieval → LLM

def load_pdf_from_data_folder(pdf_filename):
    """
    Load a PDF file from the data folder and return the loaded documents.
    
    Args:
        pdf_filename (str): Name of the PDF file in the data folder
        
    Returns:
        list: List of Document objects containing the PDF content
    """
    # Construct the full path to the PDF file in the data folder
    data_folder = os.path.join(os.path.dirname(__file__), "data")
    pdf_path = os.path.join(data_folder, pdf_filename)
    
    # Check if the file exists
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    # Create a PyPDFLoader instance
    # PyPDFLoader uses the PyPDF library under the hood to extract text from PDF files
    loader = PyPDFLoader(pdf_path)
    
    # Load the document
    # This reads the PDF and converts each page into a Document object
    # Each Document contains the page content and metadata (page number, source file, etc.)
    documents = loader.load()
    
    return documents


def split_documents_into_chunks(documents, chunk_size=10000, chunk_overlap=1000):
    """
    Split documents into smaller chunks using RecursiveCharacterTextSplitter.
    
    Args:
        documents (list): List of Document objects to split
        chunk_size (int): Maximum size of each chunk in characters (default: 1000)
        chunk_overlap (int): Number of characters to overlap between chunks (default: 100)
        
    Returns:
        list: List of Document chunks
    """
    # Create a RecursiveCharacterTextSplitter instance
    # This splitter tries to split on paragraph boundaries first, then sentences,
    # then words, and finally characters if needed. This helps preserve semantic
    # meaning better than simple character-based splitting.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,      # Maximum characters per chunk
        chunk_overlap=chunk_overlap, # Overlap between chunks to preserve context
        length_function=len,         # Function to measure chunk length
        separators=["\n\n", "\n", " ", ""]  # Splitting hierarchy: paragraphs → sentences → words → chars
    )
    
    # Split the documents into chunks
    # This combines all pages into one text, then intelligently splits it
    chunks = text_splitter.split_documents(documents)
    
    # Filter complex metadata that Chroma can't handle
    # This ensures only simple metadata fields (source, page numbers, etc.) are stored
    # Complex nested metadata structures would cause errors in Chroma
    filtered_chunks = filter_complex_metadata(chunks)
    
    return filtered_chunks


def create_vector_store(chunks, persist_directory="chroma_db", embedding_model="nomic-embed-text"):
    """
    Create embeddings for text chunks and store them in a Chroma vector database.
    
    Args:
        chunks (list): List of Document chunks to embed and store
        persist_directory (str): Directory path where the vector database will be stored
        embedding_model (str): Name of the Ollama embedding model to use
        
    Returns:
        Chroma: Vector store instance containing the embedded chunks
    """
    # Initialize the embedding model
    # OllamaEmbeddings uses a local Ollama instance to generate embeddings
    # The model "nomic-embed-text" is a high-quality, open-source embedding model
    # that creates 768-dimensional vectors capturing semantic meaning
    print(f"\nInitializing embedding model: {embedding_model}...")
    embeddings = OllamaEmbeddings(model=embedding_model)
    
    # Create the vector store using Chroma
    # Chroma is an open-source, lightweight vector database that:
    # - Stores embeddings locally in the specified directory
    # - Provides fast similarity search capabilities
    # - Persists data to disk so it can be reused later
    print(f"Creating vector store in directory: {persist_directory}...")
    vector_store = Chroma.from_documents(
        documents=chunks,           # The text chunks to embed and store
        embedding=embeddings,       # The embedding model to use
        persist_directory=persist_directory  # Where to save the database
    )
    
    # Persist the vector store to disk
    # This ensures the embeddings are saved and can be loaded later without
    # needing to re-embed all the documents
    vector_store.persist()
    
    return vector_store


if __name__ == "__main__":
    # Example: Load a PDF from the data folder
    pdf_filename = "Neamen.pdf"
    
    try:
        # Load the PDF documents
        documents = load_pdf_from_data_folder(pdf_filename)
        
        # Print the total number of pages
        # The number of documents typically equals the number of pages in the PDF
        total_pages = len(documents)
        print(f"Successfully loaded PDF: {pdf_filename}")
        print(f"Total number of pages: {total_pages}")
        
        # Optional: Print some information about the first page
        if documents:
            print(f"\nFirst page preview:")
            print(f"Page content length: {len(documents[0].page_content)} characters")
            print(f"Metadata: {documents[0].metadata}")
        
        # Split the documents into chunks
        # This is a critical step for RAG systems - see comments above for why
        print(f"\nSplitting documents into chunks...")
        chunks = split_documents_into_chunks(
            documents,
            chunk_size=1000,    # Maximum 1000 characters per chunk
            chunk_overlap=100   # 100 characters overlap between chunks
        )
        
        # Print the number of chunks created
        print(f"Total number of chunks created: {len(chunks)}")
        
        
        
        # Create embeddings and store in vector database
        # This converts each chunk into a numerical vector and stores it for fast similarity search
        print(f"\nCreating embeddings and storing in vector database...")
        vector_store = create_vector_store(
            chunks,
            persist_directory="chroma_db",
            embedding_model="nomic-embed-text"
        )
        
        # Verify the vector store was created successfully
        print(f"✓ Vector store created successfully!")
        print(f"  Database location: chroma_db/")
        print(f"  Total documents stored: {len(chunks)}")
        
        # Optional: Test the vector store with a sample query
        print(f"\nTesting vector store with a sample query...")
        test_query = "What is a MOSFET?"
        results = vector_store.similarity_search(test_query, k=2)
        print(f"  Query: '{test_query}'")
        print(f"  Retrieved {len(results)} most relevant chunks:")
        for i, result in enumerate(results, 1):
            print(f"    {i}. Chunk (length: {len(result.page_content)} chars)")
            print(f"       Preview: {result.page_content[:150]}...")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Please make sure the PDF file '{pdf_filename}' exists in the 'data' folder.")
    except Exception as e:
        print(f"An error occurred while loading the PDF: {e}")

