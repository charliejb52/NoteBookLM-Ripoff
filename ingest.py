"""
Document Ingestion Script for Local RAG System

This script loads Apple 8-K filings (current reports) and ingests them into ChromaDB
for use in the RAG pipeline. 8-K filings contain recent, event-driven information
such as quarterly earnings announcements and press releases.
"""

from langchain_community.document_loaders import BSHTMLLoader, TextLoader, UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from download_apple_8k import find_latest_filing_files
from pathlib import Path
import os


# In the RAG pipeline:
#   Raw Files → Document Loader → Text Splitter → Embeddings → Vector Store → Retrieval → LLM

def load_filing_file(file_path):
    """
    Load an HTML or text filing file using the appropriate LangChain document loader.
    
    Args:
        file_path (str): Path to the HTML or text filing file
        
    Returns:
        list: List of Document objects containing the file content
    """
    file_ext = Path(file_path).suffix.lower()
    
    if file_ext in ['.htm', '.html']:
        # Use BSHTMLLoader for HTML files (uses BeautifulSoup)
        # Alternative: UnstructuredHTMLLoader for more complex HTML parsing
        try:
            loader = BSHTMLLoader(file_path)
        except Exception as e:
            print(f"BSHTMLLoader failed for {file_path}, trying UnstructuredHTMLLoader: {e}")
            loader = UnstructuredHTMLLoader(file_path)
    elif file_ext == '.txt':
        loader = TextLoader(file_path, encoding='utf-8')
    else:
        raise ValueError(f"Unsupported file type: {file_ext}")
    
    documents = loader.load()
    # Add source file path to metadata
    for doc in documents:
        doc.metadata['source_file'] = file_path
        doc.metadata['source_filename'] = Path(file_path).name
    
    return documents


def load_8k_filings(directory="data/apple_8k_filings", num_filings=10):
    """
    Load the last N 8-K filing files from the directory.
    
    Args:
        directory (str): Directory containing the downloaded 8-K filings
        num_filings (int): Number of most recent filings to load (default: 10)
        
    Returns:
        list: List of Document objects from all loaded files
    """
    # Find the last N filing files
    filing_files = find_latest_filing_files(directory, n=num_filings)
    
    if not filing_files:
        raise FileNotFoundError(f"No 8-K filing files found in {directory}")
    
    print(f"Found {len(filing_files)} 8-K filing file(s) to load:")
    for i, file_path in enumerate(filing_files, 1):
        print(f"  {i}. {Path(file_path).name}")
    
    # Load all files and combine documents
    all_documents = []
    for file_path in filing_files:
        try:
            print(f"\nLoading: {Path(file_path).name}...")
            documents = load_filing_file(file_path)
            all_documents.extend(documents)
            print(f"  ✓ Loaded {len(documents)} document(s)")
        except Exception as e:
            print(f"  ⚠ Error loading {file_path}: {e}")
            continue
    
    return all_documents


def split_documents_into_chunks(documents, chunk_size=1000, chunk_overlap=100):
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
    import sys
    
    # Default: load last 10 8-K filings
    num_filings = 10
    if len(sys.argv) > 1:
        try:
            num_filings = int(sys.argv[1])
        except ValueError:
            print(f"Invalid number: {sys.argv[1]}. Using default: {num_filings}")
    
    try:
        print("=" * 60)
        print("Apple 8-K Filing Ingestion")
        print("=" * 60)
        
        # Load the 8-K filing documents
        print(f"\nLoading last {num_filings} 8-K filings...")
        documents = load_8k_filings(
            directory="data/apple_8k_filings",
            num_filings=num_filings
        )
        
        # Print summary
        total_documents = len(documents)
        print(f"\n✓ Successfully loaded {total_documents} total document(s) from {num_filings} filing(s)")
        
        # Optional: Print some information about the first document
        if documents:
            print(f"\nFirst document preview:")
            print(f"Content length: {len(documents[0].page_content)} characters")
            print(f"Metadata: {documents[0].metadata}")
        
        # Split the documents into chunks
        # This is a critical step for RAG systems - chunks allow the LLM to process
        # documents that are too large to fit in context windows
        print(f"\nSplitting documents into chunks...")
        chunks = split_documents_into_chunks(
            documents,
            chunk_size=1000,    # Maximum 1000 characters per chunk
            chunk_overlap=100   # 100 characters overlap between chunks
        )
        
        # Print the number of chunks created
        print(f"✓ Total number of chunks created: {len(chunks)}")
        
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
        test_query = "What are Apple's quarterly earnings?"
        results = vector_store.similarity_search(test_query, k=2)
        print(f"  Query: '{test_query}'")
        print(f"  Retrieved {len(results)} most relevant chunks:")
        for i, result in enumerate(results, 1):
            print(f"    {i}. Chunk (length: {len(result.page_content)} chars)")
            print(f"       Preview: {result.page_content[:150]}...")
        
        print("\n" + "=" * 60)
        print("Ingestion Complete!")
        print("=" * 60)
        print("\nThe 8-K filings are now available for use in the RAG pipeline.")
        print("You can use chat.py to query the ingested documents.")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"\nTo fix this:")
        print(f"  1. Make sure you've run download_apple_8k.py first to download the filings")
        print(f"  2. Check that the 'data/apple_8k_filings' folder exists")
    except Exception as e:
        print(f"An error occurred while loading the filings: {e}")
        import traceback
        traceback.print_exc()

