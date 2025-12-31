"""
Document Ingestion Script for Local RAG System

This script demonstrates how to load PDF documents using LangChain's PyPDFLoader.
Document Loaders are a crucial component in the LLM workflow for RAG systems.
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# What is a Document Loader in the LLM workflow?
# ===============================================
# A Document Loader is the first step in processing documents for RAG (Retrieval-Augmented Generation).
# It's responsible for:
# 1. Reading raw files (PDFs, text files, etc.) from various sources
# 2. Extracting text content and metadata from these files
# 3. Converting the content into a standardized "Document" format that LangChain can work with
# 4. Preparing documents for the next stages: text splitting, embedding, and vector storage
#
# In the RAG pipeline:
#   Raw Files → Document Loader → Text Splitter → Embeddings → Vector Store → Retrieval → LLM
#
# Without document loaders, we'd have to manually parse different file formats, which would be
# time-consuming and error-prone. LangChain's loaders handle the complexity of different file
# formats (PDF, Word, HTML, etc.) and provide a unified interface for document processing.

# Why do we need to split text into chunks?
# ==========================================
# Splitting text into smaller chunks is essential for RAG systems for several critical reasons:
#
# 1. **Token Limits**: LLMs have maximum context window sizes (e.g., 4K, 8K, 32K tokens).
#    A full PDF can easily exceed these limits, causing errors or truncation.
#
# 2. **Precision in Retrieval**: When searching for relevant information, smaller chunks allow
#    the system to retrieve the most specific and relevant sections rather than entire documents.
#    This improves answer quality by focusing on exactly what's needed.
#
# 3. **Embedding Quality**: Embeddings (vector representations) work better with semantically
#    coherent chunks. Smaller, focused chunks create more meaningful embeddings that capture
#    specific concepts or topics.
#
# 4. **Efficiency**: Processing smaller chunks is faster and more cost-effective, especially
#    when generating embeddings or performing similarity searches.
#
# 5. **Context Relevance**: When a user asks a question, the system retrieves only the most
#    relevant chunks. This means the LLM receives focused context, leading to more accurate
#    and relevant answers without being overwhelmed by irrelevant information.
#
# Without chunking, you'd either hit token limits, get less precise retrieval, or waste
# computational resources processing irrelevant parts of documents.

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
    
    return chunks


if __name__ == "__main__":
    # Example: Load a PDF from the data folder
    pdf_filename = "ECE230L_Syllabus_F2025.pdf"
    
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
        
        # Optional: Print information about the chunks
        if chunks:
            print(f"\nChunk size statistics:")
            chunk_sizes = [len(chunk.page_content) for chunk in chunks]
            print(f"  Average chunk size: {sum(chunk_sizes) / len(chunk_sizes):.0f} characters")
            print(f"  Smallest chunk: {min(chunk_sizes)} characters")
            print(f"  Largest chunk: {max(chunk_sizes)} characters")
            print(f"\nFirst chunk preview:")
            print(f"  Content length: {len(chunks[0].page_content)} characters")
            print(f"  Content preview: {chunks[0].page_content[:200]}...")
            print(f"  Metadata: {chunks[0].metadata}")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Please make sure the PDF file '{pdf_filename}' exists in the 'data' folder.")
    except Exception as e:
        print(f"An error occurred while loading the PDF: {e}")

