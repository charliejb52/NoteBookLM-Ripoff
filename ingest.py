"""
Document Ingestion Script for Local RAG System

This script demonstrates how to load PDF documents using LangChain's PyPDFLoader.
Document Loaders are a crucial component in the LLM workflow for RAG systems.
"""

from langchain_community.document_loaders import PyPDFLoader
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


if __name__ == "__main__":
    # Example: Load a PDF from the data folder
    # Replace 'your_syllabus.pdf' with the actual name of your PDF file
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
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Please make sure the PDF file '{pdf_filename}' exists in the 'data' folder.")
    except Exception as e:
        print(f"An error occurred while loading the PDF: {e}")

