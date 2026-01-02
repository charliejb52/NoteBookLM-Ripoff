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

# What is an Embedding?
# =====================
# An embedding is a numerical representation of text that captures its semantic meaning.
# Think of it as converting words and sentences into a dense vector (array of numbers)
# in a high-dimensional space (typically 384, 768, or 1536 dimensions).
#
# Key properties of embeddings:
# 1. **Semantic Similarity**: Similar meanings result in similar vectors. For example,
#    "dog" and "puppy" will have embeddings that are close together in vector space.
# 2. **Mathematical Operations**: You can measure similarity between texts using
#    mathematical operations like cosine similarity or Euclidean distance.
# 3. **Fixed Size**: Regardless of text length, embeddings are always the same size,
#    making them easy to store and compare.
# 4. **Context-Aware**: Modern embedding models (like nomic-embed-text) understand
#    context, so "bank" (financial) and "bank" (river) get different embeddings.
#
# In RAG systems, embeddings allow us to:
# - Convert text chunks into searchable vectors
# - Find semantically similar content (not just keyword matches)
# - Retrieve relevant information based on meaning, not just exact text matches

# Why use a Vector Database instead of a regular SQL database?
# ============================================================
# Vector databases (like Chroma) are specialized for storing and searching embeddings,
# while SQL databases are designed for structured relational data. Here's why we need them:
#
# 1. **Similarity Search**: Vector databases excel at finding similar vectors using
#    algorithms like cosine similarity, Euclidean distance, or approximate nearest
#    neighbor (ANN) search. SQL databases would require complex, slow queries to
#    compare high-dimensional vectors.
#
# 2. **Performance**: Vector databases use optimized indexing structures (like HNSW,
#    IVF, or LSH) that can search millions of vectors in milliseconds. SQL databases
#    would need to compute distances for every row, which is extremely slow.
#
# 3. **Scalability**: Vector databases are designed to handle billions of high-dimensional
#    vectors efficiently. SQL databases struggle with this scale for similarity operations.
#
# 4. **Semantic Search**: Unlike SQL's exact match or LIKE queries, vector databases
#    enable semantic search - finding content based on meaning, not just keywords.
#    For example, searching for "automobile" will also find documents about "cars".
#
# 5. **Specialized Operations**: Vector databases support operations like:
#    - Approximate nearest neighbor (ANN) search
#    - Multi-vector queries (searching with multiple query vectors)
#    - Hybrid search (combining vector similarity with metadata filters)
#
# In our RAG system:
#   Text Chunks → Embeddings → Vector Database → Similarity Search → Relevant Chunks → LLM
#
# When a user asks a question, we:
# 1. Convert the question into an embedding
# 2. Search the vector database for similar chunk embeddings
# 3. Retrieve the most relevant chunks
# 4. Pass those chunks as context to the LLM for answering

# How does Markdown formatting help LLMs understand table structure?
# ===================================================================
# Markdown tables provide explicit structural information that helps LLMs
# understand the relationships between data points:
#
# 1. **Explicit Column Headers**: Markdown tables clearly separate headers
#    from data rows using a separator line (|-----|-----|). This tells the
#    LLM which values belong to which columns.
#
#    Example:
#    | Exam | Date | Weight |
#    |------|------|--------|
#    | Midterm | Oct 15 | 30% |
#
#    The LLM can clearly see "Oct 15" is the Date for the Midterm exam.
#
# 2. **Row Delimiters**: The pipe character (|) acts as a visual delimiter
#    that separates columns. When the LLM processes "| Exam | Date |", it
#    understands these are distinct columns, not just words next to each other.
#
# 3. **Alignment Information**: Markdown tables preserve the logical structure
#    - each row represents one record
#    - each column represents one attribute
#    - cells in the same column share the same semantic meaning
#
# 4. **Context Preservation**: When a table is converted to Markdown, the
#    relationships are preserved in a linear text format that LLMs can process
#    naturally. The LLM can "read" the table row by row and understand that
#    values in the same row are related.
#
# 5. **Query Understanding**: When you ask "What is the date of the final exam?",
#    the LLM can:
#    - Identify "final exam" in the Exam column
#    - Look across the same row to find the Date column value
#    - Understand that these values are connected because they're in the same row
#
# Without Markdown formatting, tables often become unstructured text like:
#   "Exam Date Weight Midterm Oct 15 30% Final Dec 10 40%"
#   This makes it nearly impossible for the LLM to know which date belongs to which exam.
#
# With Markdown, the structure is explicit and the LLM can reason about the relationships.

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
        
        # Optional: Find and display chunks that might contain table-like content
        # Note: PyPDFLoader extracts text but may not preserve table structure perfectly
        print(f"\nSearching for chunks that might contain table-like content...")
        table_chunks = []
        for i, chunk in enumerate(chunks):
            # Look for potential table indicators (though PyPDFLoader may not preserve structure)
            content = chunk.page_content
            # Check for patterns that might indicate tabular data
            lines = content.split("\n")
            # Look for lines with multiple spaces or tabs (common in table extraction)
            has_table_like_structure = any(len(line.split()) > 3 and "  " in line for line in lines[:10])
            if has_table_like_structure and len(content) > 100:
                table_chunks.append((i, chunk))
        
        if table_chunks:
            print(f"Found {len(table_chunks)} chunk(s) with potential table-like content:")
            # Display the first chunk found
            chunk_idx, table_chunk = table_chunks[0]
            print(f"\n{'=' * 60}")
            print(f"Example: Chunk {chunk_idx}")
            print(f"{'=' * 60}")
            print(f"Chunk length: {len(table_chunk.page_content)} characters")
            print(f"Metadata: {table_chunk.metadata}")
            print(f"\nContent preview:")
            print("-" * 60)
            print(table_chunk.page_content[:500])
            print("-" * 60)
        else:
            print("No chunks with obvious table-like structure found.")
        
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
        test_query = "What is this course about?"
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

