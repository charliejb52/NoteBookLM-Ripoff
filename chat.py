"""
Chat Script for Local RAG System

This script completes the RAG (Retrieval-Augmented Generation) loop by:
1. Loading the vector database
2. Retrieving relevant chunks for user queries
3. Using an LLM to generate answers based on the retrieved context
"""

from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
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


def create_rag_chain(vector_store, llm_model="llama3"):
    """
    Create a RAG (Retrieval-Augmented Generation) chain that combines:
    - The vector store retriever
    - A prompt template
    - The LLM
    
    Args:
        vector_store (Chroma): The vector store instance
        llm_model (str): Name of the Ollama LLM model to use
        
    Returns:
        Chain: A RAG chain that can answer questions using retrieved context
    """
    # Create a retriever from the vector store
    # The retriever will fetch the most relevant chunks for any query
    retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    
    # Initialize the LLM
    # ChatOllama uses a local Ollama instance to run the LLM
    print(f"Initializing LLM model: {llm_model}...")
    llm = ChatOllama(model=llm_model, temperature=0)
    
    # Create the prompt template
    # This tells the LLM to only use the provided context and admit when it doesn't know
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that answers questions based only on the provided context.
        
Use only the information from the context to answer the question. 
If the context doesn't contain enough information to answer the question, say "I don't know" or "The provided context doesn't contain information about that." But make sure to check the entire document for information that may be relevant to the question.

Do not make up information or use knowledge outside of the provided context."""),
        ("human", "Context:\n{context}\n\nQuestion: {input}")
    ])
    
    # Create the document chain that processes the retrieved documents
    # This chain formats the retrieved chunks and passes them to the LLM
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # Create the retrieval chain that combines retrieval + document processing
    # This is the complete RAG pipeline: query → retrieve → format → LLM → answer
    rag_chain = create_retrieval_chain(retriever, document_chain)
    
    return rag_chain


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
        print("✓ Vector store loaded successfully!")
        print()
        
        # Create the RAG chain
        print("=" * 60)
        print("Initializing RAG Chain")
        print("=" * 60)
        rag_chain = create_rag_chain(vector_store, llm_model="llama3")
        print("✓ RAG chain ready!")
        print()
        
        # Interactive chat loop
        print("=" * 60)
        print("RAG Chat System")
        print("=" * 60)
        print("Ask questions about the course syllabus.")
        print("Type 'quit', 'exit', or 'q' to end the conversation.")
        print()
        
        while True:
            # Get user input
            user_query = input("You: ").strip()
            
            # Check for exit commands
            if user_query.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            # Skip empty queries
            if not user_query:
                continue
            
            try:
                # Invoke the RAG chain
                # This will:
                # 1. Convert the query to an embedding
                # 2. Retrieve the most relevant chunks from the vector store
                # 3. Format the chunks with the prompt
                # 4. Send to the LLM for answer generation
                print("\nThinking...")
                response = rag_chain.invoke({"input": user_query})
                
                # Display the answer
                print(f"\nAssistant: {response['answer']}")
                print()
                
            except KeyboardInterrupt:
                print("\n\nInterrupted by user. Goodbye!")
                break
            except Exception as e:
                print(f"\nError processing query: {e}")
                print("Please try again or type 'quit' to exit.\n")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nTo fix this:")
        print("  1. Make sure you've run ingest.py first to create the vector database")
        print("  2. Check that the 'chroma_db' folder exists in the current directory")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

