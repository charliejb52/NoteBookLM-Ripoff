"""
Hybrid RAG Chat Script

This script implements a Hybrid RAG approach that combines:
1. Vector Search: Similarity search on Chroma vector store for relevant text chunks
2. Knowledge Graph Search: Cypher queries on Neo4j to find related entities and relationships
3. Combined Context: Merges both sources for comprehensive answers
"""

from langchain_community.vectorstores import Chroma
from langchain_community.graphs import Neo4jGraph
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
import os
from typing import List, Dict, Any



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


def load_neo4j_graph(uri=None, username=None, password=None, database="neo4j"):
    """
    Load Neo4j graph connection for knowledge graph queries.
    
    Args:
        uri (str): Neo4j connection URI
        username (str): Neo4j username
        password (str): Neo4j password
        database (str): Database name
        
    Returns:
        Neo4jGraph: Connected Neo4j graph instance
    """
    # Use environment variables or defaults (matching ingest_to_kg.py)
    uri = uri or os.getenv("NEO4J_URI", "neo4j+s://9efb56c8.databases.neo4j.io")
    username = username or os.getenv("NEO4J_USERNAME", "neo4j")
    password = password or os.getenv("NEO4J_PASSWORD", "hmKfxx2T2L7nsqLkWrsrXaGYORze6Gu9Js3s1fT_Ye8")
    
    try:
        graph = Neo4jGraph(
            url=uri,
            username=username,
            password=password,
            database=database
        )
        print(f"✓ Connected to Neo4j at {uri}")
        return graph
    except Exception as e:
        print(f"⚠ Warning: Could not connect to Neo4j: {e}")
        print("  Knowledge graph search will be disabled.")
        return None


def query_knowledge_graph(graph: Neo4jGraph, query: str, max_results: int = 10) -> str:
    """
    Query the Neo4j knowledge graph to find related entities and relationships.
    Improved version with better entity matching and relationship discovery.
    
    Args:
        graph (Neo4jGraph): Neo4j graph instance
        query (str): User's query string
        max_results (int): Maximum number of results to return
        
    Returns:
        str: Formatted string containing entity relationships
    """
    if graph is None:
        return ""
    
    # Extract key terms from the query for better matching
    # Remove common question words and focus on entities
    query_lower = query.lower()
    stop_words = {'where', 'is', 'what', 'who', 'when', 'how', 'the', 'a', 'an', 'are', 'was', 'were', 'does', 'do', 'did', 'can', 'could', 'will', 'would', 'should'}
    query_terms = [term for term in query_lower.split() 
                   if term not in stop_words and len(term) > 2]  # Only keep meaningful terms
    
    # If no meaningful terms after filtering, use the whole query
    if not query_terms:
        query_terms = [query_lower]
    
    # Build a more comprehensive Cypher query
    # This query searches for entities matching any query term and returns their relationships
    # Use UNION to combine both directions of relationships
    cypher_query = """
    // Find entities that match any query term in their properties
    MATCH (e)
    WHERE any(term IN $queryTerms WHERE 
        any(key IN keys(e) WHERE 
            e[key] IS NOT NULL AND 
            toLower(toString(e[key])) CONTAINS term
        )
    )
    WITH e, labels(e) as entity_labels
    LIMIT $limit
    
    // Get outgoing relationships
    MATCH (e)-[r]->(related)
    RETURN e, type(r) as rel_type, related, entity_labels, labels(related) as related_labels
    LIMIT $limit
    """
    
    try:
        # Try the main query first
        results = graph.query(
            cypher_query,
            params={"queryTerms": query_terms, "limit": max_results}
        )
        
        # Also try reverse relationships
        if not results or len(results) < max_results:
            reverse_query = """
            MATCH (e)
            WHERE any(term IN $queryTerms WHERE 
                any(key IN keys(e) WHERE 
                    e[key] IS NOT NULL AND 
                    toLower(toString(e[key])) CONTAINS term
                )
            )
            WITH e, labels(e) as entity_labels
            LIMIT $limit
            
            // Get incoming relationships
            MATCH (related)-[r]->(e)
            RETURN e, type(r) as rel_type, related, entity_labels, labels(related) as related_labels
            LIMIT $limit
            """
            reverse_results = graph.query(
                reverse_query,
                params={"queryTerms": query_terms, "limit": max_results}
            )
            if reverse_results:
                if results:
                    results.extend(reverse_results)
                else:
                    results = reverse_results
        
        if not results:
            # Fallback 1: Try searching for specific entity types mentioned in query
            # Look for Organization entities if query mentions company names
            if any(term in query_lower for term in ['apple', 'company', 'organization', 'corporation', 'inc', 'corp']):
                org_query = """
                MATCH (org:Organization)-[r]->(related)
                WHERE any(term IN $queryTerms WHERE 
                    any(key IN keys(org) WHERE 
                        org[key] IS NOT NULL AND 
                        toLower(toString(org[key])) CONTAINS term
                    )
                )
                RETURN org as e, type(r) as rel_type, related, labels(org) as entity_labels, labels(related) as related_labels
                LIMIT $limit
                """
                results = graph.query(
                    org_query,
                    params={"queryTerms": query_terms, "limit": max_results}
                )
            
            # Fallback 2: If still no results, try location-related queries
            if not results and any(term in query_lower for term in ['where', 'location', 'headquarters', 'headquartered', 'located', 'city', 'address', 'based']):
                location_query = """
                MATCH (e)-[r:LOCATED_IN]->(loc:Location)
                WHERE any(term IN $queryTerms WHERE 
                    any(key IN keys(e) WHERE 
                        e[key] IS NOT NULL AND 
                        toLower(toString(e[key])) CONTAINS term
                    )
                )
                RETURN e, type(r) as rel_type, loc as related, labels(e) as entity_labels, labels(loc) as related_labels
                LIMIT $limit
                """
                results = graph.query(
                    location_query,
                    params={"queryTerms": query_terms, "limit": max_results}
                )
                
                # Also try reverse direction (Location -> Organization)
                if not results:
                    location_query_reverse = """
                    MATCH (loc:Location)<-[r:LOCATED_IN]-(e)
                    WHERE any(term IN $queryTerms WHERE 
                        any(key IN keys(e) WHERE 
                            e[key] IS NOT NULL AND 
                            toLower(toString(e[key])) CONTAINS term
                        )
                    )
                    RETURN e, type(r) as rel_type, loc as related, labels(e) as entity_labels, labels(loc) as related_labels
                    LIMIT $limit
                    """
                    results = graph.query(
                        location_query_reverse,
                        params={"queryTerms": query_terms, "limit": max_results}
                    )
        
        if not results:
            return ""
        
        # Format the results as a readable string
        formatted_results = []
        seen_relationships = set()  # Avoid duplicates
        
        for record in results:
            entity = record.get('e', {})
            rel_type = record.get('rel_type', 'RELATED_TO')
            related = record.get('related', {})
            entity_labels = record.get('entity_labels', [])
            related_labels = record.get('related_labels', [])
            
            # Extract entity name/id - try multiple property names
            entity_name = (entity.get('name') or 
                          entity.get('id') or 
                          entity.get('title') or
                          next((entity.get(k) for k in entity.keys() if 'name' in k.lower() or 'id' in k.lower()), None) or
                          str(entity))
            
            related_name = (related.get('name') or 
                           related.get('id') or 
                           related.get('title') or
                           next((related.get(k) for k in related.keys() if 'name' in k.lower() or 'id' in k.lower()), None) or
                           str(related))
            
            entity_type = entity_labels[0] if entity_labels else 'Entity'
            related_type = related_labels[0] if related_labels else 'Entity'
            
            # Create a unique key for this relationship
            rel_key = f"{entity_type}:{entity_name}--{rel_type}-->{related_type}:{related_name}"
            if rel_key not in seen_relationships:
                seen_relationships.add(rel_key)
                formatted_results.append(
                    f"- {entity_type} '{entity_name}' --[{rel_type}]--> {related_type} '{related_name}'"
                )
        
        return "\n".join(formatted_results) if formatted_results else ""
        
    except Exception as e:
        print(f"  Warning: Error querying knowledge graph: {e}")
        import traceback
        traceback.print_exc()
        return ""


def hybrid_retrieve(vector_store, neo4j_graph, query: str, vector_k: int = 3, kg_max_results: int = 10, debug: bool = False) -> Dict[str, Any]:
    """
    Perform hybrid retrieval combining vector search and knowledge graph search.
    
    Args:
        vector_store (Chroma): Vector store for similarity search
        neo4j_graph (Neo4jGraph): Neo4j graph for entity relationships
        query (str): User query
        vector_k (int): Number of vector search results
        kg_max_results (int): Maximum KG results
        debug (bool): Print debug information
        
    Returns:
        dict: Dictionary containing 'vector_context' and 'kg_context'
    """
    # 1. Vector Search: Get relevant text chunks
    vector_results = vector_store.similarity_search(query, k=vector_k)
    vector_context = "\n\n".join([
        f"[Chunk {i+1}]: {doc.page_content}"
        for i, doc in enumerate(vector_results)
    ])
    
    if debug:
        print(f"  Vector search: Found {len(vector_results)} chunks")
    
    # 2. Knowledge Graph Search: Get related entities
    kg_context = query_knowledge_graph(neo4j_graph, query, max_results=kg_max_results)
    
    if debug:
        print(f"  KG search: Found {len(kg_context.split(chr(10))) if kg_context else 0} relationships")
        if kg_context:
            print(f"  KG results preview:\n{kg_context[:500]}...")
    
    return {
        "vector_context": vector_context,
        "kg_context": kg_context,
        "vector_docs": vector_results
    }


def create_hybrid_rag_chain(vector_store, neo4j_graph, llm_model="llama3"):
    """
    Create a Hybrid RAG chain that combines:
    - Vector search from Chroma (text chunks)
    - Knowledge graph search from Neo4j (entities and relationships)
    - Combined context for comprehensive answers
    
    Args:
        vector_store (Chroma): The vector store instance
        neo4j_graph (Neo4jGraph): Neo4j graph instance (can be None)
        llm_model (str): Name of the Ollama LLM model to use
        
    Returns:
        function: A hybrid RAG function that takes a query and returns an answer
    """
    # Create a base retriever from the vector store with reranking
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 20})
    
    # Create a reranker model to improve the relevance of the chunks
    model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    compressor = CrossEncoderReranker(model=model, top_n=3)
    
    vector_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
    
    # Initialize the LLM
    print(f"Initializing LLM model: {llm_model}...")
    llm = ChatOllama(model=llm_model, temperature=0)
    
    # Create the hybrid prompt template
    # This tells the LLM to use both vector search results and knowledge graph results
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that answers questions using two sources of context:

1. **Relevant Text Chunks [Vector Search]**: These are semantic similarity search results from the document corpus. They contain relevant textual information.

2. **Structured Entity Relationships [Knowledge Graph]**: These are entities and their relationships extracted from the documents. They show structured connections between concepts.

Use BOTH sources to provide a comprehensive, grounded answer. 
- If information appears in both sources, you can cross-reference them for accuracy.
- If information only appears in one source, use that source.
- If the context doesn't contain enough information to answer the question, say "I don't know" or "The provided context doesn't contain information about that."

Do not make up information or use knowledge outside of the provided context."""),
        ("human", """Context from Vector Search (Text Chunks):
{vector_context}

Context from Knowledge Graph (Entity Relationships):
{kg_context}

Question: {input}

Please provide a comprehensive answer using both sources of context.""")
    ])
    
    def hybrid_rag_invoke(query: str) -> Dict[str, Any]:
        """
        Invoke the hybrid RAG pipeline.
        
        Args:
            query (str): User query
            
        Returns:
            dict: Response with 'answer' and metadata
        """
        # Perform hybrid retrieval
        retrieval_results = hybrid_retrieve(
            vector_store, 
            neo4j_graph, 
            query, 
            vector_k=3, 
            kg_max_results=10
        )
        
        # Format the combined context
        vector_context = retrieval_results["vector_context"] or "No relevant text chunks found."
        kg_context = retrieval_results["kg_context"] or "No relevant entity relationships found in the knowledge graph."
        
        # Invoke the LLM with combined context
        messages = prompt.format_messages(
            vector_context=vector_context,
            kg_context=kg_context,
            input=query
        )
        
        response = llm.invoke(messages)
        
        return {
            "answer": response.content,
            "vector_context": vector_context,
            "kg_context": kg_context,
            "vector_docs": retrieval_results["vector_docs"]
        }
    
    return hybrid_rag_invoke


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
        
        # Load Neo4j knowledge graph
        print("=" * 60)
        print("Connecting to Neo4j Knowledge Graph")
        print("=" * 60)
        neo4j_graph = load_neo4j_graph()
        if neo4j_graph:
            print("✓ Knowledge graph connected!")
        else:
            print("⚠ Knowledge graph not available - will use vector search only")
        print()
        
        # Create the Hybrid RAG chain
        print("=" * 60)
        print("Initializing Hybrid RAG Chain")
        print("=" * 60)
        hybrid_rag = create_hybrid_rag_chain(vector_store, neo4j_graph, llm_model="llama3")
        print("✓ Hybrid RAG chain ready!")
        print()
        
        # Interactive chat loop
        print("=" * 60)
        print("Hybrid RAG Chat System")
        print("=" * 60)
        print("This system uses both vector search and knowledge graph search.")
        print("Ask questions about the documents.")
        print("Type 'quit', 'exit', or 'q' to end the conversation.")
        print("Type 'debug' to see the retrieved contexts.")
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
            
            # Debug mode
            if user_query.lower() == 'debug':
                print("\nDebug mode: Enter a query to see retrieved contexts")
                debug_query = input("Debug query: ").strip()
                if debug_query:
                    retrieval_results = hybrid_retrieve(
                        vector_store, 
                        neo4j_graph, 
                        debug_query, 
                        vector_k=3, 
                        kg_max_results=10
                    )
                    print("\n" + "=" * 60)
                    print("Vector Search Results:")
                    print("=" * 60)
                    print(retrieval_results["vector_context"] or "No results")
                    print("\n" + "=" * 60)
                    print("Knowledge Graph Results:")
                    print("=" * 60)
                    print(retrieval_results["kg_context"] or "No results")
                    print()
                continue
            
            try:
                # Invoke the hybrid RAG chain
                # This will:
                # 1. Perform vector similarity search on Chroma
                # 2. Query Neo4j knowledge graph for related entities
                # 3. Combine both contexts
                # 4. Send to LLM for answer generation
                print("\nThinking...")
                response = hybrid_rag(user_query)
                
                # Display the answer
                print(f"\nAssistant: {response['answer']}")
                print()
                
            except KeyboardInterrupt:
                print("\n\nInterrupted by user. Goodbye!")
                break
            except Exception as e:
                print(f"\nError processing query: {e}")
                import traceback
                traceback.print_exc()
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

