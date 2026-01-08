"""
Neo4j Knowledge Graph Ingestion Script for Apple 10-K Filings

This script:
1. Loads Apple 10-K HTML/text files using LangChain document loaders
2. Splits documents into chunks using RecursiveCharacterTextSplitter
3. Extracts entities and relationships using LLMGraphTransformer
4. Stores the knowledge graph in Neo4j
5. Ensures proper Document-Chunk relationships and unique chunk_ids
"""

import os
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from langchain_community.document_loaders import BSHTMLLoader, TextLoader, UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.graphs import Neo4jGraph
from langchain_ollama import ChatOllama
from download_apple_10k import find_latest_filing_file
import uuid

# Try to import LLMGraphTransformer - requires langchain-experimental package
try:
    from langchain_experimental.graph_transformers import LLMGraphTransformer
    LLM_GRAPH_TRANSFORMER_AVAILABLE = True
except ImportError:
    LLM_GRAPH_TRANSFORMER_AVAILABLE = False
    print("Warning: langchain-experimental not installed. Graph extraction will be skipped.")
    print("Install it with: pip install langchain-experimental")


def find_last_n_filing_files(directory="data/apple_filings", n=5):
    """
    Find the last N filing files in the directory, sorted by date.
    
    Args:
        directory (str): Directory containing the downloaded filings
        n (int): Number of files to return (default: 5)
        
    Returns:
        list: List of file paths, sorted by date (most recent first)
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    filing_files = []
    
    # Walk through the directory structure
    # sec-edgar-downloader typically organizes files as:
    # directory/sec-edgar-downloads/sec-edgar-downloads/AAPL/10-K/YYYY-MM-DD/filing.html
    # or simply: directory/AAPL/10-K/YYYY-MM-DD/filing.html
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Look for .htm, .html, or .txt files
            if file.endswith(('.htm', '.html', '.txt')):
                file_path = os.path.join(root, file)
                
                # Try to extract date from directory structure
                # This is more reliable than file modification time
                path_parts = Path(file_path).parts
                filing_date = None
                
                for part in path_parts:
                    # Check if this part looks like a date (YYYY-MM-DD format)
                    try:
                        filing_date = datetime.strptime(part, "%Y-%m-%d")
                        break
                    except ValueError:
                        continue
                
                # If we found a date in the path, use it for comparison
                # Otherwise, fall back to file modification time
                if filing_date:
                    filing_files.append((filing_date, file_path))
                else:
                    # Fallback: use file modification time if no date in path
                    file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                    filing_files.append((file_mtime, file_path))
    
    # Sort by date (most recent first) and return top N
    filing_files.sort(key=lambda x: x[0], reverse=True)
    return [file_path for _, file_path in filing_files[:n]]


def load_filing_file(file_path):
    """
    Load an HTML or text file using the appropriate LangChain document loader.
    
    Args:
        file_path (str): Path to the HTML or text file
        
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
            print(f"BSHTMLLoader failed, trying UnstructuredHTMLLoader: {e}")
            loader = UnstructuredHTMLLoader(file_path)
    elif file_ext == '.txt':
        loader = TextLoader(file_path, encoding='utf-8')
    else:
        raise ValueError(f"Unsupported file type: {file_ext}")
    
    documents = loader.load()
    return documents


def load_last_n_filing_files(directory="data/apple_filings", n=5):
    """
    Load the last N Apple 10-K filing files from the directory.
    
    Args:
        directory (str): Directory containing the downloaded filings
        n (int): Number of files to load (default: 5)
        
    Returns:
        list: List of Document objects from all loaded files
    """
    # Find the last N filing files
    filing_files = find_last_n_filing_files(directory, n)
    
    if not filing_files:
        raise FileNotFoundError(f"No filing files found in {directory}")
    
    print(f"Found {len(filing_files)} filing file(s) to load:")
    for i, file_path in enumerate(filing_files, 1):
        print(f"  {i}. {file_path}")
    
    # Load all files and combine documents
    all_documents = []
    for file_path in filing_files:
        try:
            print(f"\nLoading: {Path(file_path).name}...")
            documents = load_filing_file(file_path)
            # Add source file path to metadata for each document
            for doc in documents:
                doc.metadata['source_file'] = file_path
                doc.metadata['source_filename'] = Path(file_path).name
            all_documents.extend(documents)
            print(f"  ✓ Loaded {len(documents)} document(s)")
        except Exception as e:
            print(f"  ⚠ Error loading {file_path}: {e}")
            continue
    
    return all_documents


def split_documents_into_chunks(documents, chunk_size=2000, chunk_overlap=200):
    """
    Split documents into chunks using RecursiveCharacterTextSplitter.
    
    Args:
        documents (list): List of Document objects to split
        chunk_size (int): Maximum size of each chunk in characters (default: 2000)
        chunk_overlap (int): Number of characters to overlap between chunks (default: 200)
        
    Returns:
        list: List of Document chunks with unique chunk_ids added to metadata
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # Add unique chunk_id to each chunk's metadata
    for i, chunk in enumerate(chunks):
        if 'chunk_id' not in chunk.metadata:
            # Generate a unique chunk_id using UUID
            chunk.metadata['chunk_id'] = str(uuid.uuid4())
        # Also add chunk index for reference
        chunk.metadata['chunk_index'] = i
    
    return chunks


def create_llm_graph_transformer(llm_model="llama3"):
    """
    Create an LLMGraphTransformer to extract entities and relationships from text.
    
    Args:
        llm_model (str): Name of the Ollama LLM model to use
        
    Returns:
        LLMGraphTransformer: Configured graph transformer
        
    Raises:
        ImportError: If langchain-experimental is not installed
    """
    if not LLM_GRAPH_TRANSFORMER_AVAILABLE:
        raise ImportError(
            "langchain-experimental is required for graph extraction. "
            "Install it with: pip install langchain-experimental"
        )
    
    # Initialize the LLM
    llm = ChatOllama(model=llm_model, temperature=0)
    
    # Define the allowed entities and relationships
    # These are examples - you can customize based on your needs
    allowed_nodes = [
        "Organization",  # Companies, institutions
        "Product",       # Products, services
        "Risk",          # Risk factors, concerns
        "Metric",        # Financial metrics, KPIs
        "Person",        # Key personnel
        "Location",      # Geographic locations
        "Technology",    # Technologies, platforms
        "Regulation",    # Regulations, compliance
    ]
    
    allowed_relationships = [
        "REPORTED",           # Organization reported metric
        "COMPETES_WITH",      # Organization competes with another
        "AFFECTS",            # Entity affects another
        "LOCATED_IN",         # Entity located in location
        "USES",               # Entity uses technology/product
        "HAS_RISK",           # Organization has risk
        "REGULATED_BY",       # Entity regulated by regulation
        "MANAGED_BY",         # Entity managed by person
    ]
    
    # Create the graph transformer
    graph_transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=allowed_nodes,
        allowed_relationships=allowed_relationships,
        strict_mode=False  # Set to True for stricter extraction
    )
    
    return graph_transformer


def initialize_neo4j_graph(uri=None, username=None, password=None, database="neo4j"):
    """
    Initialize connection to Neo4j database.
    
    Args:
        uri (str): Neo4j connection URI (default: neo4j+s://6fef0d40.databases.neo4j.io for Aura cloud)
        username (str): Neo4j username (default: neo4j)
        password (str): Neo4j password (default: from environment or hardcoded)
        database (str): Database name (default: neo4j)
        
    Returns:
        Neo4jGraph: Connected Neo4j graph instance
        
    Raises:
        ConnectionError: If unable to connect to Neo4j with helpful troubleshooting info
    """
    # Use environment variables or defaults
    # Default to Neo4j Aura cloud instance
    uri = uri or os.getenv("NEO4J_URI", "neo4j+s://9efb56c8.databases.neo4j.io")
    username = username or os.getenv("NEO4J_USERNAME", "neo4j")
    password = password or os.getenv("NEO4J_PASSWORD", "hmKfxx2T2L7nsqLkWrsrXaGYORze6Gu9Js3s1fT_Ye8")
    
    print(f"Attempting to connect to Neo4j at {uri}...")
    
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
        error_msg = str(e)
        print(f"\n❌ Error connecting to Neo4j: {error_msg}\n")
        
        # Provide helpful troubleshooting information
        if "Connection refused" in error_msg or "Could not connect" in error_msg:
            print("=" * 60)
            print("Troubleshooting Neo4j Connection")
            print("=" * 60)
            print("\nThe Neo4j database is not accessible. Please check:")
            print("\n1. Is Neo4j running?")
            print("   - Local installation: Check if Neo4j service is running")
            print("   - Docker: Run 'docker ps' to check if Neo4j container is running")
            print("   - Neo4j Desktop: Ensure the database is started")
            print("\n2. Is the connection URI correct?")
            print(f"   - Current URI: {uri}")
            print("   - For local: bolt://localhost:7687")
            print("   - For Neo4j Aura: neo4j+s://<instance-id>.databases.neo4j.io")
            print("   - Current default: neo4j+s://6fef0d40.databases.neo4j.io")
            print("\n3. Are the credentials correct?")
            print(f"   - Username: {username}")
            print("   - Password: [hidden]")
            print("\n4. To use a different Neo4j instance, set environment variables:")
            print("   export NEO4J_URI='bolt://your-host:7687'")
            print("   export NEO4J_USERNAME='your-username'")
            print("   export NEO4J_PASSWORD='your-password'")
            print("\n5. Or pass them as arguments to the ingestion function")
            print("=" * 60)
        
        raise ConnectionError(f"Could not connect to Neo4j at {uri}. See troubleshooting info above.") from e


def create_document_node(graph, document_path, document_metadata=None):
    """
    Create a Document node in Neo4j for the source file.
    
    Args:
        graph (Neo4jGraph): Neo4j graph instance
        document_path (str): Path to the source document
        document_metadata (dict): Additional metadata for the document
        
    Returns:
        str: Document ID
    """
    # Generate a unique document ID
    document_id = str(uuid.uuid4())
    document_name = Path(document_path).name
    
    # Create Document node
    query = """
    CREATE (d:Document {
        document_id: $document_id,
        name: $document_name,
        path: $document_path,
        created_at: datetime()
    })
    RETURN d.document_id as document_id
    """
    
    # Add any additional metadata
    if document_metadata:
        for key, value in document_metadata.items():
            query = query.replace("created_at: datetime()", f"created_at: datetime(), {key}: ${key}")
    
    result = graph.query(
        query,
        params={
            "document_id": document_id,
            "document_name": document_name,
            "document_path": document_path,
            **(document_metadata or {})
        }
    )
    
    return document_id


def create_chunk_nodes_and_relationships(graph, document_id, chunks):
    """
    Create Chunk nodes and PART_OF relationships to Document node.
    
    Args:
        graph (Neo4jGraph): Neo4j graph instance
        document_id (str): ID of the parent Document node
        chunks (list): List of Document chunks with chunk_id in metadata
    """
    # Create Chunk nodes and PART_OF relationships in batch
    query = """
    UNWIND $chunks AS chunk_data
    MERGE (c:Chunk {chunk_id: chunk_data.chunk_id})
    ON CREATE SET 
        c.content = chunk_data.content,
        c.chunk_index = chunk_data.chunk_index,
        c.created_at = datetime()
    WITH c, chunk_data
    MATCH (d:Document {document_id: $document_id})
    MERGE (c)-[:PART_OF]->(d)
    RETURN count(c) as chunks_created
    """
    
    chunks_data = []
    for chunk in chunks:
        chunks_data.append({
            "chunk_id": chunk.metadata.get('chunk_id'),
            "content": chunk.page_content[:1000],  # Store first 1000 chars (adjust as needed)
            "chunk_index": chunk.metadata.get('chunk_index', 0)
        })
    
    result = graph.query(query, params={"document_id": document_id, "chunks": chunks_data})
    return result[0]['chunks_created'] if result else 0


def extract_and_store_graph(graph, chunks, graph_transformer, max_chunks=None):
    """
    Extract entities and relationships from chunks and store in Neo4j.
    
    Args:
        graph (Neo4jGraph): Neo4j graph instance
        chunks (list): List of Document chunks
        graph_transformer (LLMGraphTransformer): Graph transformer for extraction
        max_chunks (int): Maximum number of chunks to process (None for all, default: 100)
    """
    # Limit chunks if max_chunks is specified
    total_chunks_available = len(chunks)
    if max_chunks is not None and max_chunks > 0:
        chunks = chunks[:max_chunks]
        if total_chunks_available > max_chunks:
            print(f"\nExtracting entities and relationships from first {len(chunks)} chunks (limited from {total_chunks_available} total)...")
        else:
            print(f"\nExtracting entities and relationships from {len(chunks)} chunks...")
    else:
        print(f"\nExtracting entities and relationships from {len(chunks)} chunks...")
    
    # Process chunks in batches to avoid overwhelming the LLM
    batch_size = 5
    total_processed = 0
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1} ({len(batch)} chunks)...")
        
        for chunk in batch:
            try:
                # Extract graph documents from the chunk
                graph_documents = graph_transformer.convert_to_graph_documents([chunk])
                
                # Add the graph documents to Neo4j
                # The method name may vary by LangChain version
                if hasattr(graph, 'add_graph_documents'):
                    graph.add_graph_documents(graph_documents)
                elif hasattr(graph, 'add_documents'):
                    # Alternative method name
                    graph.add_documents(graph_documents)
                else:
                    # Fallback: use the query method directly
                    for gdoc in graph_documents:
                        # Extract nodes and relationships from graph document
                        for node in gdoc.nodes:
                            # Create node
                            node_props = {k: v for k, v in node.properties.items() if v is not None}
                            node_query = f"MERGE (n:{node.id} {{id: $id}})"
                            if node_props:
                                set_clause = ", ".join([f"n.{k} = ${k}" for k in node_props.keys()])
                                node_query += f" ON CREATE SET {set_clause}"
                            graph.query(node_query, params={"id": node.id, **node_props})
                        
                        # Create relationships
                        for rel in gdoc.relationships:
                            rel_query = f"""
                            MATCH (source {{id: $source_id}}), (target {{id: $target_id}})
                            MERGE (source)-[r:{rel.type}]->(target)
                            """
                            graph.query(rel_query, params={
                                "source_id": rel.source.id,
                                "target_id": rel.target.id
                            })
                
                total_processed += 1
                
            except Exception as e:
                print(f"  Warning: Error processing chunk {chunk.metadata.get('chunk_id', 'unknown')}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print(f"✓ Processed {total_processed} chunks")
    return total_processed


def debug_check_chunk_document_relationships(graph):
    """
    Debug check: Verify that all Chunk nodes are linked to Document nodes via PART_OF.
    
    Args:
        graph (Neo4jGraph): Neo4j graph instance
    """
    print("\n" + "=" * 60)
    print("Debug Check: Chunk-Document Relationships")
    print("=" * 60)
    
    # Check for chunks without PART_OF relationships
    query = """
    MATCH (c:Chunk)
    WHERE NOT (c)-[:PART_OF]->(:Document)
    RETURN count(c) as orphaned_chunks
    """
    result = graph.query(query)
    orphaned_count = result[0]['orphaned_chunks'] if result else 0
    
    if orphaned_count > 0:
        print(f"⚠ Warning: {orphaned_count} Chunk nodes are not linked to any Document node")
    else:
        print("✓ All Chunk nodes are linked to Document nodes via PART_OF")
    
    # Check total chunks and relationships
    query = """
    MATCH (c:Chunk)-[r:PART_OF]->(d:Document)
    RETURN count(c) as total_chunks, count(r) as total_relationships, count(DISTINCT d) as total_documents
    """
    result = graph.query(query)
    if result:
        stats = result[0]
        print(f"  Total Chunks: {stats['total_chunks']}")
        print(f"  Total PART_OF relationships: {stats['total_relationships']}")
        print(f"  Total Documents: {stats['total_documents']}")


def debug_check_unique_chunk_ids(graph):
    """
    Debug check: Verify that all chunk_ids are unique.
    
    Args:
        graph (Neo4jGraph): Neo4j graph instance
    """
    print("\n" + "=" * 60)
    print("Debug Check: Unique chunk_id")
    print("=" * 60)
    
    # Check for duplicate chunk_ids
    query = """
    MATCH (c:Chunk)
    WITH c.chunk_id as chunk_id, count(*) as count
    WHERE count > 1
    RETURN chunk_id, count
    """
    result = graph.query(query)
    
    if result:
        print(f"⚠ Warning: Found {len(result)} duplicate chunk_ids:")
        for row in result:
            print(f"  chunk_id: {row['chunk_id']} appears {row['count']} times")
    else:
        print("✓ All chunk_ids are unique")
    
    # Get total chunk count
    query = "MATCH (c:Chunk) RETURN count(c) as total_chunks"
    result = graph.query(query)
    if result:
        print(f"  Total Chunks: {result[0]['total_chunks']}")


def ensure_constraints(graph):
    """
    Ensure Neo4j constraints for data integrity.
    
    Args:
        graph (Neo4jGraph): Neo4j graph instance
    """
    print("\n" + "=" * 60)
    print("Setting up Constraints")
    print("=" * 60)
    
    # Create unique constraint on chunk_id
    # Note: Neo4j syntax may vary by version
    try:
        # Try modern syntax first (Neo4j 4.0+)
        query = """
        CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS
        FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE
        """
        graph.query(query)
        print("✓ Created unique constraint on Chunk.chunk_id")
    except Exception as e:
        try:
            # Try older syntax (Neo4j 3.x)
            query = """
            CREATE CONSTRAINT ON (c:Chunk)
            ASSERT c.chunk_id IS UNIQUE
            """
            graph.query(query)
            print("✓ Created unique constraint on Chunk.chunk_id")
        except Exception as e2:
            # Constraint might already exist
            print(f"  Note: Constraint may already exist or syntax error: {e2}")
    
    # Create unique constraint on document_id
    try:
        query = """
        CREATE CONSTRAINT document_id_unique IF NOT EXISTS
        FOR (d:Document) REQUIRE d.document_id IS UNIQUE
        """
        graph.query(query)
        print("✓ Created unique constraint on Document.document_id")
    except Exception as e:
        try:
            query = """
            CREATE CONSTRAINT ON (d:Document)
            ASSERT d.document_id IS UNIQUE
            """
            graph.query(query)
            print("✓ Created unique constraint on Document.document_id")
        except Exception as e2:
            print(f"  Note: Constraint may already exist or syntax error: {e2}")


def ingest_filing_to_neo4j(
    filing_directory="data/apple_filings",
    num_filings=5,
    neo4j_uri=None,
    neo4j_username=None,
    neo4j_password=None,
    llm_model="llama3",
    chunk_size=2000,
    chunk_overlap=200,
    max_chunks_for_extraction=100
):
    """
    Main function to ingest the last N Apple 10-K filing files into Neo4j knowledge graph.
    
    Args:
        filing_directory (str): Directory containing the filing files
        num_filings (int): Number of most recent filings to load (default: 5)
        neo4j_uri (str): Neo4j connection URI
        neo4j_username (str): Neo4j username
        neo4j_password (str): Neo4j password
        llm_model (str): Ollama LLM model name
        chunk_size (int): Chunk size for text splitting
        chunk_overlap (int): Chunk overlap for text splitting
        max_chunks_for_extraction (int): Maximum number of chunks to process for graph extraction (default: 100, None for all)
    """
    print("=" * 60)
    print("Neo4j Knowledge Graph Ingestion")
    print("=" * 60)
    
    # Step 1: Load the last N filing files
    print(f"\nStep 1: Loading last {num_filings} filing files...")
    print(f"  Directory: {filing_directory}")
    all_documents = load_last_n_filing_files(filing_directory, n=num_filings)
    print(f"✓ Loaded {len(all_documents)} total document(s) from {num_filings} file(s)")
    
    # Step 2: Initialize Neo4j connection
    print(f"\nStep 2: Connecting to Neo4j...")
    graph = initialize_neo4j_graph(neo4j_uri, neo4j_username, neo4j_password)
    
    # Step 3: Set up constraints
    ensure_constraints(graph)
    
    # Step 4: Group documents by source file and process each file
    print(f"\nStep 3: Processing files and creating Document nodes...")
    documents_by_file = defaultdict(list)
    for doc in all_documents:
        source_file = doc.metadata.get('source_file', 'unknown')
        documents_by_file[source_file].append(doc)
    
    all_chunks = []
    document_ids = []
    
    for file_path, file_documents in documents_by_file.items():
        print(f"\n  Processing: {Path(file_path).name}")
        
        # Create Document node for this file
        document_id = create_document_node(graph, file_path)
        document_ids.append(document_id)
        print(f"    ✓ Created Document node: {document_id}")
        
        # Split this file's documents into chunks
        file_chunks = split_documents_into_chunks(file_documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        print(f"    ✓ Created {len(file_chunks)} chunks")
        
        # Create Chunk nodes and PART_OF relationships for this file
        chunks_created = create_chunk_nodes_and_relationships(graph, document_id, file_chunks)
        print(f"    ✓ Created {chunks_created} Chunk nodes with PART_OF relationships")
        
        all_chunks.extend(file_chunks)
    
    print(f"\n✓ Processed {len(documents_by_file)} file(s), created {len(all_chunks)} total chunks")
    
    # Step 5: Extract entities and relationships using LLM
    print(f"\nStep 4: Extracting entities and relationships...")
    if max_chunks_for_extraction:
        print(f"  Note: Processing only first {max_chunks_for_extraction} chunks for graph extraction (out of {len(all_chunks)} total)")
    if LLM_GRAPH_TRANSFORMER_AVAILABLE:
        try:
            graph_transformer = create_llm_graph_transformer(llm_model)
            extract_and_store_graph(graph, all_chunks, graph_transformer, max_chunks=max_chunks_for_extraction)
        except Exception as e:
            print(f"⚠ Warning: Could not extract graph entities/relationships: {e}")
            print("  Continuing without graph extraction...")
    else:
        print("⚠ Skipping graph extraction (langchain-experimental not installed)")
        print("  Install with: pip install langchain-experimental")
    
    # Step 6: Debug checks
    debug_check_chunk_document_relationships(graph)
    debug_check_unique_chunk_ids(graph)
    
    print("\n" + "=" * 60)
    print("Ingestion Complete!")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    
    # Default: load last 5 filings from data/apple_filings
    filing_directory = "data/apple_filings"
    num_filings = 5
    
    # Allow override via command line arguments
    if len(sys.argv) > 1:
        # First arg can be directory or number of filings
        if os.path.isdir(sys.argv[1]):
            filing_directory = sys.argv[1]
            if len(sys.argv) > 2:
                num_filings = int(sys.argv[2])
        else:
            num_filings = int(sys.argv[1])
    
    if not os.path.exists(filing_directory):
        print(f"Error: Directory not found: {filing_directory}")
        sys.exit(1)
    
    try:
        ingest_filing_to_neo4j(
            filing_directory=filing_directory,
            num_filings=num_filings,
            chunk_size=2000,
            chunk_overlap=200,
            llm_model="llama3"
        )
    except Exception as e:
        print(f"\nError during ingestion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

