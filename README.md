# NoteBookLM-Ripoff

Personal learning project to learn about LLM workflows/RAG pipelines

1. Document Ingestion:

The ingest.py file handles the proper loading of pdf files as document objects of each page of a given pdf. Each document object contains the test written on the page as well as important metadata like the page number and author etc.

It is important that we have a standardized document object for once we get to the embedding/training the llm steps of the project so we need a flexible ingestion pipeline that can handle all different formats of documents. I was advised to use PyPDFLoader to easily achieve this.

Early on I ran into some challenges ingesting my course syllabus because it was very table heavy, and as I soon came to understand tables are kind of the final boss of ingestion. Rather than adressing this now I have chosen to ingest my course textbook instead because it is rich with hundreds of pages of plain text and will therefore be an easier starting point.

2. Document Chunking:

Why do we need this...

there are a couple of reasons: - Whole documents may exceed the token limit of the LLM - Using the whole document as a reference is likely to give less precise responses because it will lead to lots of unrelated info being considered

to deal with both of these, we will chunk the document into smaller subsections by paragraph (or if that exceeds the limit sentences, words etc.) with a 100 char overlap which will lead to smaller chunks that have more semantic relation between their tokens.

3. Vector embedding/ vector database:

Now how will the LLM know which chunks are semantically similar to the prompt they are asked...

We need to: - embedd the chunks as vectors: Ollama embedding_model() function - store them somewhere: ChromaDB\*

\*Was advised to use a vector database which is much less clunky dealing with high dimensional data like this than a traditional SQL database.

Right now ingest.py loads the pdf as a document object, chunks it appropriatly, and embedds each chunk as a vector into the chroma_db folder.

4. Retrieval:

The current iteration of retrieval has two steps:

    1. base retrieval that returns the 20 most relevant chunks based on vector search (cosine similarity) and keyword search. This can be done very quickly and efficiently but will not result in very good responses from llama3.

    2. reranker that returns the top 3 most relevant chunks based on vector search but uses a different type of encoding called "cross encoding" which is more robust but much more computationally expensive.
