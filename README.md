# NoteBookLM-Ripoff

Personal learning project to learn about LLM workflows/RAG pipelines

1. Document Ingestion:

The ingest.py file handles the proper loading of pdf files as document objects of each page of a given pdf. Each document object contains the test written on the page as well as important metadata like the page number and author etc.

It is important that we have a standardized document object for once we get to the embedding/training the llm steps of the project so we need a flexible ingestion pipeline that can handle all different formats of documents. I was advised to use PyPDFLoader to easily achieve this.

Early on I ran into some challenges ingesting my course syllabus because it was very table heavy, and as I soon came to understand tables are kind of the final boss of ingestion. Rather than adressing this now I have chosen to instead ingest my course textbook instead because which is rich with hundreds of pages of plain text and will therefore be an easier starting point.

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

Now we need to create the mechanism for retrieving the most semantically similar chunks to the prompt to eventually hand off to the LLM for context in its response. This is fairly simple because we have all of the chunk arrays stored in a folder as a vector database making it easier to deal with.

This whole step is made possible by the invention of vector search. Traditional search engines would take strings in the prompt and look for chunks with the exact same one, but this misses many others that may have words of similar meaning. Vector search solves this problem by grading similarity between chunks based on their semantic meaning instead.

Words like puppy and dog sit very close together in "latent space" the imaginary vector space of all human knowledge. How does a vector search know that these two words are similar...

Three possible ways:

- Cosine similarity: find the cos of the angle between the vectors 1 means they are exactly the same. This works well with NLP because the sentence "I love dogs" and a whole paragraph about a passion for dogs are semantically similar, but will have very different magnitudes.
- Euclidian distance(L2 norm): find the straight line distance between the two vectors, this is not as good for NLP because two vectors could have very similar direction(semantic meaning) but be very far apart if they have different magnitudes. This is better for image/sensor stuff where intensity matters too
- Dot product: kind of a combination of the two, takes angle between and magnitude into account. Works well for reccomendation systems. Think about someone looking for a movie that has a high intensity preference for action, a movie that is purely action will yield a high dot product.

The current iteration uses vector_store.as_retriever() which uses cosine similarity.

I am running into a problem with the retrieval: the chunk detailing the final percentages corresponding to certain letter grades is consistantly not found by the retriever when I ask it about the "grading scale" which is the exact title of the section.

I know that the chunk containing the whole section exists because of the debug_chunks file so it must be something else...

I think it is the fact that basically 95% of the section is all numbers and letter grades, which although are extremely relevant to the "grading scale", do not have any true semantic meaning. Thats why the section doesn't appear as very similar to the prompt even though it is actually the most relevant of any of the chunks.

To deal with this I think I will try to implement some sort of hybrid vector search/keyowrd search method that is able to combat this issue. Not sure how that will go quite yet.
