from retriever import get_raw_docs, get_best_chunks, get_build_index, get_query_engine
from cacher import cache
from llm_interaction import ask_llm


def filter_docs(query, documents, local=True):
    '''
        Returns a subset of the document set which are filtered to be
        of high quality (mostly factual and limited bias)
    '''
    return documents


@cache
def get_rag_context(query, documents, local=True):
    '''
        This splits the documents into chunks and identifies the
        most useful documents for the given query. 

        This takes the long list of input documents and finds key portions of
        those documents to serve as the RAG documents for the LLM
    '''
    # Get the Vector Index
    vector_index = get_build_index(documents=documents, local=local)
    # Create a query engine with the specified parameters
    query_engine = get_query_engine(sentence_index=vector_index, similarity_top_k=10, rerank_top_n=5)

    engine_response = query_engine.query(query)
    context_docs = engine_response.source_nodes
    context = "\n\n".join([doc.text.replace("\n", " ") for doc in context_docs])
    return context


def llm_prompt(query, rag_context):
    return f'''
        You are an intelligent assistant. Generate a detailed response for the following query based on your own knowledge and the context fetched. 
        Query: 
        {query}

        Context:
        {rag_context}

        Instructions:
        1. Be crisp and concise.
        2. Your response should be detailed and should cover every aspect of the context.
        3. Don't include anything else in your response - no header/footer/code etc
        '''


def full_pipeline(query):
    '''
        This is the main RAG pipeline
    '''
    local = False  # if false, this will use openai 
    documents = get_raw_docs(query)
    documents = filter_docs(query, documents, local=local)
    # select key chunks from docs to inform model
    rag_ctx = get_rag_context(query, documents, local=local)
    rag_prompt = llm_prompt(query, rag_ctx)
    llm_ans = ask_llm(prompt, local=local)
    return llm_ans


def main():
    query = "what does elon musk do"
    print(full_pipeline(query))

if __name__ == '__main__':
    main()

