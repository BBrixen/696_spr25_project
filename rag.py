from openai import OpenAI
from retriever import get_raw_docs, get_best_chunks, get_build_index, get_query_engine
from cacher import cache

# handling apis
from api_keys import openai_api_key
client = OpenAI(api_key=openai_api_key)


@cache
def filter_docs(query, documents):
    '''
        Returns a subset of the document set which are filtered to be
        of high quality (mostly factual and limited bias)
    '''
    # TODO this is where we will try out different methods of filtering documents
    # TODO to use the cache, this needs to return a string
    return documents

@cache
def get_rag_context(query, documents):
    '''
        This splits the documents into chunks and identifies the
        most useful documents for the given query. 

        This takes the long list of input documents and finds key portions of
        those documents to serve as the RAG documents for the LLM
    '''
    # Get the Vector Index
    vector_index = get_build_index(documents=documents, embed_model="local:BAAI/bge-small-en-v1.5", save_dir="./vector_store/index")
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


@cache
def get_llm_response(query, rag_context):
    '''
        This will query the LLM to give the answer using the context. 
        This is the last step of the pipeline

        Followed this link for help with using chat:
        https://www.geeksforgeeks.org/how-to-use-chatgpt-api-in-python/
    '''
    messages = [ 
            {
                "role": "user", 
                "content": llm_prompt(query, rag_context)
            }
        ]

    chat = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    reply = chat.choices[0].message.content
    return reply


def full_pipeline(query):
    '''
        This is the main RAG pipeline
    '''
    documents = get_raw_docs(query)             # collect large set of documents from web search
    documents = filter_docs(query, documents)   # filter out bad documents

    rag_ctx = get_rag_context(query, documents) # select key chunks from docs to inform model
    llm_ans = get_llm_response(query, rag_ctx)  # get llm response using rag documents
    return llm_ans


def main():
    query = "what does elon musk do"
    print(full_pipeline(query))

if __name__ == '__main__':
    main()

