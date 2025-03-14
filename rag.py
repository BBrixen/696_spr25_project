import openai
from retriever import get_raw_docs, get_best_chunks
# handling apis
from api_keys import openai_api_key
openai.api_key = openai_api_key


def filter_docs(query, documents):
    '''
        Returns a subset of the document set which are filtered to be
        of high quality (mostly factual and limited bias)
    '''
    # TODO this is where we will try out different methods of filtering documents
    return documents


def get_rag_context(query, documents):
    '''
        This splits the documents into chunks and identifies the
        most useful documents for the given query. 

        This takes the long list of input documents and finds key portions of
        those documents to serve as the RAG documents for the LLM
    '''
    # # Get the Vector Index
    # vector_index = get_build_index(documents=documents, embed_model="local:BAAI/bge-small-en-v1.5", save_dir="./vector_store/index")
    # # Create a query engine with the specified parameters
    # query_engine = get_query_engine(sentence_index=vector_index, similarity_top_k=10, rerank_top_n=5)

    # context_docs = query_engine.query(query)

    combined_text = "\n\n".join([doc.text for doc in documents])
    best_chunks = get_best_chunks(combined_text, query)

    context = "\n\n".join(best_chunks)
    return context


def get_llm_response(rag_context, query):
    '''
        This will query the LLM to give the answer using the context. 
        This is the last step of the pipeline

        Followed this link for help with using chat:
        https://www.geeksforgeeks.org/how-to-use-chatgpt-api-in-python/
    '''
    # TODO if this doesnt work, combine it all into the user input
    messages = [ 
            {
                "role": "system", 
                "content": f'''
                You are an intelligent assistant. Generate a detailed response for the following query asked by the user based on your own knowledge and the context fetched. 
                Context:
                {rag_context}

                Instructions:
                1. Be crisp and concise.
                2. Your response should be detailed and should cover every aspect of the context.
                3. Don't include anything else in your response - no header/footer/code etc
                '''
            },
            {
                "role": "user",
                "content": query
            }
        ]

    chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages)

    reply = chat.choices[0].message.content
    # TODO  make the messages grow with each reponse? 
    return reply


def full_pipeline(query):
    '''
        This is the main RAG pipeline
    '''
    documents = get_raw_docs(query)             # collect large set of documents from web search
    documents = filter_docs(query, documents)   # filter out bad documents

    rag_ctx = get_rag_context(query, documents) # select key chunks from docs to inform model
    llm_ans = get_llm_response(rag_ctx, query)  # get llm response using rag documents
    return llm_ans


def main():
    query = "what is retrieval augmented generation"
    print(full_pipeline(query))

if __name__ == '__main__':
    main()

