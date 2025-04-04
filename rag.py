from retriever import get_raw_docs, get_best_chunks, get_build_index, get_query_engine
from cacher import cache
from llm_interaction import ask_llm
import filters

def filter_docs(query, documents, filter_method, local=True):
    '''
        Returns a subset of the document set which are filtered to be
        of high quality (mostly factual and limited bias)
    '''
    # currently, query param is not used for anything, but it *might* come
    # in handy for some filter methods later? 

    documents = [doc.text.replace("\n", " ") for doc in documents if filter_method(query, doc, local=local)]
    return "\n\n".join(documents)


@cache
def get_key_documents(query, documents, local=True):
    '''
        This splits the documents into chunks and identifies the
        most useful documents for the given query. 

        This takes the long list of input documents and finds key portions of
        those documents to serve as the RAG documents for the LLM
    '''
    if len(documents) == 0:
        return "No external context"  # edge case

    # Get the Vector Index and Query Engine
    vector_index = get_build_index(documents=documents, local=local)
    query_engine = get_query_engine(sentence_index=vector_index, similarity_top_k=10, rerank_top_n=5)
    try:
        engine_response = query_engine.query(query)
        context_docs = engine_response.source_nodes
        #print(context_docs)
        #print("\n\n".join([doc.text.replace("\n", " ") for doc in context_docs]))
        return context_docs
        # \n\n is doc separator since no documents contain \n in them
        # \n is removed during fetch, and removed again now just in case
    except Exception as e:
        print(f"Error getting chunks: {e}")
        return ""


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


def full_pipeline(query, filter_method):
    '''
        This is the main RAG pipeline
    '''
    local = True  # if false, this will use openai

    # initial docs from google
    documents = get_raw_docs(query)
    # select key portions of documents
    key_docs = get_key_documents(query, documents, local=local)
    # filter misinformation
    key_docs = filter_docs(query, key_docs, filter_method, local=local)
    # final prompt to llm
    rag_prompt = llm_prompt(query, key_docs)
    llm_ans = ask_llm(query, rag_prompt, local=local)
    return llm_ans


def main():
    # main hyperparameters that define how this rag query will run
    query = "what does elon musk do"
    filter_method = filters.llm_trust
    ans = full_pipeline(query, filter_method)
    print("Answer:\n\n" + ans)

if __name__ == '__main__':
    main()

