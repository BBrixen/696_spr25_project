
import numpy as np
import llama_index
from llama_index.embeddings import HuggingFaceEmbedding
import google.generativeai as palm
from llama_index.llms.palm import PaLM
import math
import openai
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex, load_index_from_storage
from llama_index.core import Settings
from llama_index.core import StorageContext
from llama_index.core.postprocessor import MetadataReplacementPostProcessor, SentenceTransformerRerank
# handling apis
from api_keys import palm_api_key, openai_api_key
openai.api_key = openai_api_key
# palm.configure(palm_api_key)

'''
following tutorial from here:
https://medium.com/@mahakal001/building-a-rag-pipeline-step-by-step-0a5e1ac68562
'''

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def chunk_text(text, chunk_size):
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

def get_embedding_model():
    return HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

def dot_product(vec1, vec2):
    return sum(a*b for a,b in zip(vec1, vec2))

def magnitude(vec):
    return math.sqrt(sum(v**2 for v in vec))

def cosine_similarity(vec1, vec2):
    dot  = dot_product(vec1, vec2)
    mag1 = magnitude(vec1)
    mag2 = magnitude(vec2)

    if mag1 == 0 or mag2 == 0:
        return 0
    return dot / (mag1*mag2)

'''
now following this tutorial:
https://www.freecodecamp.org/news/how-to-build-a-rag-pipeline-with-llamaindex/
'''

def get_build_index(documents,
        embed_model="local:BAAI/bge-small-en-v1.5", 
        save_dir="./vector_store/index"):
    """
    Builds or loads a vector store index from the given documents.

    Args:
        documents (list[Document]): A list of Document objects.
        embed_model (str, optional): The embedding model to use. Defaults to "local:BAAI/bge-small-en-v1.5".
        save_dir (str, optional): The directory to save or load the index from. Defaults to "./vector_store/index".

    Returns:
        VectorStoreIndex: The built or loaded index.
    """

    # Set index settings
    Settings.llm = watsonx_llm
    Settings.embed_model = embed_model
    Settings.node_parser = SentenceSplitter(chunk_size=1000, chunk_overlap=200)
    Settings.num_output = 512
    Settings.context_window = 3900

    # Check if the save directory exists
    if not os.path.exists(save_dir):
        # Create and load the index
        index = VectorStoreIndex.from_documents(
            [documents], service_context=Settings
        )
        index.storage_context.persist(persist_dir=save_dir)
    else:
        # Load the existing index
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir),
            service_context=Settings,
        )
    return index

def get_query_engine(sentence_index, similarity_top_k=10, rerank_top_n=5):
    """
    Creates a query engine with metadata replacement and sentence transformer reranking.

    Args:
        sentence_index (VectorStoreIndex): The sentence index to use.
        similarity_top_k (int, optional): The number of similar nodes to consider. Defaults to 6.
        rerank_top_n (int, optional): The number of nodes to rerank. Defaults to 2.

    Returns:
        QueryEngine: The query engine.
    """

    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="BAAI/bge-reranker-base"
    )
    engine = sentence_index.as_query_engine(
        similarity_top_k=similarity_top_k, node_postprocessors=[postproc, rerank]
    )
    return engine



'''
using chatgpt in python:
https://www.geeksforgeeks.org/how-to-use-chatgpt-api-in-python/
'''

def get_rag_context(query, query_engine):
    context_docs = query_engine.query(query)
    # this is where we can filter out documents
    print(context_docs)
    print(type(context_docs))
    return context_docs

def get_response(rag_context, query):
    # TODO if this doesnt work, combine it all into the user input
    messages = [ 
            {
                "role": "system", 
                "content": f'''
                You are an intelligent assistant. Generate a detailed response for the following query asked by the user based on your own knowledge and the context fetched. 
                Context: {rag_context}

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
    document_set = TODO

    # Get the Vector Index
    # TODO need to collect documents (need to use a retriever, like google)
    vector_index = get_build_index(documents=documents, embed_model="local:BAAI/bge-small-en-v1.5", save_dir="./vector_store/index")
    # Create a query engine with the specified parameters
    query_engine = get_query_engine(sentence_index=vector_index, similarity_top_k=10, rerank_top_n=5)

    rag_ctx = get_rag_context(query, query_engine)
    llm_ans = get_response(rag_ctx, query)





















