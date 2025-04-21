import llama_index
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex, load_index_from_storage
from llama_index.core import Settings
from llama_index.core import StorageContext
from llama_index.core.postprocessor import MetadataReplacementPostProcessor, SentenceTransformerRerank
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
import numpy as np
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from googlesearch import search
from bs4 import BeautifulSoup
import requests
import math
import signal
from cacher import cache
from llama_index.llms.ollama import Ollama
import threading
from urllib.parse import urlsplit



@cache
def google_scrape(url, timeout=30):
    def fetch():
        try:
            page = requests.get(url)
            if "text/html" not in page.headers["Content-Type"]:
                print(f"No html on {url}")
                return None
            soup = BeautifulSoup(page.content, "html.parser")
            text = soup.get_text(separator="\n", strip=True)
            text = text.replace("\n", " ")  # replace line breaks with spaces throughout all docs

            return text
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return None

    result = [None]
    def fill_result():
        result[0] = fetch()

    thread = threading.Thread(target=fill_result)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        print(f"Timeout: {url} took too long")
        return ""
    return result[0]


@cache
def get_google_urls(query, num_docs):
    urls = []
    for url in search(query, tld="co.in", num=num_docs, stop=num_docs, pause=2):
        urls.append(url)
    # use \n as url separator, since that wont be present
    return "\n".join(urls)


def get_raw_docs(query, num_docs=10):
    '''
        This will get the raw documents from a google search. These documents are
        then used to build the vector model for searching. 
    '''
    # TODO add support for injecting incorrect documents  (later, for testing)
    results = []
    urls = get_google_urls(query, num_docs)
    for url in urls.split("\n"):
        text = google_scrape(url)
        if text is None or text == "":
            continue
        results.append(Document(text=text, extra_info={"source": urlsplit(url).netloc}))
    return results

'''
following this tutorial:
https://www.freecodecamp.org/news/how-to-build-a-rag-pipeline-with-llamaindex/
'''

def get_key_documents(query, documents, local=True):
    docs_as_str = _get_key_documents(query, documents, local=local)
    docs_list = [str_to_doc(doc_str) for doc_str in docs_as_str.split("\n\n")]
    return docs_list


def doc_to_str(doc):
    text = doc.text.replace('\n', ' ')
    src = doc.metadata['source']
    return f"{text}\n{src}"

def str_to_doc(s):
    s = s.split('\n')
    text = s[0]
    src = s[1]
    return Document(text=text, extra_info={'source': src})


@cache
def _get_key_documents(query, documents, local=True):
    '''
        This splits the documents into chunks and identifies the
        most useful documents for the given query. 

        This takes the long list of input documents and finds key portions of
        those documents to serve as the RAG documents for the LLM
    '''
    if len(documents) == 0:
        return "No external context"  # edge case

    # Get the Vector Index and Query Engine
    vector_index = get_build_index(documents=documents, local=local)  # NOTE: still using llama2 for the vector engine, this should be fine
    query_engine = get_query_engine(sentence_index=vector_index, similarity_top_k=10, rerank_top_n=5)
    try:
        engine_response = query_engine.query(query)
        context_docs = engine_response.source_nodes
        return "\n\n".join([doc_to_str(doc) for doc in context_docs])
        # \n\n is doc separator since no documents contain \n in them
        # \n is removed during fetch, and removed again now just in case
    except Exception as e:
        print(f"Error getting chunks: {e}")
        return ""

def get_build_index(documents, local=True):
    """
    Builds or loads a vector store index from the given documents.

    Args:
        documents (list[Document]): A list of Document objects.

    Returns:
        VectorStoreIndex: The built or loaded index.
    """

    # pick llm to use
    if local:
        # Settings.llm = Llama(model_path="./models/codellama-13b.Q3_K_S.gguf")
        # Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        Settings.llm = Ollama(model="llama2", request_timeout=60.0)
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    else:
        Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", embed_batch_size=100)

    Settings.node_parser = SentenceSplitter(chunk_size=1000, chunk_overlap=200)
    Settings.num_output = 512
    Settings.context_window = 3900

    index = VectorStoreIndex.from_documents(
        documents, service_context=Settings
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

