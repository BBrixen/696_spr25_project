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


'''
following this tutorial:
https://medium.com/@mahakal001/building-a-rag-pipeline-step-by-step-0a5e1ac68562
'''

# helpers

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def chunk_text(text, chunk_size):
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

def get_embedding_model():
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    return embed_model

def get_embeddings(embed_model, text: str):
    embeddings = embed_model.get_text_embedding(text)
    return embeddings

def dot_product(vec1, vec2):
    return sum(a * b for a, b in zip(vec1, vec2))

def magnitude(vec):
    return math.sqrt(sum(v**2 for v in vec))

def cosine_similarity(vec1, vec2):
    dot_prod = dot_product(vec1, vec2)
    mag_vec1 = magnitude(vec1)
    mag_vec2 = magnitude(vec2)

    if mag_vec1 == 0 or mag_vec2 == 0:
        return 0  # Handle division by zero

    return dot_prod / (mag_vec1 * mag_vec2)


# my implementation of selecting chunks from docs
@cache
def get_best_chunks(query, text, num_chunks=10, sources=[""]*10):
    chunks = chunk_text(text, chunk_size=250)
    embedding_model = get_embedding_model()
    embeddings = [get_embeddings(embedding_model, chunk) for chunk in chunks]

    query_embedding = get_embeddings(embedding_model, query)
    ratings = [cosine_similarity(query_embedding, chunk) for chunk in embeddings]
    top_chunk_ids = np.argpartition(ratings, -num_chunks)[-num_chunks:]

    return [chunks[i] for i in top_chunk_ids]

