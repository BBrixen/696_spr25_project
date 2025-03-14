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

from cacher import cache_decorator


# some pages take too long to load
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException()


def google_scrape(url, timeout=10):
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)

        page = requests.get(url)
        if "text/html" not in page.headers["Content-Type"]:
            print(f"No html on {url}")
            return None
        soup = BeautifulSoup(page.content, "html.parser")
        text = soup.get_text(separator="\n", strip=True)

        # reset alarm after function completes
        signal.alarm(0)
        return text
    except TimeoutException:
        print(f"Timeout: {url} took too long.")
    except Exception as e:
        print(f"Error scraping {url}: {e}")
    finally:
        signal.alarm(0) # ensure alarm is disabled even if there is an exception
    return None

@cache_decorator
def get_raw_docs(query, num_docs=10):
    '''
        This will get the raw documents from a google search. These documents are
        then used to build the vector model for searching. 
    '''
    results = []
    for url in search(query, tld="co.in", num=num_docs, stop=num_docs, pause=2):
        text = google_scrape(url)
        if text is None:
            continue

        results.append(Document(text=text))
    return results

'''
following this tutorial:
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
    Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", embed_batch_size=100)
    Settings.node_parser = SentenceSplitter(chunk_size=1000, chunk_overlap=200)
    Settings.num_output = 512
    Settings.context_window = 3900

    # # Check if the save directory exists
    # if not os.path.exists(save_dir):
    #     # Create and load the index
    #     index = VectorStoreIndex.from_documents(
    #         [documents], service_context=Settings
    #     )
    #     index.storage_context.persist(persist_dir=save_dir)
    # else:
    #     # Load the existing index
    #     index = load_index_from_storage(
    #         StorageContext.from_defaults(persist_dir=save_dir),
    #         service_context=Settings,
    #     )

    # always building, because the documents might change in between uses
    index = VectorStoreIndex.from_documents(
        documents, service_context=Settings
    )
    index.storage_context.persist(persist_dir=save_dir)
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
@cache_decorator
def get_best_chunks(query, text, num_chunks=10):
    chunks = chunk_text(text, chunk_size=250)
    embedding_model = get_embedding_model()
    embeddings = [get_embeddings(embedding_model, chunk) for chunk in chunks]

    query_embedding = get_embeddings(embedding_model, query)
    ratings = [cosine_similarity(query_embedding, chunk) for chunk in embeddings]
    top_chunk_ids = np.argpartition(ratings, -num_chunks)[-num_chunks:]

    return [chunks[i] for i in top_chunk_ids]

