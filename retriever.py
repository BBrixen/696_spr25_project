import llama_index
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex, load_index_from_storage
from llama_index.core import Settings
from llama_index.core import StorageContext
from llama_index.core.postprocessor import MetadataReplacementPostProcessor, SentenceTransformerRerank
from google import google


def get_raw_docs(query):
    '''
        This will get the raw documents from a google search. These documents are
        then used to build the vector model for searching. 
    '''
    num_page = 3
    results = google.search(query, num_page)
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
    # Settings.llm = 'gtp3.5-turbo'
    Settings.embed_model = embed_model
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
        [documents], service_context=Settings
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


