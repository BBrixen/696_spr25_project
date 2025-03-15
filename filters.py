from cacher import cache

@cache
def ask_llm_trust(query, documents):
    # TODO documents needs to be a string
    return documents

@cache
def ask_llm_summarize(query, documents):
    # TODO documents needs to be a string
    return documents
