from llm_interaction import ask_llm
import string
import retriever

'''
Below are the methods for filtering. Each should take
the document that we want to investigate. Documents are passed as strings

Currently, the query is included in the method signature because
later it *might* be useful to include for checking. And all these functions
need to take the same inputs, so changing 1 means changing all of them

It should return a boolean, where false means the document should
be discarded

(For caching, the document forms the unique id for this function call,
which is different from the usual query. The thought behind this is, that
the result from this function should be the same for each document passed in.
If any of these functions does something expensive (themselves), we can cache it,
otherwise the ask_llm will end up caching for us)
'''

def no_filter(query, documents, model):
    # dont remove any of the documents
    return docs_to_str(documents)


def llm_trust(query, documents, model):
    documents = [doc.text.replace("\n", " ") for doc in documents if llm_trust_doc(query, doc, model)]
    return docs_to_str(documents)


def google_support_entailment(query, documents, model):
    documents = [doc.text.replace("\n", " ") for doc in documents if google_support_entailment_doc(query, doc, model)]
    return docs_to_str(documents)


def google_support_hit_count(query, documents, model):
    # TODO we need to implement google_hit_count
    document_scores = sort([(google_hit_count(doc), doc) for doc in documents])
    top_half = document_scores[len(document_scores):]  # TODO will this round up or down... currently too braindead to tell
    better_docs = [doc.text.replace("\n", " ") for (score, doc) in top_half]
    return docs_to_str(better_docs)



"""
Implementations of each filter
"""


def llm_trust_doc(query, document, model):
    doctxt, source = get_doctxt(document)

    prompt = f"""
    Do you believe that the following document contains only factual and unbiased information.
    Use the source of the document, as well as any other information you can infer, to aid
    in your evaluation.
    {source}
    Document:
    {doctxt}

    Instructions:
    1. Respond with ONLY "yes" or "no"
    2. A "yes" means that this document only contains factual and unbiased information.
    3. A "no" means that this document contains verifiably incorrect claims, or aims to mislead the reader through clear bias.
    """
    #4. After the yes or no, explain why the document is or is not trustworthy
    #Comment on the previous line can be used to look at LLM """logic""", though it's unclear how actually helpful that is tbh

    return ans_is_yes(ask_llm(doctxt, prompt, model))


def google_support_entailment_doc(query, document, model, threshold=0.5):
    doctxt, _ = get_doctxt(document)

    prompt = f"""
    Summarize the following chunk of text into a short question that can be google searched. The text should be summarized in context with the following query. Limit it to one sentence and do not provide an answer to the question.

    Contextual Query:
    {query}
    
    Text:
    {doctxt}
    """
    llm_search = ask_llm(doctxt, prompt, model)
    print("Google search start")
    print(llm_search)
    search_results = retriever.get_raw_docs(llm_search, 5)
    support_count = sum(1 if doc_supports_claim(x) else 0 for x in search_results)
    proportion_support = support_count / search_results
    return proportion_support > threshold


def doc_supports_claim(document, claim, model):
    doctxt, _ = get_doctxt(document)

    prompt = f"""
    Determine if the following claim is supported by the provided document. 

    Claim:
    {claim}

    Document:
    {doctxt}

    Instructions:
    1. Respond with only "yes" or "no"
    2. A "yes" means that this document supports the claim
    3. A "no" means that this document refutes the claim
    """
    # TODO we will change this to use a huggingface model
    return ans_is_yes(ask_llm(doctxt, prompt, model))


    

"""
Helpers
"""

def get_doctxt(document):
    if type(document) is str:
        source = ""
        doctxt = document
    else:
        source = f"\nSource:\n{document.metadata['source']}\n"
        doctxt = document.text.replace("\n", " ")
    return doctxt, source

def ans_is_yes(ans):
    ans = ans.strip()[0:3].lower().translate(str.maketrans('','',string.punctuation)).strip()
    return ans == 'yes'

def docs_to_str(documents):
    return "\n\n".join(documents)
