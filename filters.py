from llm_interaction import ask_llm
import string
import retriever
import requests
from bs4 import BeautifulSoup
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification

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
    documents = [doc.text.replace("\n", " ") 
            for doc in documents if llm_trust_doc(query, doc, model)]
    return docs_to_str(documents)


def google_support_entailment(query, documents, model):
    documents = []
    for doc in documents:
        if google_support_entailment_doc(query, doc, model):
            if type(doc) is str:
                documents.append(doc)
            else:
                documents.append(doc.text.replace("\n", " "))
    return docs_to_str(documents)


def google_support_hit_count(query, documents, model):
    # TODO we need to implement google_hit_count
    document_scores = sorted([(google_hit_count(query, doc, model), doc) for doc in documents])
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
    summary = google_summary(query, document, model)
    search_results = retriever.get_raw_docs(summary, 5)

    # we want to see if each document supports or refutes the summary
    support_count = sum(1 if doc_supports_claim(ref_doc, summary) else 0 
            for ref_doc in search_results)
    proportion_support = support_count / len(search_results)
    return proportion_support > threshold


def doc_supports_claim(document, claim):
    doctxt, _ = get_doctxt(document)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gal-lardo/BERT-RTE-LinearClassifier")
    model = AutoModelForSequenceClassification.from_pretrained("gal-lardo/BERT-RTE-LinearClassifier")

    # Prepare input texts
    premise = doctxt
    hypothesis = claim

    # Tokenize and predict
    inputs = tokenizer(premise, hypothesis, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    prediction = outputs.logits.argmax(-1).item()

    # Convert prediction to label
    return prediction == 1  # prediction 1 means entailment


def google_hit_count(query):
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    params = {
        "q": query,
        "hl": "en"
    }
    url = "https://www.google.com/search"

    response = requests.get(url, headers=headers, params=params)
    soup = BeautifulSoup(response.text, "html.parser")

    # Find the element with the result stats
    result_stats = soup.find("div", id="result-stats")
    if not result_stats:
        return None

    # Extract number using regex
    match = re.search(r"About ([\d,]+) results", result_stats.text)
    if match:
        return int(match.group(1).replace(",", ""))
    
    return 0


# def google_hit_count(query, document, model):
#     google_query = google_summary(query, document, model)
#     terms = [t.translate(str.maketrans('','',string.punctuation)) for t in google_query.split()]
#     terms = [t for t in terms if t != '']
#     google_query = "+".join(terms)
#     r = requests.get('http://www.google.com/search',
#                      params={'q':google_query,
#                              "tbs":"li:1"}
#                     )
# 
#     # TODO this is broken
#     soup = BeautifulSoup(r.text)
#     print(soup.prettify())
#     return 0


"""
Random, less involved helpers
"""

def google_summary(query, document, model):
    doctxt, _ = get_doctxt(document)

    prompt = f"""
    Summarize the following chunk of text into a short question that can be google searched. The text should be summarized in context with the following query. Limit it to one sentence and do not provide an answer to the question.

    Contextual Query:
    {query}
    
    Text:
    {doctxt}
    """
    return ask_llm(doctxt, prompt, model)

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
