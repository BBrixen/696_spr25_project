import json
from llm_interaction import ask_llm
import string
import retriever
import requests
from bs4 import BeautifulSoup
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def modified_sgs(query, document):
    summary = google_summary(query, document, "gemini2")
    # doing 10 here to get more fine grained results
    search_results = retriever.get_raw_docs(summary, 10)  
    print()
    print()
    print(type(search_results))
    print(type(search_results[0]))
    print()
    print()

    # we want to see if each document supports or refutes the summary
    support_count = sum(1 if doc_supports_claim(ref_doc, summary) else 0 
            for ref_doc in search_results)
    proportion_support = support_count / len(search_results)
    return proportion_support


def collect_docs():
    corrects, incorrects = [], []
    with open("./evaluation/RGB2/data/en_fact.json") as file:
        for line in file:
            json_data = json.loads(line)
            print(json_data)
            query = json_data["query"]
            for doc in json_data["positive"]:
                corrects.append((query, doc))
            for doc in json_data["negative"]:
                incorrects.append((query, doc))

    return corrects, incorrects

def score_docs(docs, filename):
    props = []
    for (query, doc) in docs:
        prop_support = modified_sgs(query, doc)
        props.append({
            "query": query,
            "doc": doc,
            "prop": prop_support
            })

    json_obj = json.dumps({"results": props}, indent=2)
    with open(filename, 'w') as file:
        file.write(json_obj)


def read_proportions():
    with open("correct_proportions.json") as file:
        json_obj = json.load(file)

    correct_props = []
    for res in json_obj["results"]:
        correct_props.append(res["prop"])

    with open("incorrect_proportions.json") as file:
        json_obj = json.load(file)

    incorrect_props = []
    for res in json_obj["results"]:
        incorrect_props.append(res["prop"])

    return correct_props, incorrect_props


def initialize():
    corrects, incorrects = collect_docs()
    score_docs(corrects, "correct_proportions.json")
    score_docs(incorrects, "incorrect_proportions.json")


def report_for_threshold(corrects, incorrects, threshold):
    def x(s):
        return str(round(100 * s, 2))
    TP = sum(1 if prop >= threshold else 0 for prop in corrects)
    FN = len(corrects) - TP

    FP = sum(1 if prop >= threshold else 0 for prop in incorrects)
    TN = len(incorrects) - FP

    f1 = (2*TP) / (2*TP + FP + FN)
    prec = TP / (TP + FP)
    recall = TP / (TP + FN)
    f2 = 5*prec*recall / (4*prec + recall)
    aoi = TN / len(incorrects)
    # print(f"\n{threshold=}")
    # print(f"accuracy correct:   {TP / len(corrects)}")
    # print(f"precision:          {prec}")
    # print(f"accuracy incorrect: {TN / len(incorrects)}")
    # print(f"F1:                 {f1}")
    # print(f"F2:                 {f2}")
    print(f"SGS( {threshold} ) & {x(recall)} & {x(prec)} & {x(aoi)} & {x(f1)} & {x(f2)} \\\\")


def main():
    corrects, incorrects = read_proportions()
    report_for_threshold(corrects, incorrects, 0.25)
    report_for_threshold(corrects, incorrects, 0.5)
    report_for_threshold(corrects, incorrects, 0.75)
    report_for_threshold(corrects, incorrects, 1)


"""
    helpers
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
    print(f"DOC SUPPORT PREDICTION: {prediction}")
    return prediction == 1  # prediction 1 means entailment


def get_doctxt(document):
    if type(document) is str:
        source = ""
        doctxt = document
    else:
        source = f"\nSource:\n{document.metadata['source']}\n"
        doctxt = document.text.replace("\n", " ")
    return doctxt, source


if __name__ == '__main__':
    main()
