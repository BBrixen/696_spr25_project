
import numpy as np
import llama_index
from llama_index.embeddings import HuggingFaceEmbedding
import google.generativeai as palm
from llama_index.llms.palm import PaLM
import math
import openai

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

# TODO

'''
using chatgpt in python:
https://www.geeksforgeeks.org/how-to-use-chatgpt-api-in-python/
'''

def get_response(rag_documents, query):
    messages = [ 
            {
                "role": "system", 
                "content": "TODO put rag documents and instructions here"
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


