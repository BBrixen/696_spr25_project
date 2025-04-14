from openai import OpenAI
from cacher import cache
import openai
from llama_index.llms.ollama import Ollama

# handling apis
#from api_keys import openai_api_key
#client = OpenAI(api_key=openai_api_key)
#openai.api_key = openai_api_key


def ask_llm(query, prompt, model):
    # the query is the original query that started everything
    # it is not used except for caching these responses
    if 'llama' in model:
        return ask_llama(query, prompt, model=model)
    else if 'gemini' in model:
        return ask_gemini(query, prompt)
    else:
        return ask_openai(query, prompt)


@cache
def ask_llama(query, prompt, model='llama2'):
    llm = Ollama(model=model, request_timeout=60.0)
    response = llm.complete(prompt)
    return response.text

@cache
def ask_gemini(query, prompt):
    # look into https://aistudio.google.com/app/apikey
    # TODO
    return None


@cache
def ask_openai(query, prompt):
    messages = [ 
            {
                "role": "user", 
                "content": prompt
            }
        ]

    #chat = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    #reply = chat.choices[0].message.content
    #return reply
