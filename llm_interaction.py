from openai import OpenAI
from cacher import cache
import openai
from llama_index.llms.ollama import Ollama
import requests
from api_keys import openai_api_key, gemini_api_key


def ask_llm(query, prompt, model):
    if 'llama' in model:
        if model == 'llama':
            model = 'llama2'
        return ask_llama(query, prompt, model=model)
    elif 'gemini' in model:
        return ask_gemini(query, prompt)
    elif 'openai' in model:
        return ask_openai(query, prompt)
    else:
        print("MODEL UNDEFINED")
        return None


@cache
def ask_llama(query, prompt, model='llama2'):
    llm = Ollama(model=model, request_timeout=60.0)
    response = llm.complete(prompt)
    return response.text


@cache
def ask_gemini(query, prompt):
    # look into https://aistudio.google.com/app/apikey

    time.sleep(4)  # deal with ratelimit issues
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={gemini_api_key}"
    headers = {'Content-Type': 'application/json'}
    data = {
            'contents': [{
                'parts': [{
                    'text': prompt
            }]}]}
    res = requests.post(url, headers=headers, json=data)
    json_res = res.json()
    print(json_res)
    return json_res['candidates'][0]['content']['parts'][0]['text']


@cache
def ask_openai(query, prompt):
    messages = [ 
            {
                "role": "user", 
                "content": prompt
            }
        ]

    #client = OpenAI(api_key=openai_api_key)
    #openai.api_key = openai_api_key
    #chat = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    #reply = chat.choices[0].message.content
    #return reply
