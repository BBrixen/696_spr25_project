from openai import OpenAI
from cacher import cache
import openai
from llama_cpp import Llama

# handling apis
from api_keys import openai_api_key
client = OpenAI(api_key=openai_api_key)
openai.api_key = openai_api_key


def ask_llm(prompt, local=True):
    if local:
        return ask_llama(prompt)
    else:
        return ask_openai(prompt)


@cache
def ask_llama(prompt):
    llm = Llama(model_path="./models/codellama-13b.Q3_K_S.gguf")
    output = llm(prompt)
    return output["choices"][0]["text"]


@cache
def ask_openai(prompt):
    messages = [ 
            {
                "role": "user", 
                "content": prompt
            }
        ]

    chat = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    reply = chat.choices[0].message.content
    return reply
