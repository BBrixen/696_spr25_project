from llm_interaction import ask_llm
import json

model = "gemini2"

def entry_is_memorized(json_str):
    return entry_is_memorized1(json_str) or entry_is_memorized2(json_str)

def entry_is_memorized1(json_str):
    json_data = json.loads(json_str)

    prompt = f"""
    You are provided with the answer and the fake answer from the RGB factuality benchmark dataset. Finish the query as appeared in the dataset. The query must exactly match the instance in the dataset. 

    Answer: {json_data['answer']}
    Fake answer: {json_data['fakeanswer']}
    Query: 
    """
    return ask_llm(prompt, prompt, model).strip() == json_data['query'].strip()

def entry_is_memorized2(json_str):
    json_data = json.loads(json_str)

    prompt = f"""
    You are provided with the query and answer from the RGB factuality benchmark dataset. Fill in the fake answer as appeared in the dataset. The fake answer must exactly match the instance in the dataset. Recall that in this dataset, the fake answer is given with the key 'fakeanswer' in the json file.

    Query: {json_data['query']}
    Answer: {json_data['answer']}
    Fake answer: 
    """
    return ask_llm(prompt, prompt, model).strip() == json_data['fakeanswer'].strip()

with open(f"./evaluation/RGB2/data/en_fact_{model}_filtered.json", 'w') as filtered_file:
    with open("./evaluation/RGB2/data/en_fact.json") as file:
        non_mem_entries = [line for line in file if not entry_is_memorized(line)]
        filtered_file.writelines(non_mem_entries)
