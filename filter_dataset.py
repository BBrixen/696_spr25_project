from llm_interaction import ask_llm
import json

model = "gemini2"

def entry_is_non_mem(json_str):
    json_data = json.loads(json_str)

    prompt = f"""
    You are provided with the answer and the fake answer from the RGB factuality benchmark dataset. Finish the query as appeared in the dataset. The query must exactly match the instance in the dataset. 

    Answer: {json_data['answer']}
    Fake answer: {json_data['fakeanswer']}
    Query: 
    """
    return ask_llm(prompt, prompt, model) != json_data['query']


with open(f"./evaluation/RGB2/data/en_fact_{model}_filtered.json", 'w') as filtered_file:
    with open("./evaluation/RGB2/data/en_fact.json") as file:
        non_mem_entries = [line for line in file if entry_is_non_mem(line)]
        filtered_file.writelines(non_mem_entries)
