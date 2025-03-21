from llm_interaction import ask_llm

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

def no_filter(query, document, local=True):
    return True  # default, all documents pass


def llm_trust(query, document, local=True):
    prompt = f"""
    Do you believe that the following document contains only factual and unbiased information.
    Use the source of the document, as well as any other information you can infer, to aid
    in your evaluation.

    Document:
    {document}

    Instructions:
    1. Respond with ONLY "yes" or "no"
    2. A "yes" means that this document only contains factual and unbiased information.
    3. A "no" means that this document contains verifiably incorrect claims, or aims to mislead the reader through clear bias.
    """

    # recall that for caching, the document is the unique id for this func call
    llm_trust_ans = ask_llm(document, prompt, local=local)
    if llm_trust_ans.lower() == 'yes':
        print(prompt)
    return llm_trust_ans.lower() == 'yes'


def google_support(query, document, local=True):
    return True  # default, all documents pass
