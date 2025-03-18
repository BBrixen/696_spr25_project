import os
import functools
import hashlib

cache_dir = "cache_rag"

def hash_query(s):
    return f"q_{hashlib.sha256(s.encode('utf-8')).hexdigest()}"

def get_dir_path(step):
    return f"./{cache_dir}/{step}"

def get_file(query, step):
    return f"{get_dir_path(step)}/{hash_query(query)}.txt"


def cache_result(query, step, text):
    '''
    caches results from LLM and retrieval lookups so that we dont keep 
    querying LLMs in each test run (saves us some money)
    '''
    # debug printing
    if type(text) is not str:
        print(f"unable to cache {step}. {type(text)=}")
        return

    # query and step form a key for the text (which is a value)
    query_fn = hash_query(query)
    dir_path = get_dir_path(step)
    os.makedirs(dir_path, exist_ok=True)

    with open(f"{dir_path}/{query_fn}.txt", 'w') as file:
        file.write(text)
    print(f"cache PUT {step}/{query_fn}")


def check_cache(query, step):
    query_fn = hash_query(query)
    dir_path = get_dir_path(step)
    os.makedirs(dir_path, exist_ok=True)
    file = get_file(query, step)

    if not os.path.isfile(file):
        return None

    query_file = get_file(query, step)
    print(f"cache GET {step}/{query_fn}")
    with open(file) as f:
        return f.read()
    

# this is the main baddie. looking hot ngl
def cache(func):
    @functools.wraps(func)
    def wrapper(query, *args, **kwargs):
        func_name = func.__name__
        query_file = get_file(query, func_name)

        # use cache result if it exists
        cached_result = check_cache(query, func_name)
        if cached_result is not None:
            return cached_result

        # call function and cache its return value for future calls
        func_retval = func(query, *args, **kwargs)
        cache_result(query, func_name, func_retval)
        return func_retval
    return wrapper

