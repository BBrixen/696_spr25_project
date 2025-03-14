import os

cache_dir = "rag_cache"


def format_str_as_filename(s):
    return "_".join(s.split())

def get_dir_path(step):
    step_dir = format_str_as_filename(step)
    dir_path = f"./{cache_dir}/{step_dir}"
    return dir_path

def get_file(query, step):
    query_file = f"{get_dir_path(step}/{format_str_as_filename(query)}.txt"
    return query_file


def cache_result(query, step, text):
    '''
    caches results from LLM and retrieval lookups so that we dont keep 
    querying LLMs in each test run (saves us some money)
    '''

    # query and step form a key for the text (which is a value)
    query_fn = format_str_as_filename(query)
    dir_path = get_dir_path(step)
    os.makedirs(dir_path, exist_ok=True)

    with open(f"{dir_path}/{query_fn}.txt", 'w') as file:
        file.write(text)

def check_cache(query, step):
    query_fn = format_str_as_filename(query)
    dir_path = get_dir_path(step)
    os.makedirs(dir_path, exist_ok=True)
    file = get_file(query, step)

    if not os.path.isfile(file):
        return None
    with open(file) as f:
        return f.read()
    

# this is the main baddie. looking hot ngl
def cache_decorator(func):
    @functools.wrap(func)
    def wrapper(query, *args, **kwargs):
        func_name = func.__name__
        query_file = get_file(query, func_name)

        # use cache result if it exists
        cached_result = check_cache(query, func_name)
        if cached_result is not None:
            print(f"cache GET {query_file}")
            return cached_result

        # call function and cache its return value for future calls
        func_retval = func(query, *args, **kwargs)
        if type(func_retval) is str:
            print(f"cache PUT {query_file}")
            cache_result(query, func_name, func_retval)
        return func_retval
    return wrapper

