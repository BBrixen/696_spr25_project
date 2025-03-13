import numpy as np
import llama_index
from llama_index.embeddings import HuggingFaceEmbedding
import google.generativeai as palm
from llama_index.llms.palm import PaLM
import math

from api_keys import palm_api_key
