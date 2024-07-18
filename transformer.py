import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so

import torch
from sentence_transformers import SentenceTransformer 
torch.set_printoptions(profile="full")
from transformers import BertTokenizer
from transformers import BertModel
from sklearn.feature_extraction import text
import pandas as pd
import subprocess
import string
import time
from sklearn.neighbors import LocalOutlierFactor
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
from transformers import GPT2Tokenizer, GPT2Model
from transformers import OpenAIGPTTokenizer, OpenAIGPTModel
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
from os import listdir



def tensorfromstring(string, neuralnet):
    tensor = []
    if neuralnet == "BERT":
        tensorsstring = string.split("tensor")

    for tensorstring in tensorsstring:
        tensorstring = tensorstring.replace("tensor","")
        tensorstring = tensorstring.replace("(","")
        tensorstring = tensorstring.replace("[","")
        tensorstring = tensorstring.replace(")","")
        tensorstring = tensorstring.replace("]","")

        nums = tensorstring.split(",")

        for num in nums:
            if(num != ''):
                tensor.append(float(num))

    output = np.array(tensor)

    return output