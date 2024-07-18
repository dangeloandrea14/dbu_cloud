import os
os.environ["OMP_NUM_THREADS"] = "3" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "3" # export OPENBLAS_NUM_THREADS=1 
os.environ["MKL_NUM_THREADS"] = "3" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "3" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "3" # export NUMEXPR_NUM_THREADS=1

import preprocessing
import transformer
import numpy as np
import pandas as pd
from os.path import exists
from transformers import logging
import argparse
import search
import searchGPU

#for filename in ./transformed_datasets/LISA/document_wide/BERT/*; do qsub launch.sh main.py 4 BERT LISA no-sentence_wide $filename; done
#python main.py --k 3 --dataset LISA --no-searching --transformer BERT

parser = argparse.ArgumentParser(description='Cloud Retrieval')

parser.add_argument('--query', type=str, 
                    help='String to search in the Cloud Retrieval system. \n  ' + 
                    'USAGE: launch without parameters to access the GUI, where you can query and set options.' +
                    ' \n Launch with option --test to perform a test query. This requires the parameters --query and --k to be defined. Use other parameters to set options. \n'
                    + ' Launch with option --fulltest to perform all kind of tests on a defined query. This requires the parameters --query and --k to be defined. ')
parser.add_argument('--transform', type=int, help= "Internal parameter to manage transformation processes")
parser.add_argument('--docsperfile', type=int, help= "Internal parameter to manage transformation processes")
parser.add_argument('--sentence_wide', action=argparse.BooleanOptionalAction, help = "The cloud will be made of tensors from each sentence rather than the entire document.")
parser.add_argument('--transformer', type=str, help= "Choose which transformer to use", choices=["BERT","Word2Vec","BERTlarge", "stateoftheart","gpt2","openaigpt","distilroberta","mm-distilroberta","splade","splademax","dpr"])
parser.add_argument('--k', type=int, help = 'Number of neighbors to consider for local reachability computations.')
parser.add_argument('--searching', action=argparse.BooleanOptionalAction, help = "Start search")
parser.add_argument('--file', type=str, help = 'Search in a specific file of transformed tensors.')
parser.add_argument('--folder', type=str, help = 'Save results in this folder')
parser.add_argument('--dataset', type=str, help = 'Search in a specific dataset.')
parser.add_argument('--label', type=str, help= "Special label for the experiment.")
parser.add_argument('--partitioning', type=int, help= "Choose the amount of parts the dataset will be partitioned into when transforming.")
parser.add_argument('--cls', action=argparse.BooleanOptionalAction, help="Transform into CLS tokens, not clouds." )
parser.add_argument('--kneuralnetwork', type=str, help="Model path to use for k prediction in case of dynamic k")

args = parser.parse_args()

transform = args.transform
docsperfile = args.docsperfile
sentence_wide = args.sentence_wide
neuralnet = args.transformer
k = args.k 
query = args.query
searching = args.searching
file = args.file
folder = args.folder
dataset = args.dataset
label = args.label
cls = args.cls
partitioning = args.partitioning
kneuralnetwork = args.kneuralnetwork

if args.label is None:
    label = ""


queries = pd.read_csv("datasets/preprocessed/" + dataset + '/' + dataset + "_queries.csv", index_col=0).index.values

folder = os.path.join("results", dataset, "document_wide", neuralnet, "k=" + str(k), label)


if args.searching and (args.k is None or args.file is None):
    parser.error("--searching mode requires --file, --query and --k to be specified.")

if(searching):
    searchGPU.getfileresults(file,k,dataset,neuralnet,label = label, kneuralnetwork=kneuralnetwork)
else:
    searchGPU.listener(dataset,neuralnet,k,label = label)
