import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

import transformer
import numpy as np
from sklearn.neighbors import KernelDensity
from transformers import BertModel
from transformers import BertTokenizer
import math
from transformers import GPT2Tokenizer, GPT2Model
from scipy import spatial
from os import listdir, system
from itertools import repeat
import pandas as pd
import subprocess
from transformers import OpenAIGPTTokenizer, OpenAIGPTModel
from sklearn.metrics.pairwise import cosine_similarity
import time
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
import warnings
import torch
from tqdm import tqdm
from joblib import dump, load
import math
torch.set_num_threads(1)

def querykfunction(clf, querytensors):
    #compute the k for each query based on its tensors.
    moe = torch.mean(querytensors,dim=0).tolist()

    moe = np.array(moe).reshape(1, -1)

    k = clf.predict(moe)

    return int( k[0] )


def getfileresults_kde(file,k,dataset_name,neuralnet, label=""):
    dataset = pd.read_csv(file)
    file = file.split("/")[5]
    file = file.split("transformed")[1]
    file = file.spli(".csv")[0]
    startend = file.split("_")

    docs = []
    documentnumbers = dataset.document.unique()

    for doc in documentnumbers:
        ddoc1 = dataset.loc[dataset.document == doc, 'tensor'].to_list()


        doc1to = []
        for t in ddoc1:
            t = transformer.tensorfromstring(t,'BERT')
            doc1to.append(torch.tensor(t))

        doc1to = torch.stack(doc1to)
        doc1to = doc1to.to(torch.float)

        #from the tensor generate the KDE model
        



        docs.append(doc1to)



def getfileresults(file, k, dataset_name, neuralnet, label = "", kneuralnetwork = None):
    #Generate results for all queries for the given file (part of transformed dataset).
    #Get all docs in this file as tensors
    dataset = pd.read_csv(file)
    dynamic = False
    if k == 0:
        dynamic = True

    file = file.split("/")[5]
    file = file.split("transformed")[1]
    file = file.split(".csv")[0]
    startend = file.split("_")
    docs = []
    documentnumbers = dataset.document.unique()
    for doc in documentnumbers:
        ddoc1 = dataset.loc[dataset.document == doc, 'tensor'].to_list()

        doc1to = []
        for t in ddoc1:
            t = transformer.tensorfromstring(t,'BERT')
            doc1to.append(torch.tensor(t))

        doc1to = torch.stack(doc1to)
        doc1to = doc1to.to(torch.float)

        docs.append(doc1to)

    #Get all queries as tensors
    queriesdataset = pd.read_csv("transformed_datasets/" + str(dataset_name) + "/queries_" + str(neuralnet) + ".csv")
    queries = []
    qdict = {}
    queriesnumbers = queriesdataset['query'].unique()

    for q in queriesnumbers:
        qq1 = queriesdataset.loc[queriesdataset['query'] == q, 'tensor'].to_list()

        q1to = []
        for t in qq1:
            t = transformer.tensorfromstring(t,'BERT')
            q1to.append(torch.tensor(t))

        q1to = torch.stack(q1to)
        q1to = q1to.to(torch.float)

        queries.append(q1to)
        qdict[q] = q1to

    nqs = len(queries)
    nds = len(docs)

    #prepare matrix of cos_sim tensors queries /documents
    querydocs = {}
    for i in range(nqs):  
        querydocs[i] = []     
        for j in range(nds):
            querydocs[i].append ( torch.div( torch.matmul(queries[i], docs[j].t()) , 
                (torch.norm(queries[i],dim=1) * torch.norm(docs[j],dim=1).reshape(-1,1)).t() ) )


    #prepare diagonal matrix of cos_sim tensors for doc / doc
    docsdocs = {}
    for i in range(nds):  
        docsdocs[i] = torch.div( torch.matmul(docs[i], docs[i].t()) , 
                (torch.norm(docs[i],dim=1) * torch.norm(docs[i],dim=1).reshape(-1,1)).t() ) 

    
    if label:
        folder = 'results/' + str(dataset_name) + '/document_wide/' + neuralnet + '/k=' + str(k) + "-" + label
        if dynamic:
            folder = 'results/' + str(dataset_name) + '/document_wide/' + neuralnet + '/k=dynamic' + "-" + label
    else:
        folder = 'results/' + str(dataset_name) + '/document_wide/' + neuralnet + '/k=' + str(k)
        if dynamic:
            folder = 'results/' + str(dataset_name) + '/document_wide/' + neuralnet + '/k=dynamic'

    #compute LRD
    if kneuralnetwork:
        clf = load(kneuralnetwork)

    LRD = {}
    for query in querydocs.keys():
        querydoc = querydocs[query]

        #Querydoc contains all the tensors, one for each doc
        for doc,docten in zip(range(len(querydoc)), querydoc):

            querydict = {}

            queryno = queriesnumbers[query]
            queryte = queriesdataset.loc[queriesdataset['query'] == queryno, 'term'].to_list()
            #ten contains n arrays, one for each query term.
            for queryterm,querytermarray in zip(queryte,docten):
                #t contains v values, one for each doc term.

                #Activate dynamic k
                if dynamic:
                    k = querykfunction( clf, qdict[queryno] )

                if k <= len(querytermarray):
                    values, kneighbors = torch.topk(querytermarray, k)
                else:
                    values, kneighbors = torch.topk(querytermarray,len(querytermarray))
                
                #Kneighbors contains the indices of the terms in the document that are k-closest to the query term.
                #values contains the cos_sim between the query term and its kneighbors

                #Now we need to compute RD(A,B) for each B in kn.
                #RD(A,B) = min(kdist(B), cos_sim(A,B))
                #We have the cos_sim(A,B) stored in values but we need to find the kdistance of each kn in its document, and that's 
                #why we use the auxiliary diagonal matrix constructed before
                kdistances = []
                for kn in kneighbors:

                    ####TEST
                    ####SET K = 1 FOR THIS
                    ####KFORNEIGHBORS = 1
                    kforneighbors = k
                    kn = kn.item()

                    #get the tensor containing all the cos_sims between the doc terms and themselves
                    docwithitself = docsdocs[doc]

                    #select the right term, the kneighbor:
                    kneigh_similarities = docwithitself[kn]

                    #kneigh_similarities is a tensor containing the cos_sim of that term with all other terms in the documents. So to obtain the kdistance:
                    if kforneighbors+1 <= len(kneigh_similarities):
                        v,i = torch.topk(kneigh_similarities, kforneighbors+1)
                    else:
                        v,i = torch.topk(kneigh_similarities,len(kneigh_similarities))
                        
                    v = v[1:]
                    i = i[1:]
                    #We need to skip one because the first one will always be itself
                    kdistance = v[-1]

                    kdistances.append(kdistance)
                
                #We are all set to compute the LRD.
                sum = 0

                for ki in range(k):
                    if ki<len(values) and ki<len(kdistances):
                        v = min(values[ki], kdistances[ki])
                        sum = sum + v

                sum = sum/len(kneighbors)

                querydict[queryterm] = sum
                
            LRD[doc] = querydict

        #Get the MEAN value among the LRDs of the query terms, for each document.
        finalscores = {}
        for key,item in LRD.items():
            finalscores[documentnumbers[key]] = torch.mean( torch.stack ( list ( item.values() ) ) ).item()

        #Make CSV out of finalscores
        resultscsv = pd.DataFrame(finalscores.items())
        resultscsv.rename(columns = {0: "document", 1 : "score"}, inplace = True)
        resultscsv.sort_values(by = "score", ascending = False, inplace = True)
        resultscsv.reset_index(drop = True, inplace = True)

        resultscsv.to_csv(folder + "/query_" + str(queriesnumbers[query]) + "/" + str(startend[0]) + "_" + str(startend[1]) + ".csv") 

def initializemodel(neuralnet, dataset):
    if neuralnet == "BERT":
        model = BertModel.from_pretrained("model/BERT")
        tokenizer = BertTokenizer.from_pretrained("model/Tokenizer")

    if neuralnet == "BERTlarge":
        model = BertModel.from_pretrained('bert-large-uncased',
                                    output_hidden_states = True, # Whether the model returns all hidden-states.
                                    )
        model.eval()

        tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
    
    if neuralnet == "gpt2":
        model = GPT2Model.from_pretrained('gpt2', output_hidden_states = True)
        model.eval() 

        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    if neuralnet == "openaigpt":
        model = OpenAIGPTModel.from_pretrained("openai-gpt", output_hidden_states = True)

        model.eval() 

        tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")
        
    if neuralnet == "stateoftheart":
        model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")

        model.eval()
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")

    if neuralnet == "distilroberta":
        model = AutoModel.from_pretrained("sentence-transformers/all-distilroberta-v1")
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-distilroberta-v1")

    if neuralnet == 'mm-distilroberta':
        model = AutoModel.from_pretrained("sentence-transformers/msmarco-distilroberta-base-v2")
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/msmarco-distilroberta-base-v2") 

    if neuralnet == 'dpr':
        model = AutoModel.from_pretrained('sentence-transformers/facebook-dpr-question_encoder-multiset-base')
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/facebook-dpr-question_encoder-multiset-base')

    return model, tokenizer

def listener(datasetname, neuralnet, k, label=""):

    dynamic = False
    if k == 0 or k == 'dynamic':
        dynamic = True

    if label:
        folder = 'results/' + str(datasetname) + '/document_wide/' + neuralnet + '/k=' + str(k) + "-" + label
        if dynamic:
            folder = 'results/' + str(datasetname) + '/document_wide/' + neuralnet + '/k=dynamic' + "-" + label
    else:
        folder = 'results/' + str(datasetname) + '/document_wide/' + neuralnet + '/k=' + str(k)
        if dynamic:
            folder = 'results/' + str(datasetname) + '/document_wide/' + neuralnet + '/k=dynamic'

    dirlist = [ item for item in os.listdir(folder) if os.path.isdir(os.path.join(folder, item)) ]

    for query in tqdm( dirlist ):

        resultsfolder = os.path.join(folder,query)
        results = pd.DataFrame(columns=['query','q0', 'document', 'position', 'score', 'label'])
        queryindex = query.split("_")[1]

        for file in os.listdir(resultsfolder):
            result = pd.read_csv(resultsfolder + "/" + file)
            result['query'] = queryindex
            result['q0'] = 'q0'
            result['label'] = 'test'
            result = result.sort_values(by='score', ascending = False)
            result.reset_index(drop = True, inplace = True)
            result['position'] = result.index.values

            results = pd.concat([results,result], ignore_index = True)
        
        results = results.loc[:, ~results.columns.str.contains('^Unnamed')]

        if label:
            if not dynamic:
                results.to_csv(folder + "/results_" + str(queryindex) + "_k=" + str(k)  + ".csv")
            else:
                results.to_csv(folder + "/results_" + str(queryindex) + "_k=dynamic" + label + ".csv")
        else:
            if not dynamic:
                results.to_csv("results/" + datasetname + "/document_wide/" + neuralnet + "/k=" + str(k) + "/results_" + str(queryindex) + "_k=" + str(k) + ".csv")
            else:
                results.to_csv("results/" + datasetname + "/document_wide/" + neuralnet + "/k=dynamic/results_" + str(queryindex) + "_k=dynamic" + ".csv")            

        for file in os.listdir(resultsfolder):
            os.remove(resultsfolder + "/" + file)

        os.rmdir(resultsfolder)
