import torch
import numpy as np 
import tensorboardX 
import tqdm 
import argparse 
import IPython 
import pdb 
import json 
import os 
import pandas as pd  
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from pytorch_transformers import (GPT2LMHeadModel, GPT2Tokenizer,GPT2Config,AdamW, cached_path, WEIGHTS_NAME, CONFIG_NAME, WarmupLinearSchedule)
import pickle 
from pandas import DataFrame



def prep(head,args,mode):
    #choose mode
    if mode == "train":
        data=pd.read_csv(head + args.path+"/train.tsv",sep='\t',encoding='utf-8',error_bad_lines=False)
        data.dropna(inplace = True)
    if mode == "dev":
        data=pd.read_csv(head + args.path+"/dev.tsv",sep='\t',encoding='utf-8',error_bad_lines=False)
        data.dropna(inplace = True)
    if mode == "test":
        data=pd.read_csv(head + args.path+"/test.tsv",sep='\t',encoding='utf-8',error_bad_lines=False)
        data.dropna(inplace = True)
    data['question'] = data['question'].astype(str)
    data['sentence'] = data['sentence'].astype(str)
    try:
        data['label'] = data['label'].astype(str)
    except:
        pass
    predata = data.values[1:,1:,].tolist()
    print(predata[:5])

    return predata



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    head = "../Dataset/"
    parser.add_argument("--path",default="WNLI",type=str)
    parser.add_argument("--mode",default="train",type=str,help="dev, train, test!!")
    args=parser.parse_args()
    
    
    #choose mode
    if args.mode == "train":
        data=DataFrame.from_csv(head + args.path+"/train.tsv",sep='\t',encoding='utf-8',error_bad_lines=False)
    if args.mode == "dev":
        data=pd.read_csv(head + args.path+"/dev.tsv",sep='\t',encoding='utf-8',error_bad_lines=False)
    if args.mode == "test":
        data=pd.read_csv(head + args.path+"/test.tsv",sep='\t',encoding='utf-8',error_bad_lines=False)
    
    predata = data.values[:,1:,].tolist()

    for data in predata:
        print(data)
    
    

    