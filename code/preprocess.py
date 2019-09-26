import torch
import numpy as np 
import tensorboardX 
import tqdm 
import argparse 
import IPython 
import pdb 
import json 
import os 
# import pandas as pd  
import csv
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from pytorch_transformers import (GPT2LMHeadModel, GPT2Tokenizer,GPT2Config,AdamW, cached_path, WEIGHTS_NAME, CONFIG_NAME, WarmupLinearSchedule)
import pickle 
from pandas import DataFrame

def read_f(reader):
    data_list = list()
    num = 0
    for row in reader:
        if num == 0:
            num +=1
            continue
        if len(row) != 4:
            continue
        try:
            data_list += [(row[1], row[2],row[3])]
        except:
            data_list = data_list[:-2]
    print("length of data list:",len(data_list))
    return data_list

def prep(head,args,mode):
    #choose mode
    if mode == "train":
        with open(head + args.path+"/train.tsv")as fin:
            reader = csv.reader(fin,delimiter='\t')
            data=read_f(reader)
        # data.dropna(inplace = True)
    if mode == "dev":
        with open(head + args.path+"/dev.tsv")as fin:
            reader = csv.reader(fin,delimiter='\t')
            data=read_f(reader)
    if mode == "test":
        with open(head + args.path+"/test.tsv")as fin:
            reader = csv.reader(fin,delimiter='\t')
            data=read_f(reader)
        
    return  data



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
    
    

    