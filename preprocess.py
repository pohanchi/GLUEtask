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





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    head = "./Dataset/"
    parser.add_argument("--path",default="WNLI",type=str)
    parser.add_argument("--mode",default="train",type=str,help="dev, train, test!!")
    args=parser.parse_args()
    
    #choose mode
    if args.mode == "train":
        data=pd.read_csv(head + args.path+"/train.tsv",sep='\t',encoding='utf-8')
    if args.mode == "dev":
        data=pd.read_csv(head + args.path+"/dev.tsv",sep='\t',encoding='utf-8')
    if args.mode == "test":
        data=pd.read_csv(head + args.path+"/test.tsv",sep='\t',encoding='utf-8')
    
    predata = data.values[:,1:,].tolist()
    print(len(predata))

    