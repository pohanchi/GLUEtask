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
from preprocess import prep

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {}, n_gpu {}".format(device, n_gpu))
    return device

def process_special_tokens():
    special_tokens = ['_sep_', '_ans_','_eos_','_pad_']
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2',unk_token='<|endoftext|>')
    special_tokens_ids = list(tokenizer.convert_tokens_to_ids(token) for token in special_tokens)
    tokenizer.add_tokens(special_tokens)
    special_tokens_ids = list(tokenizer.convert_tokens_to_ids(token) for token in special_tokens)
    return  tokenizer,special_tokens_ids,special_tokens

def random_seed_setup(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    head = "./Dataset/"
    parser.add_argument("--path",default="WNLI",type=str)
    parser.add_argument("--mode",default="train",type=str,help="dev, train, test!!")
    args=parser.parse_args()


    #data format
    data = prep(args)

    #load model



