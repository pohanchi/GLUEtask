import torch
import numpy as np 
import tensorboardX 
import tqdm 
import argparse 
import IPython 
import pdb 
import json 
import os 
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from pytorch_transformers import (GPT2LMHeadModel, GPT2Tokenizer,GPT2Config,AdamW, cached_path, WEIGHTS_NAME, CONFIG_NAME, WarmupLinearSchedule)
import pickle 
import logging
import random
import logging

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO)
logger = logging.getLogger(__name__)


def longest_length(model):
    max_length = model.config.n_positions//2 -3 -20
    ans_length = 20
    return max_length, ans_length  

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

def pre_process_datasets(datasets,input_len,a_length,seq_token,ans_token,end_token,pad_token):
    """ Pre-process datasets containing lists of tuples(story, 1st continuation, 2nd continuation, label)

        To Transformer inputs of shape (n_batch, n_alternative, length) comprising for each batch, continuation:
        input_ids[batch, :] = [story_token] + story[:story_length] + [question_token] + question[:que_length] +[end_token]"""
    tensor_datasets = []
    count = 1
    for dataset in datasets:
        n_batch = len(dataset)
        input_ids = np.zeros((n_batch,800),dtype=np.int64)
        answer_span = np.zeros((n_batch,a_length))

        lm_labels = np.full((n_batch,800),fill_value=-1,dtype=np.int64)
        for i , (sent1,sent2,ans) in enumerate(dataset):
            if count == 1:
                cannot_calculate_loss = len(sent1 + [seq_token] + sent2)
                text= sent1 + [seq_token] + sent2 + [ans_token] + ans + [end_token]
                only_need_length = len(text)
                if only_need_length > 800:
                    continue
                input_ids[i,:only_need_length] = text
                lm_labels[i,:only_need_length] = text
                input_ids[i,only_need_length:] = pad_token
                lm_labels[i,:cannot_calculate_loss] = -1
            else:
                cannot_calculate_loss = len(sent1 + [seq_token] + sent2)
                text = sent1 + [seq_token] + sent2 + [ans_token]
                only_need_length = len(text)
                if only_need_length > 800:
                    continue 
                answer_span[i,:len(ans)] = ans
                answer_span[i,len(ans):] = pad_token
                only_need_length = len(text)
                input_ids[i,:only_need_length] = text
                lm_labels[i,:only_need_length] = text
                input_ids[i,only_need_length:] = pad_token
                lm_labels[i,:cannot_calculate_loss] = -1
        count += 1    
        all_inputs = (input_ids, lm_labels,answer_span)
        tensor_datasets+= [tuple(torch.tensor(t) for t in all_inputs)]
    return tensor_datasets

def random_seed_setup(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    return
