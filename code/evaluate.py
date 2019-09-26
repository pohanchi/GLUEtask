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
from utils import process_special_tokens, random_seed_setup

def top_k_top_p_filtering(logits,
                          top_k=0,
                          top_p=0.0,
                          filter_value=-float('Inf'),device='cpu'):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits,top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1)
        # IPython.embed()
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
            ..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        row_index = (sorted_indices_to_remove == 1).nonzero()[:,0]
        logits[row_index,indices_to_remove] = filter_value
    return logits



def sample_sequence(model,
                    context,
                    tokenizer,
                    num_samples=1,
                    temperature=1,
                    top_k=0,
                    top_p=0.9,
                    is_xlnet=False,
                    device='cpu',argmax=False):
    generated = context.clone().detach()
    generated = generated.unsqueeze(0)
    with torch.no_grad():
        end_token = "_eos_"
        end_word = tokenizer.convert_tokens_to_ids(end_token)
        next_token = "_none_"
        start_time = time.time()
        past = None
        count = 0
        while next_token != end_word:
            output = model(generated)
            
            next_token_logits = output[0][:, -1, :] / temperature
            
            if not argmax:
                filtered_logits = top_k_top_p_filtering(
                    next_token_logits, top_k=top_k, top_p=top_p,device=device)
                next_tokens = torch.multinomial(
                    F.softmax(filtered_logits, dim=-1), num_samples=num_samples,replacement=True)
                next_tokens_output=torch.mode(next_tokens, dim=-1, keepdim=True)
                next_token=next_tokens_output[0]
                
            else:
                next_token = torch.argmax(next_token_logits,keepdim=True,dim=-1)

            if count > 0:
                if next_token != end_word:
                    next_token_matrix = torch.cat((next_token_matrix,next_token),dim=1)
            else:
                if next_token != end_word:
                    next_token_matrix = next_token
            if next_token==end_word and count==0:
                return torch.tensor([[]])

            if count >=20:
                break

            generated = torch.cat((generated, next_token), dim=1)
            count += 1
        end_time = time.time()
        # print("It took {} seconds".format(end_time-start_time))

    return next_token_matrix

def evaluate_and_summary(args,special_tokens_ids,model,dev_dataloader,test_dataloader,writer,loss):
    
    model = model.eval()
    device=args.device
    n_gpu = torch.cuda.device_count()
    logger.info("device: {}, n_gpu {}".format(device, n_gpu))
    answer_dict = dict()
    compared_dict = dict()
    tqdm_bar = tqdm(eval_dataloader, desc="Evaluating")
    for step, data in enumerate(tqdm_bar):
        start_time = time.time()
        sentence, answer, ID_index = tuple(t.to(device) for t in data)
        sentence = sentence[sentence != special_tokens_ids[3]].long()
        answer = answer[answer != special_tokens_ids[3]].long()

        out = sample_sequence(
            model=model,
            context=sentence,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=args.device,
            is_xlnet=False,
            tokenizer=tokenizer,
            argmax=args.argmax,num_samples=args.sample)
        
        end_time = time.time()
        
        # print("It costs {} seconds for generate data!!".format(end_time-start_time))
        out_ = out[:, :].tolist()
        answer_ = tokenizer.decode(answer.tolist(),clean_up_tokenization_spaces=True)
        for i in range(len(out_)):
            text = tokenizer.decode(out_[i], clean_up_tokenization_spaces=True,skip_special_tokens=True)
            answer_dict[ID_list[ID_index[0][i]]] = text
            compared_dict[ID_list[ID_index[0][i]]] = (text,answer_)
