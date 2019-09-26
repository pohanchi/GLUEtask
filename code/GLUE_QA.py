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

def longest_length(model):
    max_length = model.config.n_positions//2 -3
    return max_length

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

def pre_process_datasets(datasets,input_len,seq,ans_token,end_token,pad_token):
    """ Pre-process datasets containing lists of tuples(story, 1st continuation, 2nd continuation, label)

        To Transformer inputs of shape (n_batch, n_alternative, length) comprising for each batch, continuation:
        input_ids[batch, :] = [story_token] + story[:story_length] + [question_token] + question[:que_length] +[end_token]"""
    tensor_datasets = []
    for dataset in datasets:
        n_batch = len(dataset)
        input_ids = np.zeros((n_batch,input_len),dtype=np.int64)
        lm_labels = np.full((n_batch,input_len),fill_value=-1,dtype=np.int64)
        for i , (sent1,sent2,ans) in enumerate(dataset):
            cannot_calculate_loss = len(sent1 + [seq] + sent2)
            text= sent1 + [seq_token] + sent2 + [ans_token] + ans + [end_token]
            only_need_length = len(text)
            input_ids[i,:only_need_length] = text
            lm_labels[i,:only_need_length] = text
            input_ids[i,only_need_length:] = pad_token
            lm_labels[i,:cannot_calculate_loss] = -1
            
        all_inputs = (input_ids, lm_labels)
        tensor_datasets+= [tuple(torch.tensor(t) for t in all_inputs)]
    return tensor_datasets


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
    parser.add_argument('--model_name', type=str, default='gpt2',help='pretrained model name')
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--mode",default="train",type=str,help="dev, train, test!!")
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,help="Epsilon for Adam optimizer.")
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training \
                        steps to perform. Override num_train_epochs.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before\
                        performing a backward/update pass.")
    parser.add_argument('--learning_rate', type=float, default=6.25e-5)
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--lm_coef', type=float, default=0.9)
    parser.add_argument('--n_valid', type=int, default=374)
    parser.add_argument("--fp16", action='store_true',help="accerlate the model")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    args = parser.parse_args()
    print(args)

    #init setup
    random_seed_setup(args)
    device=setup_device()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


    #data format
    data = prep(args)

    #load model and tokenizer
    tokenizer,special_tokens_ids,special_tokens =process_special_tokens()
    model = GPT2LMHeadModel.from_pretrained(args.model_name)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    #decide the prefer length
    max_length = longest_length(model)
    
    if isinstance(data[0][2]) == int:
        for i in range(len(data)):
            if data[i][2] == 0:
                data[i][2] = "related"
            else:
                data[i][2]= "unrelated"

    dataset = (data,)

    def tokenize_and_encode(obj):
        """ Tokenize and encode a nested object """
        if isinstance(obj, str):
            return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
        elif isinstance(obj, set):
            return obj
        return list(tokenize_and_encode(o) for o in obj)
    
    #encode to index
    encoded_datasets = tokenize_and_encode(dataset)

    input_length = max(len(seq1[:max_length]) + len(seq2[:max_length]) + len(ans[:a_length]) + 3  \
                            for dataset in encoded_datasets for seq1, seq2, ans  in dataset)
    input_length = min(input_length, model.config.n_positions)  # Max size of input for the pre-trained model
    tensor_datasets = pre_process_datasets(encoded_datasets, input_length,*special_tokens_ids)





