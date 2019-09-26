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
import logging
import tqdm 
from tqdm import tqdm, trange

from tensorboardX import SummaryWriter

from torch.utils.data import (
    DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from pytorch_transformers import (GPT2LMHeadModel, GPT2Tokenizer, GPT2Config,
                                  AdamW, cached_path, WEIGHTS_NAME, CONFIG_NAME, WarmupLinearSchedule)
import pickle
from preprocess import prep
from evaluate import evaluate_and_summary,Dump_json
from utils import longest_length, setup_device, process_special_tokens, pre_process_datasets, random_seed_setup

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    head = "../Dataset/"
    parser.add_argument("--path", default="WNLI", type=str)
    parser.add_argument('--model_name', type=str,
                        default='gpt2', help='pretrained model name')
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_LLL", default="", type=str,
                        help="which one ewc you want to run ? there have three method: EWC, SI, MAS, IMM")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--seed', type=int, default=42)

    # train
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument("--eval_step", default=100, type=int)
    parser.add_argument("--adam_epsilon", default=1e-8,
                        type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=6.25e-5)
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--lm_coef', type=float, default=0.9)
    parser.add_argument('--n_valid', type=int, default=374)
    parser.add_argument("--fp16", action='store_true',
                        help="accerlate the model")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    # evaluate
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training \
                        steps to perform. Override num_train_epochs.")
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help="Number of updates steps to accumulate before\
                        performing a backward/update pass.")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--argmax", action="store_true")
    parser.add_argument("--sample", type=int, default=1)
    args = parser.parse_args()

    args = parser.parse_args()
    print(args)

    # init setup
    random_seed_setup(args)
    device = setup_device()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # data format
    data = prep(head, args, "train")
    dev_data = prep(head, args, "dev")
    # test_data = prep(head, args,"test")
    
    # load model and tokenizer
    tokenizer, special_tokens_ids, special_tokens = process_special_tokens()
    model = GPT2LMHeadModel.from_pretrained(args.model_name)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    # decide the prefer length
    max_length,a_length = longest_length(model)

    print(data[:5])
    if isinstance(data[0][2], int):
        for i in range(len(data)):
            if data[i][2] == 0:
                data[i][2] = "related"
            else:
                data[i][2] = "unrelated"

    datasets = (data, dev_data)

    def tokenize_and_encode(obj):
        """ Tokenize and encode a nested object """
        if isinstance(obj, str):

            return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
        elif isinstance(obj, set):
            return obj
        return list(tokenize_and_encode(o) for o in obj)
    
    # encode to index
    encoded_datasets = tokenize_and_encode(datasets)
    # Max size of input for the pre-trained model
    input_length = 800
    tensor_datasets = pre_process_datasets(
        encoded_datasets, input_length,a_length, *special_tokens_ids)

    train_dataset = TensorDataset(*(tensor_datasets[0]))
    dev_dataset = TensorDataset(*(tensor_datasets[1]))
    # test_dataset = TensorDataset(*tensor_datasets[2])

    train_sampler = RandomSampler(train_dataset)
    dev_sampler = RandomSampler(dev_dataset)
    # test_sampler = RandomSampler(test_dataset)

    train_dataloader=  DataLoader(train_dataset, sampler=train_sampler,
                     batch_size=2)
    dev_dataloader= DataLoader(dev_dataset, sampler=dev_sampler, batch_size=1)
    # test_dataloader(test_dataset, sampler=dev_sampler, batch_size=1)

    if args.do_train:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        num_train_optimization_steps = len(
            train_dataloader) * args.num_train_epochs
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = WarmupLinearSchedule(
            optimizer, warmup_steps=args.warmup_steps, t_total=num_train_optimization_steps)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level,loss_scale=128)

    if args.do_train:
        writer = SummaryWriter(args.output_dir+"/tensorboard/")
        nb_tr_steps, tr_loss, exp_average_loss = 0, 0, None
        model.train()
        if args.do_LLL != "":
            importance = args.importance
            Reg = Regularization(
                model=model, mode="EWC", dataloader=old_dataloader, device=device, optimizer=optimizer)
        step_step = 1
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_steps = 0
            tqdm_bar = tqdm(train_dataloader, desc="Training")
            for i, batch in enumerate(tqdm_bar):
                batch = tuple(t.to(device) for t in batch)
                input_ids, lm_labels,_ = batch
                losses = model(input_ids, labels=lm_labels)
                if args.do_LLL != "":
                    loss = losses[0] + importance * Reg.penalty(model)
                else:
                    loss = losses[0]

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm)

                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()

                # evalaute
                if step_step % args.eval_step == 0:
                    if args.do_eval:
                        evaluate_and_summary(
                            args,special_tokens_ids,tokenizer,model, dev_dataloader, writer, loss,step_step,device)

                optimizer.zero_grad()
                tr_loss += loss.item()
                step_step += 1
                exp_average_loss = loss.item() if exp_average_loss is None else 0.7 * \
                    exp_average_loss+0.3*loss.item()
                nb_tr_steps += 1
                tqdm_bar.desc = "Training loss: {:.2e} lr: {:.2e}".format(
                    exp_average_loss, scheduler.get_lr()[0])

    # Save a trained model
    if args.do_train:
        # Save a trained model, configuration and tokenizer
        model_to_save = model.module if hasattr(
            model, 'module') else model  # Only save the model it-self
        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        tokenizer.save_pretrained(args.output_dir)
        tokenizer.save_vocabulary(args.output_dir)
        # Load a trained model and vocabulary that you have fine-tuned
        model = OpenAIGPTLMHeadModel.from_pretrained(args.output_dir)
        tokenizer = OpenAIGPTTokenizer.from_pretrained(args.output_dir)
        model.to(device)
        # Dump_json(args,special_tokens_ids,model, test_dataloader, writer, step_step)
