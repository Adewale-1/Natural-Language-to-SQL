import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        # Initialize the tokenizer
        self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        self.bos_token_id = self.tokenizer.get_vocab()['<extra_id_0>']  
        self.split = split
        
        # Process the data
        self.encoder_inputs, self.decoder_inputs, self.decoder_targets = self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        # Load the natural language queries and SQL queries
        nl_path = os.path.join(data_folder, f"{split}.nl")
        sql_path = os.path.join(data_folder, f"{split}.sql")
        
        nl_queries = load_lines(nl_path)
        
        # For test set, we don't have SQL queries
        if split != "test":
            sql_queries = load_lines(sql_path)
        else:
            sql_queries = [""] * len(nl_queries) 
        
        # Tokenize the inputs and outputs
        encoder_inputs = []
        decoder_inputs = []
        decoder_targets = []
        
        for nl_query, sql_query in zip(nl_queries, sql_queries):
            # Tokenize natural language query for encoder
            tokenized_nl = tokenizer.encode(nl_query, add_special_tokens=True, return_tensors="pt").squeeze(0)
            encoder_inputs.append(tokenized_nl)
            
            if split != "test":
                # Tokenize SQL query for decoder
                tokenized_sql = tokenizer.encode(sql_query, add_special_tokens=True, return_tensors="pt").squeeze(0)
                
                # Decoder input starts with BOS token and excludes the last token
                decoder_input = torch.tensor([self.bos_token_id] + tokenized_sql[:-1].tolist())
                decoder_inputs.append(decoder_input)
                
                # Decoder target is the full SQL query (without BOS token)
                decoder_targets.append(tokenized_sql)
            else:
                # For test set, only need a BOS token for initial decoder input
                decoder_input = torch.tensor([self.bos_token_id])
                decoder_inputs.append(decoder_input)
                decoder_targets.append(torch.tensor([]))  
        
        return encoder_inputs, decoder_inputs, decoder_targets
    
    def __len__(self):
        return len(self.encoder_inputs)

    def __getitem__(self, idx):
        if self.split != "test":
            return (
                self.encoder_inputs[idx],
                self.decoder_inputs[idx],
                self.decoder_targets[idx]
            )
        else:
            return (
                self.encoder_inputs[idx],
                self.decoder_inputs[idx],
                None  
            )

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_inputs = [item[0] for item in batch]
    decoder_inputs = [item[1] for item in batch]
    decoder_targets = [item[2] for item in batch]
    
    # Pad sequences
    encoder_ids = pad_sequence(encoder_inputs, batch_first=True, padding_value=PAD_IDX)
    decoder_inputs_padded = pad_sequence(decoder_inputs, batch_first=True, padding_value=PAD_IDX)
    decoder_targets_padded = pad_sequence(decoder_targets, batch_first=True, padding_value=PAD_IDX)
    
    # Create attention masks (1 for non-padding tokens, 0 for padding)
    encoder_mask = (encoder_ids != PAD_IDX).float()
    
    # Extract the initial decoder input (first token) for each sequence in the batch
    initial_decoder_inputs = torch.tensor([decoder_inputs[i][0] for i in range(len(decoder_inputs))])
    
    return encoder_ids, encoder_mask, decoder_inputs_padded, decoder_targets_padded, initial_decoder_inputs

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_inputs = [item[0] for item in batch]
    decoder_inputs = [item[1] for item in batch]
    
    # Pad sequences
    encoder_ids = pad_sequence(encoder_inputs, batch_first=True, padding_value=PAD_IDX)
    
    # Create attention masks (1 for non-padding tokens, 0 for padding)
    encoder_mask = (encoder_ids != PAD_IDX).float()
    
    # Extract the initial decoder input (first token) for each sequence in the batch
    initial_decoder_inputs = torch.tensor([decoder_inputs[i][0] for i in range(len(decoder_inputs))])
    
    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    # Load all data splits for prompting
    train_nl_path = os.path.join(data_folder, "train.nl")
    train_sql_path = os.path.join(data_folder, "train.sql")
    dev_nl_path = os.path.join(data_folder, "dev.nl")
    dev_sql_path = os.path.join(data_folder, "dev.sql")
    test_nl_path = os.path.join(data_folder, "test.nl")
    
    train_x = load_lines(train_nl_path)
    train_y = load_lines(train_sql_path)
    dev_x = load_lines(dev_nl_path)
    dev_y = load_lines(dev_sql_path)
    test_x = load_lines(test_nl_path)
    
    return train_x, train_y, dev_x, dev_y, test_x