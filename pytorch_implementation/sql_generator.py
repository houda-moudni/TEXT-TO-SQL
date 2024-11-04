from model import build_transformer
from dataset import CustomDataset, Utils
from config import get_config, get_weights_file_path, latest_weights_file_path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from torch.utils.tensorboard import SummaryWriter

import json

def get_model(seq_len_src, seq_len_tgt,vocab_src_len, vocab_tgt_len, d_model):
    model = build_transformer(seq_len_src, seq_len_tgt,vocab_src_len, vocab_tgt_len, d_model)
    return model


def load_model(model_filename,seq_len_src, seq_len_tgt,vocab_src_len, vocab_tgt_len, d_model):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state = torch.load(model_filename)
    model = get_model(seq_len_src, seq_len_tgt,vocab_src_len, vocab_tgt_len, d_model).to(device)
    model.load_state_dict(state['model_state_dict'])
    
    return model

def json_loader(file_path,encoding='utf-8'):

    with open(file_path,'r', encoding=encoding) as file:
        content = json.load(file)

        return content

def pad_sequence(sequence, max_length):
    padded_sequence = sequence[:max_length] + ['<PAD>'] * (max_length - len(sequence))
    return padded_sequence



class SqlGenerator():

    def __init__(self, model_filename):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.config = json_loader('config.json', encoding='latin-1')
        
        self.src_vocab = self.config["qst_train_vocab"]
        self.tgt_vocab = self.config["qer_train_vocab"]
        
        self.len_src_vocab = len(self.src_vocab)
        self.len_tgt_vocab = len(self.tgt_vocab)
        
        self.len_seq_src = self.config["seq_len_src"]
        self.len_seq_tgt = self.config["seq_len_tgt"]

        self.d_model = self.config["d_model"]

        self.model = load_model(model_filename,self.len_seq_src, self.len_seq_tgt, self.len_src_vocab, self.len_tgt_vocab,self.d_model) 


    def generate_sql(self, question):
        
        self.model.eval()

        sequence = [question.split()]
        sequence = pad_sequence(sequence, self.len_seq_src)
        indexed_question = [self.tgt_vocab[token] for token in sequence]

        src = torch.tensor(indexed_question).unsqueeze(0).to(self.device)
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)

        with torch.no_grad():
            encoder_output = self.model.encode(src, src_mask)
            tgt = self.decode(encoder_output, src_mask)

            predictions = tgt.cpu().numpy()

        return predictions
    

    def decode(self,encoder_output, src_mask):

        tgt = torch.zeros(1, self.len_seq_tgt, dtype=torch.long).to(self.device)  
        
        for i in range(1, self.len_seq_tgt):
            tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
            decoder_output = self.model.decode(encoder_output, src_mask, tgt, tgt_mask)
            proj_output = self.model.project(decoder_output)

            # Get the next token (this example uses greedy decoding)
            next_token = proj_output[:, -1, :].argmax(dim=-1)
            tgt[:, i] = next_token

            if next_token.item() == self.tgt_vocab.stoi['</S>']: 
                break

        return tgt
    






