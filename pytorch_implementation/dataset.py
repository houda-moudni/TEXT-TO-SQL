import torch
from torch.utils.data import Dataset, DataLoader
import json

class Utils :
    
    def __init__(self,file, attribut):
        self.file = file
        self.attribut = attribut
        
    def json_loader(self):
        sentences = []
        with open(self.file) as file:
            content = json.load(file)
        for line in content:
            if self.attribut in line:
                sentences.append(line[self.attribut].strip())
            else:
                print(f"Warning: Attribute '{self.attribut}' not found in JSON line: {line}")
        return sentences
    

    def get_max_len(self, sentences):
        max_len = 0
        for sentence in sentences:
            if len(sentence.split()) > max_len:
                max_len= len(sentence.split())
        return max_len



    def get_vocabulary(self, sentences):

        dictionary_set = []
        dictionary = {}

        for sentence in sentences: 
            dictionary_set.extend(set(sentence.lower().split()))

        set_dic = set(dictionary_set)
        for i, token in enumerate(set_dic):
            dictionary[token] = i+3

        dictionary['<PAD>']  = 0
        dictionary['<UNK>']  = 1
        dictionary['<S>']  = 2
        dictionary['</S>']  = len(dictionary)
        return dictionary
    

    def utils(self):

        sentences = self.json_loader()
        max_len = self.get_max_len(sentences)
        dictionary = self.get_vocabulary(sentences)
        
        return dictionary , sentences, max_len


class CustomDataset(Dataset):
    def __init__(self, source_sentences, target_sentences, source_vocab, target_vocab,max_len_src, max_len_tgt):
        self.source_sentences = source_sentences
        self.target_sentences = target_sentences
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        self.max_len_src = max_len_src
        self.max_len_tgt = max_len_tgt


    def __len__(self):
        return len(self.source_sentences)

    def __getitem__(self, idx):
        source_sequence, _ = self.tokenize_and_encode(self.source_sentences[idx], self.source_vocab, add_tag=True)
        target_sequence, _ = self.tokenize_and_encode(self.target_sentences[idx], self.target_vocab, add_tag=True)
        
        source_sequence = self.pad_sequence(source_sequence, self.max_len_src)
        target_sequence = self.pad_sequence(target_sequence, self.max_len_tgt)
        
        return torch.tensor(source_sequence, dtype=torch.long), torch.tensor(target_sequence, dtype=torch.long)  
    
    
    def tokenize_and_encode(self, sentence, vocab, add_tag=True):
        tokens = sentence.split() 
        if add_tag:
            tokens = ['<S>'] + tokens + ['</S>'] 
        
        self.token_ids = [vocab[token] if token in vocab else vocab['<UNK>'] for token in tokens] 
        return self.token_ids, tokens

    def pad_sequence(self, sequence, max_length):
        padded_sequence = sequence[:max_length] + [self.source_vocab['<PAD>']] * (max_length - len(sequence))
        return padded_sequence
    

    def tokens_to_words(self, encoded_sequence, vocab):
        decoded_seq = []
        for i in range(len(encoded_sequence)):
            for key, value in vocab.items():
                if  encoded_sequence[i] == value:
                    decoded_seq.append(key)

        return decoded_seq
                    

