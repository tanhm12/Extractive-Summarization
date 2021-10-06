import torch 
import os
from config import CNNConfig

from sklearn.utils import shuffle

from transformers import BertModel, BertTokenizerFast

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.nn import LSTM, Conv2d, Linear
from torch.nn.functional import max_pool2d
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


class Bert_Embedding(nn.Module):
    def __init__(self, bert: BertModel, config: CNNConfig):
        super(Bert_Embedding, self).__init__()
        self.bert = bert
        self.bert_hidden = config.bert_hidden * config.bert_n_layers
        self.get_n_layers = config.bert_n_layers
        self.config = config
        
        self.windows_size = config.windows_size
        self.out_channels = config.out_channels
        self.lstm_embedding_size = len(self.windows_size) * config.MAX_SEQ_LEN  
        self.filters = nn.ModuleList([nn.Conv2d(1, self.out_channels,
                                                (i, self.bert_hidden)) for i in self.windows_size])
        self.relu = nn.ReLU()
        
    def forward(self, x, document_mask, attention_mask):
        lens = [mask_i.sum().item() for mask_i in document_mask]

        batch, doc_len, seq_len = list(x.shape)
        x = x.reshape((batch*doc_len, seq_len))
        attention_mask = attention_mask.reshape((batch*doc_len, seq_len))        

        last_hds, pooler_output, hidden_states = self.bert(x, attention_mask, output_hidden_states=True, return_dict=False)
        embeddings = torch.cat(hidden_states[-self.get_n_layers:], axis=-1)  # batch, doc_len, seq_len, self.bert_hidden
        # print(embeddings.shape)
        embeddings = embeddings.reshape((batch * doc_len, 1,  seq_len, self.bert_hidden))  # batch * doc_len, 1, MAX_SEQ_LEN, bert_hidden
        lstm_inputs = []

        for i in range(len(self.windows_size)):
            temp_out = self.filters[i](embeddings).squeeze(-1)  # batch * doc_len, self.out_channels, MAX_SEQ_LEN - self.windows_size[i] + 1
            cnn_result = torch.mean(temp_out, dim=1) # average along out_channels axis
            if cnn_result.shape[1] < self.config.MAX_SEQ_LEN: # pad cnn_result to MAX_SEQ_LEN
                pad_tensor = torch.zeros((cnn_result.shape[0], self.config.MAX_SEQ_LEN - cnn_result.shape[1])).to(cnn_result.device)
                cnn_result = torch.cat([cnn_result, pad_tensor], axis=1)
            lstm_inputs.append(cnn_result)
        lstm_inputs = torch.cat(lstm_inputs, dim=-1).reshape((batch, doc_len, self.lstm_embedding_size)) 
        lstm_inputs = lstm_inputs * torch.nn.functional.sigmoid(lstm_inputs)  # Swish 
        lstm_inputs = pack_padded_sequence(lstm_inputs, lens, batch_first=True, enforce_sorted=False)

        return lstm_inputs


class Document_Encoder(nn.Module):
    def __init__(self, embedding_size, config: CNNConfig):
        super(Document_Encoder, self).__init__()

        self.config = config
        self.embedding_size = embedding_size
        self.doc_encoder = nn.LSTM(self.embedding_size, config.lstm_hidden, num_layers=1,
                            bidirectional=True, batch_first=True)

    def forward(self, lstm_inputs):
        _, doc_encoder_out = self.doc_encoder(lstm_inputs)

        return doc_encoder_out

class Sentence_Extractor(nn.Module):
    def __init__(self, embedding_size, config: CNNConfig):
        super(Sentence_Extractor, self).__init__()

        self.config = config
        self.embedding_size = embedding_size
        self.sentence_extractor = nn.LSTM(self.embedding_size, config.lstm_hidden, num_layers=1,
                                  bidirectional=True, batch_first=True)
        self.dropout_layer = nn.Dropout(0.3)

    def forward(self, lstm_inputs, encoder_in):
        out_packed, (_, __) = self.sentence_extractor(lstm_inputs, encoder_in)
        out, out_lens = pad_packed_sequence(out_packed, batch_first=True)
        out = self.dropout_layer(out)
        return out

class Model(nn.Module):
    def __init__(self, bert, config: CNNConfig):
        super(Model, self).__init__()
        self.config = config
        self.embeddings = Bert_Embedding(bert, config=config)
        self.doc_encoder = Document_Encoder(self.embeddings.lstm_embedding_size, config=config)
        self.sentence_extractor = Sentence_Extractor(self.embeddings.lstm_embedding_size, config=config)

        self.linear = Linear(config.lstm_hidden * 2, 1) 
        self.loss_func = nn.BCELoss()
        self.loss_padding_value = -100
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, document_mask, attention_mask, y=None):
        lstm_inputs = self.embeddings(x, document_mask, attention_mask)

        doc_encoder_out = self.doc_encoder(lstm_inputs)  
        encoder_in = doc_encoder_out

        out = self.sentence_extractor(lstm_inputs, encoder_in)
        out = self.sigmoid(self.linear(out).squeeze(-1))
        # print(out.shape, mask.shape)
        # out *= mask
        
        if y is not None:
            y = pad_sequence(y, batch_first=True, padding_value=self.loss_padding_value).type(torch.FloatTensor).to(out.device)
            loss = self.loss_func(out, y)
            out = nn.functional.softmax(out, dim=-1)
            return out, loss

        return nn.functional.softmax(out, dim=-1)

    def predict(self, tokenizer, doc):
        pass


def torch_save(dir, save_dict):
    os.makedirs(dir, exist_ok=True)

    for name in save_dict:
        torch.save(save_dict[name], os.path.join(dir, name + '.pt'))
    
def torch_load_all(dir):
    save_dict = {}
    for name in os.listdir(dir):
        save_dict[name.replace('.pt', '')] = torch.load(os.path.join(dir, name), map_location=torch.device('cpu'))

    return save_dict