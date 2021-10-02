import torch 
import torch

from config import CNNConfig
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class ESDataset(Dataset):
    def __init__(self, encodings, labels=None, keys=None):
        self.encodings = encodings
        self.labels = labels
        self.keys = keys
        self.encoding_keys = ['input_ids', 'attention_mask']

    def __getitem__(self, idx):
        item = {key: torch.tensor(self.encodings[idx][key]) for key in self.encoding_keys}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings)

def collate_fn(data):
    keys = data[0].keys()

    result = {k: [item[k] for item in data] for k in keys}
    input_ids = result['input_ids']
    result['document_mask'] = [torch.tensor([1] * len(input_ids[i])) for i in range(len(input_ids))]
    
    for k in result:
        if k != 'labels':
            result[k] = pad_sequence(result[k], batch_first=True)
    
    return result