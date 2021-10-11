from transformers import BertModel, BertTokenizerFast
from transformers import RobertaModel, RobertaTokenizerFast


class CNNConfig:
    def __init__(self):
        self.train_data_dir = 'data/cnndm/cnn/train'
        self.val_data_dir = 'data/cnndm/cnn/valid'
        self.test_data_dir = 'data/cnndm/cnn/test'

        self.bert_type = BertModel
        self.tokenizer_type = BertTokenizerFast
        self.bert_name = 'prajjwal1/bert-small'

        self.MAX_SEQ_LEN = 128 # token level
        self.MAX_DOC_LEN = 48

        self.bert_hidden = 512
        self.bert_n_layers = 4

        self.windows_size = [1, 3, 5, 10]
        self.out_channels = 50
        self.lstm_hidden = 256
        self.device = 'cuda:3'

        self.batch_size = 6
        self.num_epochs = 7
        self.warmup_steps = 500
        self.gradient_accumulation_steps = 16
        self.print_freq = 0.05
        self.save_dir = './save/cnn'


class VietnewsConfig:
    def __init__(self):
        self.train_data_dir = 'data/vietnews_processed/train'
        self.val_data_dir = 'data/vietnews_processed/valid'
        self.test_data_dir = 'data/vietnews_processed/test'

        self.bert_type = RobertaModel
        self.tokenizer_type = RobertaTokenizerFast
        self.bert_name = 'Zayt/viRoberta-l6-h384-word-cased'  # my Roberta pretrained model on  4gb of text

        self.MAX_SEQ_LEN = 96 # token level
        self.MAX_DOC_LEN = 32

        self.bert_hidden = 384
        self.bert_n_layers = 3

        self.windows_size = [1, 3, 5, 10]
        self.out_channels = 50
        self.lstm_hidden = 256
        self.device = 'cuda:3'

        self.batch_size = 6
        self.num_epochs = 7
        self.warmup_steps = 500
        self.gradient_accumulation_steps = 16
        self.print_freq = 0.05
        self.save_dir = './save/vietnews'

# config = CNNConfig()