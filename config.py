from transformers import BertModel, BertTokenizerFast

class CNNConfig:
    def __init__(self):
        self.train_data_dir = 'data/cnndm/cnn/train'
        self.val_data_dir = 'data/cnndm/cnn/valid'
        self.test_data_dir = 'data/cnndm/cnn/test'

        self.bert_type = BertModel
        self.tokenizer_type = BertTokenizerFast
        self.bert_name = 'prajjwal1/bert-small'

        self.MAX_SEQ_LEN = 128
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

# config = CNNConfig()