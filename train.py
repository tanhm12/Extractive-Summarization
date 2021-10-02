import json
import os
import numpy as np
from datetime import datetime


from nltk.tokenize import sent_tokenize as nltk_sent_tokenize
from rouge_score import rouge_scorer
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score

from config import CNNConfig
from model import Model
from data_utils import ESDataset, collate_fn


import torch
from torch.utils.data import DataLoader

config = CNNConfig()
tokenizer = config.tokenizer_type.from_pretrained(config.bert_name)

def load_json(file):
    with open(file) as f:
        return json.load(f)

def load_text(dir):
    docs = load_json(os.path.join(dir, 'docs.json'))
    labels = load_json(os.path.join(dir, 'labels.json'))
    return docs, labels

def get_encodings(docs, labels, toknizer=tokenizer, config=config):
    keys = list(docs.keys())
    encodings = []
    return_labels = []

    for k in tqdm(keys):
        encodings.append(tokenizer(docs[k][:config.MAX_DOC_LEN], truncation=True,
                                   max_length=config.MAX_SEQ_LEN, padding='max_length'))
        return_labels.append(labels[k][:config.MAX_DOC_LEN])
    
    return keys, encodings, return_labels

def torch_save(dir, save_dict):
    os.makedirs(dir, exist_ok=True)

    for name in save_dict:
        torch.save(save_dict[name], os.path.join(dir, name + '.pt'))
    
def torch_load_all(dir):
    save_dict = {}
    for name in os.listdir(dir):
        save_dict[name.replace('.pt', '')] = torch.load(os.path.join(dir, name))
    
    return save_dict


def train(model, train_loader, val_loader, optimizer=None, scheduler=None, config=config, start_epoch=0):
    model.train()
    all_train_loss, all_dev_loss, all_dev_f1 = [], [], []
    best_dev_loss = 1e9
    best_dev_f1 = 0

    print_freq = config.print_freq
    batch_size = config.batch_size
    epochs = config.num_epochs
    gradient_accumulation_steps = config.gradient_accumulation_steps
    save_dir = config.save_dir

    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5e-3, epochs=epochs, steps_per_epoch=len(train_loader.dataset))
    print_after = int(print_freq * len(train_loader.dataset) / batch_size)
    for epoch in range(start_epoch, epochs):
        print_counter = 0
        total_loss = []
        print('Epoch {} started on {}'.format(epoch, datetime.now().strftime('%d/%m/%Y %H:%M:%S')))
        for step, item in enumerate(tqdm(train_loader)):
            ids = item['input_ids'].to(config.device)
            document_mask = item['document_mask'].to(config.device)
            attention_mask = item['attention_mask'].to(config.device)
            label = item['labels']
            logits, loss = model(ids, document_mask, attention_mask, y=label)
            
            loss = loss / gradient_accumulation_steps
            loss.backward()
            if (step+1) % gradient_accumulation_steps == 0 or step == len(train_loader.dataset)-1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            total_loss.append(loss.item())
            if step > print_counter:
                print('Step: {}, loss: {}, total loss: {}'.format(step, loss.item(), np.mean(total_loss)))
                print_counter += print_after
        
        print('Train loss:', np.mean(total_loss))
        all_train_loss.append(total_loss)
        result_dict = {'epoch': epoch, 'all_train_loss': all_train_loss}

        save_dict = {
            'model_state_dict': model.state_dict(),
            'config': config,
            # 'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
            # 'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'result_dict': result_dict
            }
        if val_loader is not None:
            dev_loss, dev_f1 = eval(model, val_loader, config)
            all_dev_loss.append(dev_loss)
            all_dev_f1.append(dev_f1)
            print('Dev loss: {}, Dev F1: {}'.format(dev_loss, dev_f1))

            custom_save_dict = {
                'model_state_dict': model.state_dict(),
                'config': config,
                'result_dict': result_dict
            }
            if dev_loss < best_dev_loss:
                torch_save(os.path.join(save_dir, 'best-model-loss'), custom_save_dict)
                best_dev_loss = dev_loss
            if dev_f1 > best_dev_f1:
                torch_save(os.path.join(save_dir, 'best-model-f1'), custom_save_dict)
                best_dev_f1 = dev_f1

            result_dict.update ({
                'all_dev_loss': all_dev_loss,
                'all_dev_f1': all_dev_f1,
                'best_dev_loss': best_dev_loss,
                'best_dev_f1': best_dev_f1
                })

        
        torch_save(os.path.join(save_dir, 'checkpoint_{}'.format(epoch)), save_dict)

        # torch_save(os.path.join(save_dir, 'model_{}.pt'.format(str(epoch))),
        #            model, config, epoch, optimizer, scheduler, all_train_loss, all_dev_loss, best_dev_loss)
        print('Finish epoch {} on {}. \n'.format(epoch, datetime.now().strftime('%d/%m/%Y %H:%M:%S')))
 

def eval(model, val_loader, get_report=True):
    model.eval()
    total_loss = []
    y_pred = []
    y_true = []


    with torch.no_grad():
        for item in val_loader:
            ids = item['input_ids'].to(config.device)
            document_mask = item['document_mask'].to(config.device)
            attention_mask = item['attention_mask'].to(config.device)
            labels = item['labels']
            logits, loss = model(ids, document_mask, attention_mask, y=labels)
            
            prob = logits
            batch_y_true = []
            batch_y_pred = [[] for i in range(len(labels))]
            for j, sent in enumerate(labels):
                last_index = len(labels[j])
                batch_y_true.extend(labels[j])
                temp_prob = np.argsort(prob[j, :last_index].tolist())
                batch_y_pred[j] = [0] * len(sent)
                # Get top 4 best sentence
                for k in temp_prob[-4:]:
                    batch_y_pred[j][k] = 1
            y_true.extend(batch_y_true)
            for sent in batch_y_pred:
                y_pred.extend(sent)
            total_loss.append(loss.item())

    if get_report:
        print(classification_report(y_true, y_pred))
        
    model.train()    
    
    return np.mean(total_loss), f1_score(y_true, y_pred)

if __name__ == '__main__':
    # load text
    train_texts, train_dict_labels = load_text(config.train_data_dir)
    val_texts, val_dict_labels = load_text(config.val_data_dir)

    # get input encodings
    print('Make encoding ...')
    train_keys, train_encodings, train_labels = get_encodings(train_texts, train_dict_labels)
    val_keys, val_encodings, val_labels = get_encodings(val_texts, val_dict_labels)

    # make dataloader
    train_dataset = ESDataset(train_encodings, train_labels, train_keys)
    val_dataset = ESDataset(val_encodings, val_labels, val_keys)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)

    bert = config.bert_type.from_pretrained(config.bert_name)
    model = Model(bert, config=config).to(config.device)
    train(model, train_loader, val_loader, config=config)

