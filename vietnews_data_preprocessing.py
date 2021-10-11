import json
import os
import numpy as np
import pandas as pd
import re

# speed up stanza for sent_tokenize by using gpu
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import stanza
from underthesea import sent_tokenize as sent_tokenize_uts, word_tokenize as word_tokenize_uts
from rouge_score import rouge_scorer
from tqdm import tqdm

LOWER = False
LENGTH_THRESHOLD = 10
rouge_factors = {'rouge1': 0.2, 'rouge2': 0.3, 'rougeL': 0.5}  


## This is stanza tokenizer, it is more accurate but much slower than nltk tokenizer
# def sent_tokenize(doc):
#     doc = nlp(doc)
#     sentences = []
#     for sentence in doc.sentences:
#         # print(sentence.tokens[0])
#         sentence = ' '.join([token.text for token in sentence.tokens])
#         if len(sentence) > LENGTH_THRESHOLD:
#             sentences.append(sentence)
    
#     return sentences

LOWER = False
LENGTH_THRESHOLD = 15
rouge_factors = {'rouge1': 0.3, 'rouge2': 0.3, 'rougeL': 0.4}  

def sent_tokenize(doc):
    return sent_tokenize_uts(doc)

def word_tokenize(text, format='list'):
    return word_tokenize_uts(text, format=format)

def parse_file(file):
    with open(file, encoding='utf-8') as f:
        document = f.read().rstrip().split("\n\n")
    for i in range(len(document)):
        document[i] = [text for text in document[i].split('\n') if len(text) > 0]
        
    if len(document) < 4:
        title, summary, doc = document 
        img_caption = ''
    else:
        title, summary, doc, img_caption = document 
    summary = sent_tokenize(summary[0])
    return title, summary, doc, img_caption


def make_label(doc, sum, scorer):
    doc_size = len(doc)
    res = [0] * doc_size
    n = min(len(sum), doc_size)
    # f1 of rouge-L
    for j in range(n):
        score = [scorer.score(sum[j], sent_i) for sent_i in doc]
        score = [( 
            x['rouge1'][2] * rouge_factors['rouge1'] + \
            x['rouge2'][2] * rouge_factors['rouge2'] + \
            x['rougeL'][2] * rouge_factors['rougeL']
            ) for x in score]
        sent_pos = np.argmax(score)
        for i in range(doc_size):
            if res[sent_pos] == 1:
                score[sent_pos] = 0
                sent_pos = np.argmax(score)
            else:
                res[sent_pos] = 1
                break
        # print(score[sent_pos])
        # print(doc[sent_pos])
        # print(sum[j], "\n")
    return res

def process(files):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
    img_captions = {}
    titles = {}
    docs = {}
    summaries = {}
    labels = {}
    remove_files = []
    for idx in tqdm(range(len(files))):
        # if idx%1000 == 0:
        #     print('\n', os.getpid(), idx)
        try:
            title, summary, doc, img_caption = parse_file(os.path.join(files[idx]))
        except:
            continue
        # print(title, summary, doc, img_caption)
        if len(doc) < len(summary) or len(doc) == 0 or len(summary) == 0:
            remove_files.append(files[idx])   
            continue    
        label = make_label(doc, summary, scorer)  # binary array indicates which sentence would be chosen for summary
        titles[files[idx]] = title  # list of title (list of text)
        docs[files[idx]] = doc  # list of body sentence (list of text)
        labels[files[idx]] = label  
        summaries[files[idx]] = summary  # list of summary sentence (list of text)
        img_captions[files[idx]] = img_caption  # list of image caption (list of text)
        # if idx%5000 == 0:
        #     a = list(zip(label, doc))
        #     for i in a:
        #         print(len(i[1]), i[0], i[1])
        #     print('##########\n','\n'.join(summary))
    return titles, docs, labels, summaries, img_captions, remove_files

def json_dump(obj, file):
    with open(file, 'w', encoding='utf8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)

def process_and_write(files, write_dir):
    titles, docs, labels, summaries, img_captions, remove_files = process(files)

    os.makedirs(write_dir, exist_ok=True)
    json_dump(docs, os.path.join(write_dir, 'docs.json'))
    json_dump(labels, os.path.join(write_dir, 'labels.json'))
    json_dump(summaries, os.path.join(write_dir, 'summaries.json'))
    json_dump(titles, os.path.join(write_dir, 'titles.json'))
    json_dump(img_captions, os.path.join(write_dir, 'img_captions.json'))
    json_dump(remove_files, os.path.join(write_dir, 'remove_files.json'))


if __name__ == '__main__':
    base_dir = 'data/vietnews/data'

    def get_files(name):
        dir = os.path.join(base_dir, name)
        return [os.path.join(dir, file) for file in os.listdir(dir)]

    train_files = get_files('train_tokenized')
    valid_files = get_files('val_tokenized')
    test_files = get_files( 'test_tokenized')


    base_write_dir = 'data/viet_new_processed'
    process_and_write(valid_files, os.path.join(base_write_dir, 'valid'))
    process_and_write(test_files, os.path.join(base_write_dir, 'test'))
    process_and_write(train_files, os.path.join(base_write_dir, 'train'))