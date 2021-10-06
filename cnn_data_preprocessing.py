import json
import os
import numpy as np
import pandas as pd
import re

# speed up stanza for sent_tokenize by using gpu
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import stanza
from nltk.tokenize import sent_tokenize as nltk_sent_tokenize
from rouge_score import rouge_scorer
from tqdm import tqdm

LOWER = False
LENGTH_THRESHOLD = 10
rouge_factors = {'rouge1': 0.4, 'rouge2': 0.3, 'rougeL': 0.3}  


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

def sent_tokenize(doc):
    return nltk_sent_tokenize(doc)

# def reconstruct_text(text):
#     return re.sub('\s([?.!"](?:\s|$))', '', text)

def parse_file(file):
    with open(file, encoding='utf-8') as f:
        document = f.read().rstrip().split("\n\n@highlight\n\n")
    summary = document[1:]
    doc = sent_tokenize(document[0])
    return doc, summary


def make_label(doc, sum, scorer):
    doc_size = len(doc)
    res = [0] * doc_size
    n = min(len(sum), doc_size)
    for j in range(n):
        score = [scorer.score(sum[j], sent_i) for sent_i in doc]
        score = [( 
            # x['rouge1'][2] * rouge_factors['rouge1'] + \
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

def process(data_dir, files):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    docs = {}
    summaries = {}
    labels = {}
    remove_files = []
    for idx in tqdm(range(len(files))):
        # if idx%1000 == 0:
        #     print('\n', os.getpid(), idx)
        doc, summary = parse_file(os.path.join(data_dir, files[idx]))
        if len(doc) < len(summary) or len(doc) == 0 or len(summary) == 0:
            remove_files.append(files[idx])   
            continue    
        label = make_label(doc, summary, scorer)
        docs[files[idx]] = doc
        labels[files[idx]] = label
        summaries[files[idx]] = summary
        # if idx%5000 == 0:
        #     a = list(zip(label, doc))
        #     for i in a:
        #         print(len(i[1]), i[0], i[1])
        #     print('##########\n','\n'.join(summary))
    return docs, labels, summaries, remove_files

def json_dump(obj, file):
    with open(file, 'w', encoding='utf8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)

def process_and_write(data_dir, files, write_dir):
    print(data_dir, write_dir)
    docs, labels, summaries, remove_files = process(data_dir, files)

    os.makedirs(write_dir, exist_ok=True)
    json_dump(docs, os.path.join(write_dir, 'docs.json'))
    json_dump(labels, os.path.join(write_dir, 'labels.json'))
    json_dump(summaries, os.path.join(write_dir, 'summaries.json'))
    json_dump(remove_files, os.path.join(write_dir, 'remove_files.json'))


if __name__ == '__main__':
    # url list from https://github.com/abisee/cnn-dailymail
    with open('data/cnndm/filenames/cnn_files.json') as f:
        filenames = json.load(f)

    train_files = filenames['train']
    valid_files = filenames['valid']
    test_files = filenames['test']


    # stanza.download(lang='en')
    # nlp = stanza.Pipeline(lang='en', processors='tokenize')

    base_write_dir = 'data/cnndm/cnn'
    process_and_write('cnn/stories', valid_files, os.path.join(base_write_dir, 'valid'))
    process_and_write('cnn/stories', train_files, os.path.join(base_write_dir, 'train'))
    process_and_write('cnn/stories', test_files, os.path.join(base_write_dir, 'test'))