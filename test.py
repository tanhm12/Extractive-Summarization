import json
import numpy as np
import os
import pandas as pd
import torch

from tqdm import tqdm
from torch.utils.data import DataLoader
from rouge_score import rouge_scorer

from data_utils import ESDataset, collate_fn
from model import Model
from config import CNNConfig


def torch_load_all(dir):
    save_dict = {}
    for name in os.listdir(dir):
        save_dict[name.replace('.pt', '')] = torch.load(os.path.join(dir, name))
    
    return save_dict

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)

def load_json(file):
    with open(file) as f:
        return json.load(f)

def load_text(dir, return_sum=False):
    docs = load_json(os.path.join(dir, 'docs.json'))
    labels = load_json(os.path.join(dir, 'labels.json'))
    if return_sum:
        summaries = load_json(os.path.join(dir, 'summaries.json'))
        return docs, labels, summaries
    return docs, labels


def get_test_loader(docs, config):
    encodings = []
    for doc in docs:
        encodings.append(tokenizer(doc[:config.MAX_DOC_LEN], truncation=True,
                                   max_length=config.MAX_SEQ_LEN, padding='max_length'))
    
    test_dataset = ESDataset(encodings)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    return test_loader


def get_all_probs(model, all_doc, config):
    res = []
    test_loader = get_test_loader(all_doc, config=config)
    for item in tqdm(test_loader):
        ids = item['input_ids'].to(config.device)
        document_mask = item['document_mask'].to(config.device)
        attention_mask = item['attention_mask'].to(config.device)
        prob = model(ids, document_mask, attention_mask)[0].tolist()
        res.append(prob)
            
    return res

def calculateSimilarity(sentence, doc):
    score = scorer.score('\n'.join(doc), sentence)
    return np.mean([score['rouge2'][2], score['rougeLsum'][2]])

def choose_summary_mmr(doc, prob, k=3, alpha=0.9):
    prob = np.array(prob)
    idx = [np.argmax(prob)]
    prob[idx[0]] = 0
    summary = [doc[idx[0]]]

    while len(idx) < min(k, len(doc)):
        mmr = -100 * np.ones_like(prob)
        for i, sent in enumerate(doc):
            if prob[i] != 0:
                mmr[i] = alpha * prob[i] - (1-alpha) * calculateSimilarity(sent, summary)
        pos = np.argmax(mmr)
        prob[pos] = 0
        summary.append(doc[pos])
        idx.append(pos)
    summary = sorted(list(zip(idx, summary)), key=lambda x: x[0])
    return [x[1] for x in summary]

def choose_summary(doc, prob, k=3):
    idx = torch.topk(torch.tensor(prob), k=k).indices.tolist()
    return [doc[i] for i in sorted(idx)]


def test(documents, summaries, all_probs, choose_summary=choose_summary, k=3, save_dir='./save/cnn_test_result'):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)
    res = {'rouge1': [], 'rouge2': [], 'rougeLsum': []}
    for i, document in enumerate(tqdm(documents)):
        processed_document = document[: len(all_probs[i])]
        score = scorer.score('\n'.join(summaries[i]), '\n'.join(choose_summary(processed_document, all_probs[i], k))) # target, prediction
        for cate in res:
            res[cate].append(score[cate][2])  # f1 score
        # print(i, [res[cate][-1] for cate in res])
    print('\n\nResult :')
    for cate in res:
        x = np.mean(res[cate])
        print(cate, x)
        res[cate].extend(['', x])
    res['doc_id'] = list(range(len(documents))) + ['', 'overall']

    df = pd.DataFrame.from_dict(res)
    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(os.path.join(save_dir, 'result.csv'), index=False)


if __name__ == '__main__':
    final_dict = torch_load_all('save/cnn/best-model-f1')
    config: CNNConfig = final_dict['config']

    bert = config.bert_type.from_pretrained(config.bert_name)
    tokenizer = config.tokenizer_type.from_pretrained(config.bert_name)
    model = Model(bert, config).to(config.device)
    model.load_state_dict(final_dict['model_state_dict'])
    model.eval()

    test_texts, test_dict_labels, test_summaries = load_text(config.test_data_dir, return_sum=True)

    test_ids = list(test_texts.keys())
    docs = [test_texts[i] for i in test_ids]
    summaries = [test_summaries[i] for i in test_ids]

    probs = get_all_probs(model, docs, config)
    test(docs, summaries, probs, choose_summary, k=3, save_dir='./save/cnn_test_result')


