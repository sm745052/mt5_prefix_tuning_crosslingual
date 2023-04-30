
import torch
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils.data import DataLoader
from datasets.arrow_dataset import Dataset as HFDataset
from datasets.load import load_metric, load_dataset
from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorForTokenClassification, BertConfig
from transformers import default_data_collator, EvalPrediction
import numpy as np
def Convert_tydia_to_squad(example):
    new_example = {}
    new_example['id'] = example['paragraphs'][0]['qas'][0]['id']
    new_example['title'] = example['paragraphs'][0]['qas'][0]['id']   # example['paragraphs'][0]['title']
    new_example['context'] = example['paragraphs'][0]['context']
    new_example['question'] = example['paragraphs'][0]['qas'][0]['question']
    new_example['answers'] = {}
    new_example['answers']['text'] = [example['paragraphs'][0]['qas'][0]['answers'][0]['text']]
    new_example['answers']['answer_start'] = [example['paragraphs'][0]['qas'][0]['answers'][0]['answer_start']]
    return new_example



def load_tydiqa(path):
    '''
    return splitted data [train and validation]
    '''
    raw_datasets = load_dataset('json', data_files={'train': path}, field='data')
            # {'paragraphs': [{'context': 'Quantum field theory naturally began with the study of electromagnetic interactions, as the electromagnetic field was the only known classical field as of the 1920s.[8]:1', 'qas': [{'answers': [{'answer_start': 159, 'text': '1920s'}], 'id': '12', 'question': 'When was quantum field theory developed?'}]}]}
    raw_datasets['train'] = raw_datasets['train'].map(Convert_tydia_to_squad)
    column_names = raw_datasets['train'].column_names
    print(column_names)
    return raw_datasets

