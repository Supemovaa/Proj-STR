import pandas as pd
import os
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import sentencepiece as spm
import pickle as pkl
import torch
import os
from training_config import LANG

class MySet(Dataset):
    def __init__(self, rawdata, tokenizer: spm.SentencePieceProcessor, max_length=90):
        self.data = []
        for i in range(len(rawdata)):
            s1, s2 = rawdata['Text'].iloc[i].lower().split('\n')
            tok1 = MySet.encode(tokenizer, s1, max_length)
            tok2 = MySet.encode(tokenizer, s2, max_length)
            self.data.append({
                's1': tok1,
                's2': tok2,
                'score': rawdata['Score'].iloc[i]
			})

    @staticmethod
    def encode(tokenizer, x, max_length):
        tok = tokenizer.EncodeAsIds(x)
        if len(tok) < max_length:
            tok = tok + [0] * (max_length - len(tok))
        elif len(tok) > max_length:
            tok = tok[:max_length]
        return tok
    
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return len(self.data)

def collate_fn(batch):
    s1 = torch.tensor([obj['s1'] for obj in batch])
    s2 = torch.tensor([obj['s2'] for obj in batch])
    score = torch.tensor([obj['score'] for obj in batch], dtype=torch.float32)
    return s1, s2, score

if __name__ == '__main__':
    base_path = '/home/maty/proj/Semantic_Relatedness_SemEval2024/Track A'
    all_text = []
    cnt = 0
    for lang in LANG:
        corpus_path = os.path.join(base_path, f'{lang}/{lang}_train.csv')
        data = pd.read_csv(corpus_path, usecols=['Text', 'Score'])
        sentences = data['Text'].apply(lambda x: x.lower().split('\n')).explode()
        all_text.append(sentences)
        cnt += len(sentences)
    assert cnt == 27122
    pd.concat(all_text, axis=0).to_csv('./tmp/all_corpus.txt', header=False, index=False)

    spm.SentencePieceTrainer.Train(input='./tmp/all_corpus.txt', model_prefix='spm', vocab_size=20000)
    os.system('mv spm.model ./tmp/')
    os.system('mv spm.vocab ./tmp/')

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load('./tmp/spm.model')

    all_train, all_dev, all_test = [], [], []
    for lang in LANG:
        train_path = os.path.join(base_path, f'{lang}/{lang}_train.csv')
        dev_path = os.path.join(base_path, f'{lang}/{lang}_dev_with_labels.csv')
        test_path = os.path.join(base_path, f'{lang}/{lang}_test_with_labels.csv')
        traindata = pd.read_csv(train_path, usecols=['Text', 'Score'])
        devdata = pd.read_csv(dev_path, usecols=['Text', 'Score'])
        testdata = pd.read_csv(test_path, usecols=['Text', 'Score'])
        all_train.append(MySet(traindata, tokenizer))
        all_dev.append(MySet(devdata, tokenizer))
        all_test.append(MySet(testdata, tokenizer))
    trainloader = DataLoader(ConcatDataset(all_train), batch_size=64, shuffle=True, num_workers=12, collate_fn=collate_fn)
    devloader = DataLoader(ConcatDataset(all_dev), batch_size=64, shuffle=False, num_workers=12, collate_fn=collate_fn)
    
    testloaders = dict()
    for i in range(len(LANG)):
        testloaders.setdefault(
            list(LANG.keys())[i], DataLoader(all_test[i], batch_size=64, shuffle=False, num_workers=12, collate_fn=collate_fn)
        )

    with open('./tmp/train_loader.pkl', 'wb') as f:
        pkl.dump(trainloader, f)
    with open('./tmp/dev_loader.pkl', 'wb') as f:
        pkl.dump(devloader, f)
    with open('./tmp/test_loaders.pkl', 'wb') as f:
        pkl.dump(testloaders, f)