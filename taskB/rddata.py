import pandas as pd
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
import torch
import pickle as pkl
import argparse

dev_path = "../../Semantic_Relatedness_SemEval2024/Track B/eng/eng_dev_with_labels.csv"
test_path = "../../Semantic_Relatedness_SemEval2024/Track B/eng/eng_test_with_labels.csv"
batch_size = 64
max_length = 20
workers = 6

class MySet(Dataset):
    def __init__(self, rawdata, tokenizer: spm.SentencePieceProcessor, max_length=20):
        super().__init__()
        self.sen = []
        self.score = []
        for i in range(len(rawdata)):
            self.score.append(rawdata.iloc[i]["Score"])
            sen1, sen2 = rawdata["Text"].iloc[i][0], rawdata["Text"].iloc[i][1]
            tok1 = MySet.encode(tokenizer, sen1, max_length)
            tok2 = MySet.encode(tokenizer, sen2, max_length)
            self.sen.append({
                'sample0': torch.tensor(tok1),
                'sample1': torch.tensor(tok1)
            })
            self.sen.append({
                'sample0': torch.tensor(tok2),
                'sample1': torch.tensor(tok2)
            })
        self.score = torch.tensor(self.score, dtype=torch.float32)
    
    @staticmethod
    def encode(tokenizer, x, max_length):
        tok = tokenizer.EncodeAsIds(x)
        if len(tok) < max_length:
            tok = tok + [0] * (max_length - len(tok))
        elif len(tok) > max_length:
            tok = tok[:max_length]
        return tok
    def __getitem__(self, index):
        return self.sen[index]
    def __len__(self):
        return len(self.score)
    
    def collate_fn(self, batch):
        text = []
        for obj in batch:
            text.append(obj['sample0'])
            text.append(obj['sample1'])
        text = torch.stack(text)
        return text

class MyTestSet(Dataset):
    def __init__(self, rawdata, tokenizer: spm.SentencePieceProcessor, max_length=20):
        super().__init__()
        self.sen1, self.sen2 = [], []
        self.score = []
        for i in range(len(rawdata)):
            self.score.append(rawdata.iloc[i]["Score"])
            sen1, sen2 = rawdata["Text"].iloc[i][0], rawdata["Text"].iloc[i][1]
            tok1 = MySet.encode(tokenizer, sen1, max_length)
            tok2 = MySet.encode(tokenizer, sen2, max_length)
            self.sen1.append(torch.tensor(tok1))
            self.sen2.append(torch.tensor(tok2))
        self.score = torch.tensor(self.score, dtype=torch.float32)
    
    def __getitem__(self, index):
        return self.sen1[index], self.sen2[index], self.score[index]
    def __len__(self):
        return len(self.score)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wtrain', type=int, default=0)
    args = parser.parse_args()

    dev_data = pd.read_csv(dev_path, usecols=["Text", "Score"])
    test_data = pd.read_csv(test_path, usecols=["Text", "Score"])

    if args.wtrain == 1:
        with open('./tmp/corpus.txt', 'w') as f:
            for s in dev_data["Text"]:
                s1, s2 = s.lower().strip().split('\n')
                f.write(s1 + '\n' + s2 + '\n')
        spm.SentencePieceTrainer.Train(input='./tmp/corpus.txt', 
                                       model_prefix='spm', 
                                       vocab_size=1600)

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load('spm.model')
    dev_data['Text'] = dev_data['Text'].apply(lambda x: x.lower().strip().split('\n'))
    test_data['Text'] = test_data['Text'].apply(lambda x: x.lower().strip().split('\n'))

    dev_set = MySet(dev_data, tokenizer)
    dev_loader = DataLoader(dev_set, batch_size=batch_size, shuffle=True, num_workers=workers, collate_fn=dev_set.collate_fn)
    test_loader = DataLoader(MyTestSet(test_data, tokenizer), batch_size=batch_size, shuffle=False, num_workers=workers) 
    
    with open("./tmp/dev_loader.pkl", 'wb') as f:
        pkl.dump(dev_loader, f)
    with open('./tmp/test_loader.pkl', 'wb') as f:
        pkl.dump(test_loader, f)

    for batch in dev_loader:
        print(batch)
        break