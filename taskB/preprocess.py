import pandas as pd
from torch.utils.data import DataLoader
import torch
import lightning as L
from transformers import AutoTokenizer, DebertaV2Tokenizer
from training_config import config

class DataModule(L.LightningDataModule):
    def __init__(self, model_path, batch_size=8, num_workers=12):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
    def prepare_data(self):
        pass
    def load_data(self, data_path, stage):
        data = pd.read_csv(data_path, usecols=['Text', 'Score'])
        data['Text'] = data['Text'].apply(lambda x: x.lower().strip().split('\n'))
        data['s0'] = data['Text'].apply(lambda x: x[0])
        data['s1'] = data['Text'].apply(lambda x: x[1])
        
        if stage == 'fit':
            ret = []
            for i in range(len(data)):
                ret.append({
                    'sample1': data['s0'].iloc[i],
                    'sample2': data['s0'].iloc[i]
                })
                ret.append({
                    'sample1': data['s1'].iloc[i],
                    'sample2': data['s1'].iloc[i]
                })
            del data
            return ret
        elif stage == 'predict':
            ret = []
            for i in range(len(data)):
                ret.append({
                    's0': data['s0'].iloc[i],
                    's1': data['s1'].iloc[i],
                    'Score': data['Score'].iloc[i]
                })
            del data
            return ret
        return
    
    def setup(self, stage):
        if stage == 'fit':
            # self.dev = self.load_data(config.dev_path, stage)
            self.dev = self.load_data(config.train_path, stage)
        if stage == 'predict':
            self.test = self.load_data(config.test_path, stage)
        
    def train_dataloader(self):
        return DataLoader(
            self.dev,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn_test,
            num_workers=self.num_workers,
        )
    
    def collate_fn_test(self, batch):
        text1 = [obj["s0"] for obj in batch]
        text2 = [obj["s1"] for obj in batch]
        score = [obj["Score"] for obj in batch]

        text1 = self.tokenizer(text1, return_tensors="pt", padding=True, 
                               truncation=True)
        text2 = self.tokenizer(text2, return_tensors="pt", padding=True, 
                               truncation=True)
        score = torch.tensor(score, dtype=torch.float32)

        return text1, text2, score
    
    def collate_fn(self, batch):
        text = []
        for obj in batch:
            text.append(obj['sample1'])
            text.append(obj['sample2'])
        text = self.tokenizer(text, return_tensors='pt', padding=True, 
                              truncation=True)
        return text
    
if __name__ == '__main__':
    datamodule = DataModule(config.model_path, 2, config.num_workers)
    datamodule.prepare_data()
    datamodule.setup('fit')
    for batch in datamodule.train_dataloader():
        s = batch
        for i in range(s.input_ids.shape[0]):
            print(datamodule.tokenizer.decode(s.input_ids[i, :]))
        break
        