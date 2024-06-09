import pandas as pd
from torch.utils.data import DataLoader
import torch
import lightning as L
from transformers import AutoTokenizer
from training_config import config

LANG = {
    'amh': 'amh_Ethi',
    'arq': 'ary_Arab',
    'ary': 'ary_Arab',
    'eng': 'eng_Latn',
    'hau': 'hau_Latn',
    'kin': 'kin_Latn',
    'mar': 'mar_Deva',
    'tel': 'tel_Telu'
}

class DataModule(L.LightningDataModule):
    def __init__(self, model_path, batch_size=8, num_workers=12):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
    def prepare_data(self):
        pass
    def load_data(self, data_path):
        data = pd.read_csv(data_path, usecols=['s0', 's1', 'Score'])
        ret = []
        for i in range(len(data)):
            ret.append({
                's0': data['s0'].iloc[i],
                's1': data['s1'].iloc[i],
                'Score': data['Score'].iloc[i]
            })
        del data
        return ret
    
    def setup(self, stage, lang=None):
        if stage == 'fit':
            self.train = []
            for lang in LANG.keys():
                self.train += self.load_data(config.get_train_path(lang))
            self.dev = []
            for lang in LANG.keys():
                self.dev += self.load_data(config.get_dev_path(lang))
        if stage == 'predict':
            assert lang is not None
            self.test = self.load_data(config.get_test_path(lang))
        
    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
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
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
        )
    
    def collate_fn(self, batch):
        text1 = [obj["s0"] for obj in batch]
        text2 = [obj["s1"] for obj in batch]
        score = [obj["Score"] for obj in batch]

        text1 = self.tokenizer(text1, return_tensors="pt", padding=True, 
                               truncation=True)
        text2 = self.tokenizer(text2, return_tensors="pt", padding=True, 
                               truncation=True)
        score = torch.tensor(score, dtype=torch.float32)

        return text1, text2, score
    
if __name__ == '__main__':
    datamodule = DataModule(config.model_path, config.batch_size, config.num_workers)
    datamodule.prepare_data()
    datamodule.setup('fit')
    for batch in datamodule.train_dataloader():
        s0, s1, score = batch
        print(s0.keys())
        print(s0.input_ids.shape)
        print(s1.input_ids.shape)
        print(score.shape)
        break