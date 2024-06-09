import torch
import torch.nn as nn
from transformers import AutoModel
import lightning as L
from scipy.stats import spearmanr
import numpy as np
from training_config import config

# 0.633
class lstmModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.emb = nn.Embedding(10000, embedding_dim)
        self.lstm = nn.GRU(embedding_dim, hidden_dim, 
                            num_layers=4, batch_first=True, 
                            dropout=0.5, bidirectional=True)
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.linear = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.fc = nn.Sequential(
            nn.Linear(256 * 3, 256 * 3),
            nn.ReLU(),
            nn.Linear(256 * 3, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

    def encode(self, x):
        x = self.emb(x)
        s, _ = self.lstm(x)
        out = self.linear(torch.mean(self.norm(s), dim=1))
        return out

    def forward(self, x, y):
        s1 = self.encode(x)
        s2 = self.encode(y)
        out = self.cos(s1, s2)
        return out
    
class BERTModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(config.model_path)
        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.criterion = nn.MSELoss()

    def forward(self, x, y):
        s1 = self.encoder(**x).last_hidden_state.mean(dim=1)
        s2 = self.encoder(**y).last_hidden_state.mean(dim=1)
        out = self.cos(s1, s2)
        return out
    
    def compute_metrics(self, y, yhat, mode):
        score = spearmanr(y.cpu().detach().numpy(), yhat.cpu().detach().numpy())[0]
        if np.isnan(score) or np.isinf(score) or score < 0 or score > 1:
            score = 0.
        log_dict = {}
        log_dict[f"{mode}/correlation"] = score
        return log_dict

    def training_step(self, batch, batch_idx):
        s0, s1, score = batch
        outputs = self(s0, s1)
        loss = self.criterion(outputs, score)
        log_dict = self.compute_metrics(score.view(-1), outputs.view(-1), 'train')
        log_dict['train/loss'] = loss.item()
        self.log_dict(log_dict, prog_bar=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        s0, s1, score = batch
        outputs = self(s0, s1)
        loss = self.criterion(outputs, score)
        log_dict = self.compute_metrics(score.view(-1), outputs.view(-1), 'val')
        log_dict['val/loss'] = loss.item()
        self.log_dict(log_dict, prog_bar=True, sync_dist=True)
        return loss
    
    def predict_step(self, batch, batch_idx):
        text1, text2, score = batch
        y_hat = self(text1, text2)
        return y_hat.view(-1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),
                                lr=config.lr, 
                                weight_decay=config.weight_decay)

class Accumulater:
    def __init__(self, gap):
        self.sum = 0
        self.num = 0
        self.gap = gap
    @property
    def avg(self):
        return self.sum / self.num
    def increment(self, inputs):
        self.sum = self.sum + inputs * self.gap
        self.num += self.gap
    def clear(self):
        self.sum = 0
        self.num = 0