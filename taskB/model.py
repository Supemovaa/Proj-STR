import torch
import torch.nn as nn
from transformers import AutoModel
import lightning as L
from scipy.stats import spearmanr
import numpy as np
from training_config import config
import torch.nn.functional as F

# 0.6473
class lstmModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.emb = nn.Embedding(1600, embedding_dim)
        self.lstm = nn.GRU(embedding_dim, hidden_dim, 
                            num_layers=2, batch_first=True, 
                            dropout=0.5, bidirectional=True)
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.linear = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

    def forward(self, x):
        x = self.emb(x)
        s, _ = self.lstm(x)
        out = self.linear(torch.mean(self.norm(s), dim=1))
        return out
    
class Criterion(nn.Module):
    def __init__(self, temp=0.05):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.temp = temp
    def forward(self, x):
        labels = torch.arange(x.shape[0], device=x.device)
        labels = (labels - labels % 2 * 2) + 1
        sim = F.cosine_similarity(x.unsqueeze(0), x.unsqueeze(1), dim=-1)
        sim = sim - torch.eye(x.shape[0], device=x.device) * 1e12
        sim = sim / self.temp
        loss = self.ce(sim, labels)
        return loss

# 0.746
class BERTModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(config.model_path)
        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.criterion = Criterion()

    def forward(self, x):
        out = self.encoder(**x).last_hidden_state.mean(dim=1)
        return out
    
    def compute_metrics(self):
        pass

    def training_step(self, batch, batch_idx):
        s = batch
        outputs = self(s)
        loss = self.criterion(outputs)
        log_dict = {}
        log_dict['train/loss'] = loss.item()
        self.log_dict(log_dict, prog_bar=True, sync_dist=True)
        return loss
    
    def predict_step(self, batch, batch_idx):
        text1, text2, score = batch
        encode1 = self(text1)
        encode2 = self(text2)
        y_hat = F.cosine_similarity(encode1, encode2, dim=-1)
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