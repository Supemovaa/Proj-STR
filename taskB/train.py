import torch
import torch.nn as nn
from model import lstmModel, Accumulater, Criterion
import sentencepiece as spm
import numpy as np
import pickle as pkl
from rddata import batch_size, MySet, MyTestSet
import argparse
from scipy.stats import spearmanr
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = 'cuda:3'

def draw_fig(train_loss, epochs):
    fig, ax1 = plt.subplots()

    # 绘制loss曲线
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='red')
    ax1.plot(epochs, train_loss, 'r-', label='Train Loss')
    ax1.tick_params(axis='y', labelcolor='red')

    # 为了图例正确显示，我们需要合并图例
    lines, labels = ax1.get_legend_handles_labels()
    ax1.legend(lines, labels, loc='upper left')

    plt.title('Loss over Epochs')
    plt.savefig('./fig.png')


def train_step(model, loader, criterion, optimizer, loss_writer):
    for s in loader:
        s = s.to(device)
        model.train()
        outputs = model(s)
        loss = criterion(outputs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_writer.increment(loss.item())
    return loss_writer.avg


def train(model, loader, epochs):
    criterion = Criterion(temp=0.05)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=50, gamma=0.5)
    loss_writer = Accumulater(batch_size)
    tl = []
    for epoch in range(epochs):
        train_loss = train_step(model, loader, criterion, optimizer, loss_writer)
        loss_writer.clear()
        print(f"epoch {epoch + 1}: train loss = {train_loss}\n-------------------\n")
        tl.append(train_loss)
        # scheduler.step()
    draw_fig(tl, np.arange(epochs) + 1)

@torch.inference_mode()
def final_test(model, loader):
    all_p = []
    all_s = []
    model.eval()
    for s1, s2, score in loader:
        s1, s2, score = s1.to(device), s2.to(device), score.to(device)
        encoded1 = model(s1)
        encoded2 = model(s2)
        outputs = F.cosine_similarity(encoded1, encoded2, dim=-1)
        all_p.append(outputs.cpu().detach())
        all_s.append(score.cpu().detach())
    all_p = torch.concat(all_p, dim=0).numpy()
    all_s = torch.concat(all_s, dim=0).numpy()
    spearm = spearmanr(all_p, all_s)
    return spearm

if __name__ == '__main__':
    torch.manual_seed(3407)
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', type=int, default=0)
    args = parser.parse_args()

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load('spm.model')
    model = lstmModel(embedding_dim=512, hidden_dim=768).to(device)

    with open('./tmp/test_loader.pkl', 'rb') as f:
        tloader = pkl.load(f)
    with open('./tmp/dev_loader.pkl', 'rb') as f:
        dloader = pkl.load(f)
    if args.t == 1:
        train(model, dloader, 500)
    print(final_test(model, tloader))