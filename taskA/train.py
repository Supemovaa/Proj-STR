import torch
import torch.nn as nn
from model import lstmModel, Accumulater
import sentencepiece as spm
import numpy as np
import pickle as pkl
from rddata import MySet, collate_fn
import argparse
from scipy.stats import spearmanr
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = 'cuda'

def plot_metrics(train_loss, train_corr, dev_loss, dev_corr, epochs):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='tab:blue')
    ax1.plot(range(epochs), train_loss, label='Train Loss', color='blue', marker='o')
    ax1.plot(range(epochs), dev_loss, label='Dev Loss', color='red', marker='o')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()  # 创建第二个纵轴
    ax2.set_ylabel('Correlation', color='tab:green')
    ax2.plot(range(epochs), train_corr, label='Train Corr', color='green', marker='o')
    ax2.plot(range(epochs), dev_corr, label='Dev Corr', color='orange', marker='o')
    ax2.tick_params(axis='y', labelcolor='tab:green')
    ax2.legend(loc='upper right')

    fig.tight_layout()  # 调整图表布局以适应标签
    plt.grid(True)
    plt.savefig('testpic.png')

def train_step(model, loader, criterion, optimizer, loss_writer, corr_writer):
    for s1, s2, score in loader:
        s1, s2, score = s1.to(device), s2.to(device), score.to(device)
        model.train()
        outputs = model(s1, s2)
        loss = criterion(outputs, score)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_writer.increment(loss.item())
        corr_writer.increment(spearmanr(outputs.detach().cpu().numpy(), score.detach().cpu().numpy())[0])
    return loss_writer.avg, corr_writer.avg

@torch.inference_mode()
def test_step(model, loader, criterion, loss_writer, corr_writer):
    model.eval()
    for s1, s2, score in loader:
        s1, s2, score = s1.to(device), s2.to(device), score.to(device)
        outputs = model(s1, s2)
        loss = criterion(outputs, score)
        loss_writer.increment(loss.item())
        corr_writer.increment(spearmanr(outputs.cpu().numpy(), score.cpu().numpy())[0])
    return loss_writer.avg, corr_writer.avg

@torch.inference_mode()
def final_test(model, loaders):
    model.eval()
    for lang, loader in loaders.items():
        prediction, reference = [], []
        for s1, s2, score in loader:
            s1, s2, score = s1.to(device), s2.to(device), score.to(device)
            outputs = model(s1, s2)
            prediction.append(outputs.cpu().numpy())
            reference.append(score.cpu().numpy())
        prediction = np.concatenate(prediction, axis=0)
        reference = np.concatenate(reference, axis=0)
        print(f"{lang}: correlation = {spearmanr(prediction, reference)[0]}")
    pass

def train(model, train_loader, dev_loader, epochs):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.5)
    loss_writer = Accumulater(64)
    corr_writer = Accumulater(64)
    tl, tc, dl, dc = [], [], [], []
    for epoch in range(epochs):
        train_loss, train_corr = train_step(model, train_loader, criterion, optimizer, loss_writer, corr_writer)
        loss_writer.clear()
        corr_writer.clear()
        dev_loss, dev_corr = test_step(model, dev_loader, criterion, loss_writer, corr_writer)
        loss_writer.clear()
        corr_writer.clear()
        print(f"epoch {epoch + 1}: train loss = {train_loss}, train corr = {train_corr}, dev loss = {dev_loss}, dev corr = {dev_corr}\n-------------------\n")
        tl.append(train_loss)
        tc.append(train_corr)
        dl.append(dev_loss)
        dc.append(dev_corr)
        scheduler.step()
    plot_metrics(tl, tc, dl, dc, epochs)

if __name__ == '__main__':
    torch.manual_seed(3407)
    parser = argparse.ArgumentParser()
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load('./tmp/spm.model')
    model = lstmModel(embedding_dim=512, hidden_dim=768).to(device)
    with open('./tmp/train_loader.pkl', 'rb') as f:
        trloader = pkl.load(f)
    with open('./tmp/test_loaders.pkl', 'rb') as f:
        tloaders = pkl.load(f)
    with open('./tmp/dev_loader.pkl', 'rb') as f:
        dloader = pkl.load(f)
    train(model, trloader, dloader, 60)
    final_test(model, tloaders)