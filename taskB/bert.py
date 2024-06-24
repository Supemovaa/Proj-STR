import pandas as pd
from transformers import AutoModel, AutoTokenizer, BertPreTrainedModel
import numpy as np
from scipy.stats import spearmanr
import torch
from tqdm import tqdm

test_path = "../Semantic_Relatedness_SemEval2024/Track B/eng/eng_test_with_labels.csv"
test_data = pd.read_csv(test_path, usecols=["Text", "Score"])
gt = test_data["Score"].to_numpy()

# 0.735
tokenizer = AutoTokenizer.from_pretrained('/home/maty/models/distilbert-base-uncased')
model = AutoModel.from_pretrained('/home/maty/models/distilbert-base-uncased')

all_test = test_data['Text'].apply(lambda x: x.split('\n')).tolist()

pred = []
for i in tqdm(range(len(all_test))):
    sen1, sen2 = all_test[i]
    encoded1 = tokenizer(sen1, return_tensors='pt', return_attention_mask=False)
    encoded2 = tokenizer(sen2, return_tensors='pt', return_attention_mask=False)
    out1 = model(**encoded1).last_hidden_state.detach()
    out2 = model(**encoded2).last_hidden_state.detach()
    vec1 = out1.squeeze(0).mean(dim=0)
    vec2 = out2.squeeze(0).mean(dim=0)
    score = np.dot(vec1, vec2) / np.linalg.norm(vec1, 2) / np.linalg.norm(vec2, 2)
    pred.append(score)
pred = np.array(pred)

print(spearmanr(pred, gt))