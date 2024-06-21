import pandas as pd
from scipy.stats import spearmanr
import numpy as np
from gensim.models import word2vec, fasttext, KeyedVectors
import gensim.downloader as api
from itertools import chain
import re

dev_path = "../../Semantic_Relatedness_SemEval2024/Track B/eng/eng_dev.csv"
test_path = "../../Semantic_Relatedness_SemEval2024/Track B/eng/eng_test_with_labels.csv"
test_data = pd.read_csv(test_path, usecols=["Text", "Score"])
dev_data = pd.read_csv(dev_path)
gt = test_data["Score"].to_numpy()

def proc(x):
    x = re.sub(r'[^\w\s]', ' ', x.lower())
    s1, s2 = x.split('\n')
    return [s1.split(), s2.split()]

all_dev = dev_data["Text"].apply(proc).tolist()
all_test = test_data["Text"].apply(proc).tolist()
all_dev = list(chain.from_iterable(all_dev))

# model = fasttext.FastText(all_dev)
# model = word2vec.Word2Vec(all_dev, vector_size=100, min_count=1)
# 0.565
# model = KeyedVectors.load_word2vec_format('./tmp/glove-wiki-gigaword-50.gz')
# 0.665
model = api.load('word2vec-google-news-300')

def get_mean_vector(sen, dim):
    vec = np.zeros((dim, ))
    l = 0
    for w in sen:
        try:
            vec += np.array(model[w])
        except:
            pass
        l += 1
    return vec / l

# mean
pred = []
for i in range(len(all_test)):
    sen1, sen2 = all_test[i]
    vec1 = get_mean_vector(sen1, 300)
    vec2 = get_mean_vector(sen2, 300)
    if np.linalg.norm(vec1, 2) == 0 or np.linalg.norm(vec2, 2) == 0:
        score = 0.
        if np.linalg.norm(vec1, 2) == 0:
            print(sen1)
        if np.linalg.norm(vec2, 2) == 0:
            print(sen2)
    else:
        score = np.dot(vec1, vec2) / np.linalg.norm(vec1, 2) / np.linalg.norm(vec2, 2)
    pred.append(score)
pred = np.array(pred)

print(spearmanr(pred, gt))