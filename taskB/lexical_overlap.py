import pandas as pd
from scipy.stats import spearmanr
import numpy as np
import re

if __name__ == '__main__':
    s = 0
    path = f'/home/maty/proj/Semantic_Relatedness_SemEval2024/Track B/eng/eng_test_with_labels.csv'
    test_data = pd.read_csv(path, usecols=['Text', 'Score'])
    gt = test_data['Score'].to_numpy()
    pred = []
    for i in range(len(test_data)):
        # sen = re.sub(r'[^\w\s]', ' ',  test_data['Text'].iloc[i].lower())
        sen = test_data['Text'].iloc[i].lower()
        s0, s1 = sen.split('\n')
        # score = 2 * len(set(s0).intersection(set(s1))) / (len(set(s0)) + len(set(s1)))
        score = 2 * len(set(s0).intersection(set(s1))) / len(set(s0).union(set(s1)))
        pred.append(score)
    pred = np.array(pred)
    spearman = spearmanr(pred, gt)[0]
    print(spearman)