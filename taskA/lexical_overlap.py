import pandas as pd
from scipy.stats import spearmanr
import numpy as np
import re
from training_config import LANG, config

# test_path = "../../Semantic_Relatedness_SemEval2024/Track A/eng/eng_test_with_labels.csv"
# test_data = pd.read_csv(test_path, usecols=["Text", "Score"])

# gt = test_data["Score"].to_numpy()
# pred = []
# for i in range(len(test_data)):
#     sens = test_data["Text"].iloc[i].lower()
#     sens = re.sub(r'[^\w\s]', ' ', sens).split('\n')
#     sen1 = sens[0].split()
#     sen2 = sens[1].split()
#     score = 2 * len(set(sen1).intersection(set(sen2))) / (len(set(sen1)) + len(set(sen2)))
#     pred.append(score)
# pred = np.array(pred)

if __name__ == '__main__':
    s = 0
    for lang in LANG.keys():
        test_data = pd.read_csv(config.get_test_path(lang), usecols=['s0', 's1', 'Score'])
        gt = test_data['Score'].to_numpy()
        pred = []
        for i in range(len(test_data)):
            s0 = test_data['s0'].iloc[i]
            s1 = test_data['s1'].iloc[i]
            # score = 2 * len(set(s0).intersection(set(s1))) / (len(set(s0)) + len(set(s1)))
            score = 2 * len(set(s0).intersection(set(s1))) / len(set(s0).union(set(s1)))

            pred.append(score)
        pred = np.array(pred)
        spearman = spearmanr(pred, gt)[0]
        print(f"{lang}:, {spearman}")
        print('=========================================')
        s += spearman
    print(f"avg spearman: {s / 8}")