import pandas as pd
from scipy.stats import spearmanr
import numpy as np
from training_config import LANG, config
import re

if __name__ == '__main__':
    s = 0
    for lang in LANG.keys():
        test_data = pd.read_csv(config.get_test_path(lang), usecols=['s0', 's1', 'Score'])
        gt = test_data['Score'].to_numpy()
        pred = []
        for i in range(len(test_data)):
            s0 = re.sub(r'[^\w\s]', ' ', test_data['s0'].iloc[i])
            s1 = re.sub(r'[^\w\s]', ' ', test_data['s1'].iloc[i])
            score = 2 * len(set(s0).intersection(set(s1))) / (len(set(s0)) + len(set(s1)))
            pred.append(score)
        pred = np.array(pred)
        spearman = spearmanr(pred, gt)[0]
        print(f"{lang}: {spearman}")
        print('=========================================')
        s += spearman
    print(f"avg spearman: {s / 8}")