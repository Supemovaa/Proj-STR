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

class RoBERTaConfig():
    def __init__(self):
        self.seed = 3407
        self.wandb_name = 'SemEval-2024-Task1'
        self.model_name = 'distilroberta'
        self.model_path = f'/home/maty/models/{self.model_name}'
        self.weight_decay = 1e-4
        self.lr = 1e-5
        self.max_epoches = 15
        self.batch_size = 8
        self.max_length = 20
        self.num_workers = 12
        self.strategy = 'ddp_find_unused_parameters_true'

    def get_train_path(self, lang):
        return f"../data/{lang}/{lang}_train.csv"
    def get_dev_path(self, lang):
        return f"../data/{lang}/{lang}_dev_with_labels.csv"
    def get_test_path(self, lang):
        return f"../data/{lang}/{lang}_test_with_labels.csv"

config = RoBERTaConfig()