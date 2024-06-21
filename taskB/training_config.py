class BERTConfig():
    def __init__(self):
        self.seed = 3407
        self.wandb_name = 'SemEval-2024-Task2'
        self.model_name = 'distilbert-base-uncased'
        self.model_path = f'/home/maty/models/{self.model_name}'
        self.weight_decay = 1e-5
        self.lr = 1e-5
        self.max_epoches = 25
        self.batch_size = 8
        self.grad_accumulate = 2
        self.train_path = "../Semantic_Relatedness_SemEval2024/Track A/eng/eng_train.csv"
        self.dev_path = "../Semantic_Relatedness_SemEval2024/Track B/eng/eng_dev_with_labels.csv"
        self.test_path = "../Semantic_Relatedness_SemEval2024/Track B/eng/eng_test_with_labels.csv"
        self.max_length = 20
        self.num_workers = 12
        # self.strategy = 'ddp_find_unused_parameters_true'
        self.strategy = 'ddp'

config = BERTConfig()