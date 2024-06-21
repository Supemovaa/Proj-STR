from preprocess import DataModule
from model import BERTModel
import lightning as L
import torch
from scipy.stats import spearmanr
from training_config import config

datamodule = DataModule(config.model_path, 8, config.num_workers)
datamodule.setup('predict')
model = BERTModel.load_from_checkpoint('/home/maty/proj/taskB/SemEval-2024-Task2-distilbert-base-uncased-lr=1e-05/epoch=24-step=4300.ckpt')
model.eval()
trainer = L.Trainer(
    accelerator='gpu',
    devices=1,
    num_nodes=1,
    precision='16-mixed',
    enable_progress_bar=True,
)
out = trainer.predict(model, datamodule.test_dataloader())
prediction = torch.cat(out, dim=0).view(-1)
reference = torch.tensor([x['Score'] for x in datamodule.test])

print(spearmanr(prediction.numpy(), reference.numpy()))