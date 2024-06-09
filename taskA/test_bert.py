from preprocess import DataModule
from model import BERTModel
import lightning as L
import torch
from scipy.stats import spearmanr
from training_config import config, LANG

datamodule = DataModule(config.model_path, 8, config.num_workers)

model_path = '/home/maty/proj/taskA/lightning_logs/edolji0r/checkpoints/epoch=14-step=3180.ckpt'
model = BERTModel.load_from_checkpoint(model_path)
model.eval()

s = 0
for lang in LANG.keys():
    datamodule.setup('predict', lang)

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

    sp, p = spearmanr(prediction.numpy(), reference.numpy())
    print(f"{lang}: spearman={sp}, p_value={p}")
    print('=============================================')
    s += sp
print(f'avg spearman: {s / 8}')