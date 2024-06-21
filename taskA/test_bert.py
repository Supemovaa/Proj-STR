from preprocess import DataModule
from model import BERTModel
import lightning as L
import torch
from scipy.stats import spearmanr
from training_config import config, LANG

datamodule = DataModule(config.model_path, 8, config.num_workers)

for i in range(20):
    model_path = f'/home/maty/proj/taskA/original-distilroberta-lr=1e-05/epoch={i}-step={212 * (i+1)}.ckpt'
    model = BERTModel.load_from_checkpoint(model_path)
    model.eval()

    score = {}
    s = 0
    for lang in LANG.keys():
        datamodule.setup('predict', lang)

        trainer = L.Trainer(
            accelerator='gpu',
            devices=1,
            num_nodes=1,
            precision='16-mixed',
            enable_progress_bar=False,
        )
        out = trainer.predict(model, datamodule.test_dataloader())
        prediction = torch.cat(out, dim=0).view(-1)
        reference = torch.tensor([x['Score'] for x in datamodule.test])

        sp, p = spearmanr(prediction.numpy(), reference.numpy())
        score.setdefault(lang, (sp, p))

    for k, v in score.items():
        print(f"{k}: spearman={v[0]}, p_value={v[1]}")
        print('=============================================')
        s += v[0]
    print(f'avg spearman: {s / 8}')
    print('*********************************************************************')