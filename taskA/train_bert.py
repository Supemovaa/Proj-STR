from preprocess import DataModule
from model import BERTModel
import lightning as L
from pytorch_lightning.loggers import WandbLogger
from training_config import config
import warnings
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint

L.seed_everything(config.seed)
warnings.filterwarnings('ignore')
L.seed_everything(config.seed)
datamodule = DataModule(config.model_path, config.batch_size, config.num_workers)
datamodule.setup('fit')
model = BERTModel()
trainer = L.Trainer(
    accelerator='gpu',
    strategy=config.strategy,
    devices=[0, 1, 2, 3],
    logger=WandbLogger(name=config.wandb_name),
    accumulate_grad_batches=config.grad_accumulate,
    max_epochs=config.max_epoches,
    check_val_every_n_epoch=1,
    precision='16-mixed',
    enable_progress_bar=True,
    enable_checkpointing=True,
    callbacks=[ModelCheckpoint(dirpath=f'./{config.wandb_name}-{config.model_name}-lr={config.lr}/', every_n_epochs=1, save_top_k=-1)]
)
trainer.fit(model, datamodule)