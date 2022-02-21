import sys
import os
import json
import wandb
import random
import pytorch_lightning as pl
sys.path.append(os.getcwd())
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from config.config import train_line_parser
from drought_impact_forecasting.models.LSTM_model import LSTM_model
from Data.data_preparation import Earth_net_DataModule
from callbacks import WandbTrain_callback
from datetime import datetime

def main():
    # load configs
    cfg = train_line_parser()

    if not cfg["training"]["offline"]:
        os.environ["WANDB_MODE"]="online"
    else:
        os.environ["WANDB_MODE"]="offline"

    wandb.init(entity="eth-ds-lab", project="Drought Impact Forecasting", config=cfg)

    # store the model name to wandb
    with open(os.path.join(wandb.run.dir, "run_name.txt"), 'w') as f:
        try:
            f.write(wandb.run.name)
        except:
            f.write("offline_run_" + str(datetime.now()))

    # setup model
    if cfg["training"]["checkpoint"] is not None:
        # Resume training from checkpoint
        model = LSTM_model.load_from_checkpoint(cfg["training"]["checkpoint"])
        cfg = model.cfg
    else:
        model = LSTM_model(cfg)

    with open(os.path.join(wandb.run.dir, "Training.json"), 'w') as fp:
        json.dump(cfg, fp)
    
    wandb_logger = WandbLogger(project='DS_Lab', config=cfg, job_type='train', offline=True)
    
    random.seed(cfg["training"]["seed"])
    pl.seed_everything(cfg["training"]["seed"], workers=True)

    EN_dataset = Earth_net_DataModule(data_dir = cfg["data"]["pickle_dir"], 
                                     train_batch_size = cfg["training"]["train_batch_size"],
                                     val_batch_size = cfg["training"]["val_1_batch_size"], 
                                     test_batch_size = cfg["training"]["val_2_batch_size"], 
                                     mesoscale_cut = cfg["data"]["mesoscale_cut"],
                                     fake_weather = cfg["training"]["fake_weather"])
    
    # build back the datasets for safety
    EN_dataset.serialize_datasets(wandb.run.dir)
    
    # load callbacks
    wd_callbacks = WandbTrain_callback(cfg = cfg, print_preds=True)
    # create folder for runtime models
    runtime_model_folder = os.path.join(wandb.run.dir,"runtime_model")
    os.mkdir(runtime_model_folder)
    checkpoint_callback = ModelCheckpoint(dirpath=runtime_model_folder, 
                                          save_on_train_epoch_end=True, 
                                          save_top_k = -1,
                                          filename = 'model_{epoch:03d}')

    # set up trainer
    trainer = Trainer(max_epochs=cfg["training"]["epochs"], 
                      logger=wandb_logger,
                      devices=cfg["training"]["devices"],
                      accelerator=cfg["training"]["accelerator"],
                      callbacks=[wd_callbacks, checkpoint_callback])

    # run training
    trainer.fit(model, EN_dataset)

    if not cfg["training"]["offline"]:
        wandb.finish()
    
if __name__ == "__main__":
    main()
