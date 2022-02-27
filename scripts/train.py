import sys
import os
from os import listdir
from pathlib import Path
import json
import wandb
import random
import pytorch_lightning as pl
sys.path.append(os.getcwd())
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from config.config import train_line_parser
from drought_impact_forecasting.models.EN_model import EN_model
from Data.data_preparation import Earth_net_DataModule
from callbacks import WandbTrain_callback
from datetime import datetime

def main():
    # load configs
    model_type, cfg_model, cfg_training = train_line_parser()

    if not cfg_training["offline"]:
        os.environ["WANDB_MODE"]="online"
    else:
        os.environ["WANDB_MODE"]="offline"
    
    if cfg_training["checkpoint"] is not None:
        # resume training from where we left off
        old_wandb_dir = str(Path(cfg_training["checkpoint"]).parents[2])
        old_run_id = [f for f in listdir(old_wandb_dir) if 'run-' in f][0][4:-6]

        wandb.init(entity="eth-ds-lab",
                   project="Drought Impact Forecasting",
                   config={"model_type":model_type,"training":cfg_training,"model":cfg_model},
                   id=old_run_id,
                   resume="must")

        wandb_logger = WandbLogger(project='DS_Lab',
                                   config={"model_type":model_type,"training":cfg_training,"model":cfg_model},
                                   job_type='train',
                                   offline=True,
                                   id=old_run_id)

    else:
        wandb.init(entity="eth-ds-lab",
                   project="Drought Impact Forecasting",
                   config={"model_type":model_type,"training":cfg_training,"model":cfg_model})

        wandb_logger = WandbLogger(project='DS_Lab',
                                   config={"model_type":model_type,"training":cfg_training,"model":cfg_model},
                                   job_type='train',
                                   offline=True)

    # store the model name to wandb
    with open(os.path.join(wandb.run.dir, "run_name.txt"), 'w') as f:
        try:
            f.write(wandb.run.name)
        except:
            f.write("offline_run_" + str(datetime.now()))

    # setup model
    model = EN_model(model_type, cfg_model, cfg_training)

    with open(os.path.join(wandb.run.dir, "Training.json"), 'w') as fp:
        json.dump(cfg_training, fp)    
    with open(os.path.join(wandb.run.dir, model_type + ".json"), 'w') as fp:
        json.dump(cfg_model, fp)

    random.seed(cfg_training["seed"])
    pl.seed_everything(cfg_training["seed"], workers=True)

    EN_dataset = Earth_net_DataModule(data_dir = cfg_training["pickle_dir"], 
                                      train_batch_size = cfg_training["train_batch_size"],
                                      val_batch_size = cfg_training["val_1_batch_size"], 
                                      test_batch_size = cfg_training["val_2_batch_size"], 
                                      mesoscale_cut = cfg_training["mesoscale_cut"],
                                      fake_weather = cfg_training["fake_weather"])
    
    # build back the datasets for safety
    EN_dataset.serialize_datasets(wandb.run.dir)
    
    # load callbacks
    wd_callbacks = WandbTrain_callback(print_preds=True)
    # create folder for runtime models
    runtime_model_folder = os.path.join(wandb.run.dir,"runtime_model")
    os.mkdir(runtime_model_folder)
    checkpoint_callback = ModelCheckpoint(dirpath=runtime_model_folder, 
                                          save_on_train_epoch_end=True, 
                                          save_top_k = -1,
                                          filename = 'model_{epoch:03d}')

    # set up trainer    
    trainer = Trainer(max_epochs=cfg_training["epochs"], 
                      logger=wandb_logger,
                      devices=cfg_training["devices"],
                      accelerator=cfg_training["accelerator"],
                      callbacks=[wd_callbacks, checkpoint_callback], 
                      num_sanity_val_steps=1)

    # run training
    if cfg_training["checkpoint"] is None:
        trainer.fit(model, EN_dataset)
    else:
        trainer.fit(model, datamodule=EN_dataset, ckpt_path=cfg_training["checkpoint"])

    if not cfg_training["offline"]:
        wandb.finish()

if __name__ == "__main__":
    main()
    