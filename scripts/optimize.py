import sys
import os
import json
import random

sys.path.append(os.getcwd())

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from config.config import train_line_parser
from drought_impact_forecasting.models.EN_model import EN_model
from Data.data_preparation import Earth_net_DataModule
from callbacks import WandbTrain_callback

import wandb
from datetime import datetime

import optuna
from optuna.trial import TrialState
from optuna.integration import PyTorchLightningPruningCallback
import warnings
warnings.filterwarnings('ignore')

def objective(trial):

    # load configs
    model_type, cfg_model, cfg_training = train_line_parser()

    # set up search space
    cfg_model["n_layers"] = trial.suggest_int('nl', 2, 4)
    cfg_model["hidden_channels"] = trial.suggest_int('hc', 15, 22)
    cfg_model["layer_norm"] = trial.suggest_categorical("lm", [True,False])
    cfg_training["start_learn_rate"] = trial.suggest_float("lr", 1e-5, 1e-4, log=True)
    cfg_training["patience"] = trial.suggest_int("pa", 3, 20)
    cfg_training["optimizer"] = trial.suggest_categorical("op", ["adam","adamW"])

    # kernel sizes must be odd to be symmetric
    cfg_model["kernel"] = 3 + 2 * trial.suggest_int('k', 0, 2)
    cfg_model["memroy_kernel"] = 3 + 2 * trial.suggest_int('mk', 0, 2)

    if not cfg_training["offline"]:
        os.environ["WANDB_MODE"]="online"
    else:
        os.environ["WANDB_MODE"]="offline"

    wandb.init(entity="eth-ds-lab",
               project="DIF Optimization",
               config={"model_type":model_type,"training":cfg_training,"model":cfg_model})

    # store the model name to wandb
    with open(os.path.join(wandb.run.dir, "run_name.txt"), 'w') as f:
        try:
            f.write(wandb.run.name)
        except:
            f.write("offline_run_" + str(datetime.now()))

    with open(os.path.join(wandb.run.dir, "Training.json"), 'w') as fp:
        json.dump(cfg_training, fp)    
    with open(os.path.join(wandb.run.dir, model_type + ".json"), 'w') as fp:
        json.dump(cfg_model, fp)
    
    wandb_logger = WandbLogger(project='DS_Lab',
                               config={"model_type":model_type,"training":cfg_training,"model":cfg_model},
                               job_type='train',
                               offline=True)
    
    random.seed(cfg_training["seed"])
    pl.seed_everything(cfg_training["seed"], workers=True)

    EN_dataset = Earth_net_DataModule(data_dir = cfg_training["pickle_dir"], 
                                     train_batch_size = cfg_training["train_batch_size"],
                                     val_batch_size = cfg_training["val_1_batch_size"], 
                                     test_batch_size = cfg_training["val_2_batch_size"], 
                                     mesoscale_cut = cfg_training["mesoscale_cut"])
    
    # to build back the datasets for safety
    EN_dataset.serialize_datasets(wandb.run.dir)
    
    # load Callbacks
    wd_callbacks = WandbTrain_callback(print_preds=True)
    # Create folder for runtime models
    runtime_model_folder = os.path.join(wandb.run.dir, "runtime_model")

    if not os.path.isdir(runtime_model_folder):
        os.mkdir(runtime_model_folder)
    
    checkpoint_callback = ModelCheckpoint(dirpath=runtime_model_folder, 
                                          save_on_train_epoch_end=True, 
                                          save_top_k = -1,
                                          filename = 'model_{epoch:03d}')
    prun_callback = PyTorchLightningPruningCallback(trial, monitor='epoch_validation_ENS')

    # setup Trainer
    trainer = Trainer(max_epochs=cfg_training["epochs"], 
                      logger=wandb_logger,
                      devices=cfg_training["devices"],
                      accelerator=cfg_training["accelerator"],
                      callbacks=[wd_callbacks, checkpoint_callback, prun_callback],
                      num_sanity_val_steps=0)

    # setup model
    model = EN_model(model_type, cfg_model, cfg_training)
    
    # run training
    trainer.fit(model, EN_dataset)

    if not cfg_training["offline"]:
        wandb.finish()
    
    return trainer.callback_metrics["epoch_validation_ENS"].item()

if __name__ == "__main__":

    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction='maximize', pruner=pruner)
    study.optimize(objective, n_trials=10000, timeout=1000000)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  ENS Score: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
