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
from drought_impact_forecasting.models.Peephole_LSTM_model import Peephole_LSTM_model
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

    # Load configs
    cfg = train_line_parser()

    # Set up search space
    cfg["model"]["n_layers"] = trial.suggest_int('nl', 2, 3)
    cfg["model"]["hidden_channels"] = trial.suggest_int('hc', 15, 20)
    cfg["training"]["start_learn_rate"] = trial.suggest_float("lr", 1e-6, 1e-4, log=True)
    cfg["training"]["training_loss"] = trial.suggest_categorical("tl", ["l1","l2","Huber"])
    #cfg["training"]["patience"] = trial.suggest_int("pa", 3, 20)
    #cfg["training"]["optimizer"] = trial.suggest_categorical("op", ["adam","adamW"])
    #cfg["training"]["layer_norm"] = trial.suggest_categorical("lm", [True,False])

    # Kernel sizes must be odd to be symmetric
    cfg["model"]["kernel"] = 3 + 2 * trial.suggest_int('k', 0, 3)
    #cfg["model"]["memroy_kernel"] = 3 + 2 * trial.suggest_int('mk', 0, 2)

    if not cfg["training"]["offline"]:
        os.environ["WANDB_MODE"]="online"
    else:
        os.environ["WANDB_MODE"]="offline"

    wandb.init(entity="eth-ds-lab", project="DIF Optimization", config=cfg)

    # Store the model name to wandb
    with open(os.path.join(wandb.run.dir, "run_name.txt"), 'w') as f:
        try:
            f.write(wandb.run.name)
        except:
            f.write("offline_run_" + str(datetime.now()))

    with open(os.path.join(wandb.run.dir, "Training.json"), 'w') as fp:
        json.dump(cfg, fp)
    
    wandb_logger = WandbLogger(project='DS_Lab', config=cfg, job_type='train', offline=True)
    
    random.seed(cfg["training"]["seed"])
    pl.seed_everything(cfg["training"]["seed"], workers=True)

    EN_dataset = Earth_net_DataModule(data_dir = cfg["data"]["pickle_dir"], 
                                     train_batch_size = cfg["training"]["train_batch_size"],
                                     val_batch_size = cfg["training"]["val_1_batch_size"], 
                                     test_batch_size = cfg["training"]["val_2_batch_size"], 
                                     mesoscale_cut = cfg["data"]["mesoscale_cut"])
    
    # To build back the datasets for safety
    EN_dataset.serialize_datasets(wandb.run.dir)
    
    # Load Callbacks
    wd_callbacks = WandbTrain_callback(cfg = cfg, print_preds=False)
    # Create folder for runtime models
    runtime_model_folder = os.path.join(wandb.run.dir,"runtime_model")

    if not os.path.isdir(runtime_model_folder):
        os.mkdir(runtime_model_folder)
    
    checkpoint_callback = ModelCheckpoint(dirpath=runtime_model_folder, 
                                          save_on_train_epoch_end=True, 
                                          save_top_k = 2,
                                          monitor='epoch_training_loss',
                                          filename = 'model_{epoch:03d}')
    prun_callback = PyTorchLightningPruningCallback(trial, monitor='epoch_training_loss')

    # Setup Trainer
    trainer = Trainer(max_epochs=cfg["training"]["epochs"], 
                      logger=wandb_logger,
                      devices = cfg["training"]["devices"],
                      accelerator=cfg["training"]["accelerator"],
                      callbacks=[wd_callbacks, checkpoint_callback, prun_callback],
                      check_val_every_n_epoch = 1000,
                      num_sanity_val_steps=0)

    # Setup Model
    model = Peephole_LSTM_model(cfg)
    
    # Run training
    trainer.fit(model, EN_dataset)

    if not cfg["training"]["offline"]:
        wandb.finish()
    
    return trainer.callback_metrics["epoch_training_loss"].item()

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
