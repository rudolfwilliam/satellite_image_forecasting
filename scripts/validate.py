import sys
import os
import wandb
import numpy as np
sys.path.append(os.getcwd())
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from config.config import validate_line_parser
from drought_impact_forecasting.models.EN_model import EN_model
from Data.data_preparation import Earth_net_DataModule
from scripts.callbacks import WandbTest_callback

def main():
    
    cfg = validate_line_parser()

    print("Validating experiment {0}".format(cfg['run_name']))
    print("Validating model at epoch {0}".format(cfg['epoch_to_validate']))

    wandb.login()

    # GPU handling
    # print("GPU count: {0}".format(gpu_count))

    wandb_logger = WandbLogger(entity="eth-ds-lab", project="DIF Testing", offline=True)

    # always use same val_2 data from Data folder
    EN_dataset = Earth_net_DataModule(data_dir=cfg['dataset_dir'],
                                      train_batch_size=cfg['batch_size'],
                                      val_batch_size=cfg['batch_size'],
                                      test_batch_size=cfg['batch_size'],
                                      test_set=cfg['test_set'],
                                      mesoscale_cut=[39,41])
    
    callbacks = WandbTest_callback(cfg['run_name'], cfg['epoch_to_validate'], cfg['test_set'])

    # setup Trainer
    trainer = Trainer(logger=wandb_logger, callbacks=[callbacks], accelerator='auto')

    # setup Model
    model = EN_model.load_from_checkpoint(cfg['model_path'])
    model.eval()

    # run validation
    trainer.test(model=model, dataloaders=EN_dataset)

    wandb.finish()

if __name__ == "__main__":
    main()
