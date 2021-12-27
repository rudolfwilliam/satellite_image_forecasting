import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torch
import os
from os import path
import numpy as np
from pathlib import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable
from drought_impact_forecasting.models.utils.utils import mean_prediction, last_prediction, ENS
import json
import wandb

class SDVI_Train_callback(pl.Callback):
    def __init__(self):
        self.runtime_model_folder = os.path.join(wandb.run.dir,"runtime_model")
        os.mkdir(os.path.join(wandb.run.dir,"runtime_model"))
    
    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", unused = None) -> None:
        
        torch.save(trainer.model.state_dict(), os.path.join(self.runtime_model_folder, "model_"+str(trainer.current_epoch)+".torch"))
        return super().on_train_epoch_end(trainer, pl_module)
    


class WandbTrain_callback(pl.Callback):
    def __init__(self, cfg, print_preds = True):
        self.print_preds = print_preds
        self.print_sample = None
        self.print_table = None

        self.step_train_loss = []
        self.epoch_train_loss = []
        self.validation_loss = []

        self.runtime_prediction = os.path.join(wandb.run.dir,"runtime_pred") 
        self.r_pred = os.path.join(self.runtime_prediction,"r")
        self.g_pred = os.path.join(self.runtime_prediction,"g")
        self.b_pred = os.path.join(self.runtime_prediction,"b")
        self.i_pred = os.path.join(self.runtime_prediction,"i")
        self.img_pred = os.path.join(self.runtime_prediction,"img")
        self.channel_list = [self.r_pred, self.g_pred, self.b_pred, self.i_pred]

        for dir_path in [self.runtime_prediction,
                         self.r_pred,self.g_pred,self.b_pred,self.i_pred,self.img_pred]:
            if not path.isdir(dir_path):
                os.mkdir(dir_path)
        #wandb.init()

        

        wandb.define_metric("step")
        wandb.define_metric("epoch")


        wandb.define_metric('batch_training_loss', step_metric = "step")
        wandb.define_metric('epoch_training_loss', step_metric = "epoch")

        #self.log_ENS_baseline(val_1_data)

        # define our custom x axis metric
        pass

    def log_ENS_baseline(self, data):
        scores_mean = np.zeros((data.__len__(), 5))
        scores_last = np.zeros((data.__len__(), 5))

        for i in range(data.__len__()):
            all_data = data.__getitem__(i)

            T = all_data.size()[3]

            t0 = round(all_data.shape[-1]/3) #t0 is the length of the context part

            # For last/mean baseline we don't need weather
            context = all_data[:5, :, :, :t0].unsqueeze(0) # b, c, h, w, t
            target = all_data[:5, :, :, t0:].unsqueeze(0) # b, c, h, w, t

            preds_mean = mean_prediction(context, mask_channel=True).permute(0,3,1,2,4)
            preds_last = last_prediction(context, mask_channel=4).permute(0,3,1,2,4)

            _, part_scores_mean = ENS(prediction = preds_mean, target = target)
            _, part_scores_last = ENS(prediction = preds_last, target = target)

            scores_mean[i, :] = part_scores_mean
            scores_last[i, :] = part_scores_last
            
        avg_scores_mean = np.mean(scores_mean, axis = 0)
        avg_scores_last = np.mean(scores_last, axis = 0)
        wandb.log({ 
                                'baseline_ENS_mean':  avg_scores_mean[0],
                                'baseline_mad_mean':  avg_scores_mean[1],
                                'baseline_ssim_mean': avg_scores_mean[2],
                                'baseline_ols_mean':  avg_scores_mean[3],
                                'baseline_emd_mean':  avg_scores_mean[4],
                                'baseline_ENS_last':  avg_scores_last[0],
                                'baseline_mad_last':  avg_scores_last[1],
                                'baseline_ssim_last': avg_scores_last[2],
                                'baseline_ols_last':  avg_scores_last[3],
                                'baseline_emd_last':  avg_scores_last[4]
                            })
                

    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs, batch, batch_idx: int, dataloader_idx: int) -> None:
        tr_loss = float(outputs['loss'])
        self.step_train_loss.append(tr_loss)
        trainer.logger.experiment.log({ 
                                        'step': trainer.global_step,
                                        'batch_training_loss': tr_loss
                                    })
        return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
    
    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", unused = None) -> None:
        # compute the avg training loss
        e_loss = sum(self.step_train_loss)/len(self.step_train_loss)
        self.epoch_train_loss.append(e_loss)
        # resetting the per-batch training loss
        self.step_train_loss = []
        lr = trainer.lr_schedulers[0]['scheduler'].optimizer.param_groups[0]['lr']

        pl_module.log('epoch_training_loss', e_loss, on_epoch=True, on_step=False)
        pl_module.log('lr', lr, on_epoch=True, on_step=False)

        #torch.save(trainer.model.state_dict(), os.path.join(self.runtime_model_folder, "model_"+str(trainer.current_epoch)+".torch"))
        return super().on_train_epoch_end(trainer, pl_module)
    
    def on_validation_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs, batch, batch_idx: int, dataloader_idx: int) -> None:
        self.validation_loss.append(outputs)
        # Assigning the picture
        if self.print_preds:
            if self.print_sample is None:
                self.print_sample = batch[0:,...]
                self.log_groundtruth(trainer.model, self.print_sample)
        
        return super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:

        v_loss = np.mean(np.vstack(self.validation_loss), axis = 0)

        # resetting the per-batch validation loss
        self.validation_loss = []
        if not trainer.sanity_checking:
            trainer.logger.experiment.log({ 
                                    'epoch': trainer.current_epoch,
                                    'epoch_validation_ENS':  v_loss[0],
                                    'epoch_validation_mad':  v_loss[1],
                                    'epoch_validation_ssim': v_loss[2],
                                    'epoch_validation_ols':  v_loss[3],
                                    'epoch_validation_emd':  v_loss[4]
                                })
                            
            if self.print_preds:
                self.log_predictions(trainer.model, self.print_sample, trainer.current_epoch)
            return {"epoch_validation_ENS" : v_loss[0]}

    def on_test_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs, batch, batch_idx: int, dataloader_idx: int) -> None:

        return super().on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
    
    def log_predictions(self, model, sample, epoch):
        preds, delta_preds, baselines = model(sample[:1, :, :, :, :10])
        delta = np.flip(delta_preds[0, :4, :, :, 0].cpu().numpy().transpose(1, 2, 0).astype(float), -1)
        #delta_gt = np.flip(((self.sample[:4, :, :, 9] - means[0])[0]).cpu().numpy().transpose(1, 2, 0).astype(float), -1)
     
        figs = []
        for i, c in enumerate(self.channel_list):
            fig, ax = plt.subplots()
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            im = ax.imshow(delta[:, :, i], cmap='inferno')
            fig.colorbar(im, cax=cax, orientation='vertical')
            plt.savefig(c + "/epoch_" + str(epoch) + ".png")
            plt.close()
            figs.append(wandb.Image(plt.imread(c + "/epoch_" + str(epoch) + ".png"), 
                                    caption = "epoch: {0} c: {1}".format(epoch, c[-1])))
            plt.close(fig)

        wandb.log({"epoch": epoch, "pred_imgs": figs})

    def log_groundtruth(self, model, sample):
        preds, delta_preds, baselines = model(sample[:1, :, :, :, :10])
        delta_gt = np.flip(((sample[:1,:4, :, :, 9] - baselines[...,0])[0]).cpu().numpy().transpose(1, 2, 0).astype(float), -1)
        figs = []
        for i, c in enumerate(self.channel_list):
            fig, ax = plt.subplots()
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            im = ax.imshow(delta_gt[:, :, i], cmap='inferno')
            fig.colorbar(im, cax=cax, orientation='vertical')
            fig.savefig(c + "/gt.png")

            figs.append(wandb.Image(plt.imread(c + "/gt.png"), 
                                    caption = "ground truth c: {0}".format(c[-1])))
            plt.close(fig)
        
        #self.print_table = wandb.Table(columns=["id", "r", "g", "b", "i"], data = [figs])

        wandb.log({"epoch": -1,"pred_imgs": figs})
        

class WandbTest_callback(pl.Callback):
    def __init__(self, wandb_name_model_to_test) -> None:
        self.wandb_name_model_to_test = wandb_name_model_to_test
        super().__init__()
    def on_test_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs, batch, batch_idx: int, dataloader_idx: int) -> None:
        with open(os.path.join(wandb.run.dir,"scores_"+self.wandb_name_model_to_test+".csv"), 'a') as filehandle:
            for i in range(len(outputs)):
                filehandle.write(str(outputs[i,1]) + "," + str(outputs[i,2]) + "," + str(outputs[i,3])+ "," + str(outputs[i,4]) + ","+ str(outputs[i,0]) + '\n')

        return super().on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs, batch, batch_idx: int, dataloader_idx: int) -> None:
        return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
    pass

class Prediction_Callback(pl.Callback):
    def __init__(self, ms_cut, train_dir, test_dir, dataset, print_predictions, timestamp):
        self.sample = dataset.__getitem__(0)
        self.print_predictions = print_predictions
        self.epoch = 0
        self.instance_folder = os.getcwd() + "/model_instances/model_" + timestamp
        self.runtime_model_folder = self.instance_folder + "/runtime_model"
        self.runtime_prediction = self.instance_folder + "/runtime_pred"
        self.r_pred = self.runtime_prediction + "/r"
        self.g_pred = self.runtime_prediction + "/g"
        self.b_pred = self.runtime_prediction + "/b"
        self.i_pred = self.runtime_prediction + "/i"
        self.img_pred = self.runtime_prediction + "/img"


        # Set up Prediction directory structure if necessary
        for dir_path in [self.instance_folder,
                         self.runtime_model_folder,self.runtime_prediction,
                         self.r_pred,self.g_pred,self.b_pred,self.i_pred,self.img_pred]:
            if not path.isdir(dir_path):
                os.mkdir(dir_path)
        
        if not os.path.isfile(self.instance_folder + "/scores.csv"):
            with open(self.instance_folder + "/scores.csv", 'w') as filehandle:
                filehandle.write("mad, ssim, ols, emd, score\n")
        
        self.channel_list = [self.r_pred, self.g_pred, self.b_pred, self.i_pred]
    

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", unused: "Optional" = None
    ) -> None:

        torch.save(trainer.model.state_dict(), self.runtime_model_folder + "/model_"+str(self.epoch)+".torch")

        if self.print_predictions:
            # take 10 context and predict 1 (index from )
            preds, delta_preds, means = trainer.model(torch.unsqueeze(self.sample[:, :, :, :10], dim=0))
            '''
            metrics = trainer.callback_metrics
            metrics = {k:float(v) for k, v in metrics.items()}
            metrics['train_loss'] = [float(metrics['train_loss'])]
            metrics['lr'] = [float(metrics['lr'])]
            metrics['online_val_loss'] = [float(metrics['online_val_loss'])]
            '''
            pre_pred = np.flip(preds[0][0, :3, :, :].cpu().numpy().transpose(1, 2, 0).astype(float), -1)

            delta = np.flip(delta_preds[0][0, :4, :, :].cpu().numpy().transpose(1, 2, 0).astype(float), -1)

            delta_gt = np.flip(((self.sample[:4, :, :, 9] - means[0])[0]).cpu().numpy().transpose(1, 2, 0).astype(float), -1)
            
            ims = []
            for i, c in enumerate(self.channel_list):
                plt.imshow(delta[:, :, i])
                plt.colorbar()
                plt.savefig(c + "/epoch_" + str(self.epoch) + ".png")
                plt.close()
                ims.append(wandb.Image(plt.imread(c + "/epoch_" + str(self.epoch) + ".png"), 
                                       caption = "epoch: {0} c: {1}".format(self.epoch, c[-1])))
                    

            wandb.log({"Runtime Predictions":ims})

            # values need to be between 0 and 1
            cor_pred = np.clip(pre_pred, 0, 1)
            '''
            if self.epoch == 0:
                with open(self.instance_folder  + "/metrics.json", 'w') as fp:
                    json.dump(metrics, fp)
            else:
                with open(self.instance_folder  + "/metrics.json", "r+") as fp:
                    data = json.load(fp)
                    data['train_loss'] = data['train_loss'] + metrics['train_loss'] 
                    data['lr'] = data['lr'] + metrics['lr']  
                    data['online_val_loss'] = data['online_val_loss'] + metrics['online_val_loss'] 
                    fp.seek(0)
                    json.dump(data, fp)       '''     

            plt.imsave(self.img_pred + "/epoch_" + str(self.epoch) + ".png", cor_pred)
            ims = []
            # store different rgb values of delta separately
            if self.epoch == 0:
                plt.imsave(self.img_pred + "/gt.png", np.clip(np.flip(self.sample[:3, :, :, 9].detach().cpu().numpy().
                                                                transpose(1, 2, 0).astype(float), -1),0,1))
                
                # ground truth delta
                delta_gt = (self.sample[:4, :, :, 9] - means[0])[0]
                for i, c in enumerate(self.channel_list):
                    plt.imshow(np.flip(delta_gt.detach().cpu().numpy().transpose(1, 2, 0).astype(float), -1)[:, :, i])
                    plt.colorbar()
                    plt.savefig(c + "/gt.png")
                    plt.close()
                    ims.append(wandb.Image(plt.imread(c + "/gt.png"), 
                                           caption = "ground truth c: {1}".format(self.epoch, c[-1])))
                wandb.log({"Runtime Predictions":ims})

            ims = []
            for i, c in enumerate(self.channel_list):
                plt.imshow(delta[:, :, i])
                plt.colorbar()
                plt.savefig(c + "/epoch_" + str(self.epoch) + ".png")
                plt.close()
                ims.append(wandb.Image(plt.imread(c + "/epoch_" + str(self.epoch) + ".png"), 
                                       caption = "epoch: {0} c: {1}".format(self.epoch, c[-1])))

            wandb.log({"Runtime Predictions":ims})
            # in the very first epoch, store ground truth


            self.epoch += 1
    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:

        return super().on_train_end(trainer, pl_module)





