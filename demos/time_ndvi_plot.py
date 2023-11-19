import sys
import os
import numpy as np
from matplotlib import gridspec
import datetime as dt
import copy
import argparse
import pickle
import torch
from scipy.interpolate import interp1d

sys.path.append(os.getcwd())

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from load_model_data import *

dates = {
    "extreme": ['2018-01-28','2018-11-23'],
    "seasonal": ['2017-05-28','2020-04-11'],
    "ood": ['2017-07-02','2017-11-28'],
    "iid" : ['2017-06-20','2017-11-16'] 
}

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def main(index=0):

    parser = argparse.ArgumentParser(
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-i', '--index', type=int, default=0, help='index of datacube to use')
    parser.add_argument('-a', '--all', type=bool, default=False, help='whether to use all extreme samples')
    parser.add_argument('-t', '--tile', type=str, default=None, help='tile to use')
    parser.add_argument('-v', '--var', type=str, default='precipitation', help='which variable to replace with 2019 data')
    args = parser.parse_args()

    print("Evaluating on index " + str(args.index))
    print("Evaluating on whole extreme dataset: " + str(args.all))

    if False:
        num_processed_cubes = int(len(os.listdir('demos/visualizations/saved_vals')) / 4)
        print("So far " + str(num_processed_cubes) + " cubes have been processed.")

        dataset = "extreme"
        #truth, context, target, npf = load_data_point(test_context_dataset = "Data/small_data/"+dataset+"_context_data_paths.pkl", 
        #                                        test_target_dataset = "Data/small_data/"+dataset+"_target_data_paths.pkl",
        #                                        index=index)

        truth_full = torch.zeros(1,11,128,128,60)
        pred1_full = torch.zeros(1,4,128,128,40)
        pred2_full = torch.zeros(1,4,128,128,40)
        pred_mo_full = torch.zeros(1,4,128,128,40)

        for j in range(num_processed_cubes):

            if (j % 100) == 0:
                print(j)

            truth = torch.load("demos/visualizations/saved_vals/truth_" + str(j) + ".pt")
            pred1 = torch.load("demos/visualizations/saved_vals/pred1_" + str(j) + ".pt")
            pred2 = torch.load("demos/visualizations/saved_vals/pred2_" + str(j) + ".pt")
            pred_mo = torch.load("demos/visualizations/saved_vals/pred_mo_" + str(j) + ".pt")

            truth_full += truth
            pred1_full += pred1
            pred2_full += pred2
            pred_mo_full += pred_mo

        truth_full /= num_processed_cubes
        pred1_full /= num_processed_cubes
        pred2_full /= num_processed_cubes
        pred_mo_full /= num_processed_cubes

        torch.save(truth_full, "demos/visualizations/saved_vals/truth_all.pt")
        torch.save(pred1_full, "demos/visualizations/saved_vals/pred1_all.pt")
        torch.save(pred2_full, "demos/visualizations/saved_vals/pred2_all.pt")
        torch.save(pred_mo_full, "demos/visualizations/saved_vals/pred_mo_all.pt")

        plot_ndvi(truth = truth_full, 
                    preds = [pred_mo_full,pred1_full,pred2_full], 
                    dates_bound = dates[dataset], 
                    model_names=["2019 weather","SGConvLSTM","SGEDConvLSTM"], 
                    filename = "demos/visualizations/"+dataset+"_ndvi.pdf",
                    prec_temp = False)

    # Load 2019 weather data
    truth_19, context_19, target_19, npf_19 = load_data_point(train_dataset = "Data/germany_data/train_data_paths.pkl",
                                                index=0)

    if not args.all:

        model1 = load_model()
        #model2 = load_model("trained_models/top_performant_autoenc.ckpt")
        model2 = load_model("trained_models/SGEDConvLSTM.ckpt")
        dataset = "extreme"
        truth, context, target, npf = load_data_point(test_context_dataset = "Data/small_data/"+dataset+"_context_data_paths.pkl", 
                                                test_target_dataset = "Data/small_data/"+dataset+"_target_data_paths.pkl",
                                                index=index)
        #truth_19, context_19, target_19, npf_19 = load_data_point(train_dataset = "Data/ger_data/train_data_paths.pkl",
        #                                        index=index)
        
        npf_19 = truth_19[:,5:,...]
        truth_modified = copy.deepcopy(truth)
        truth_modified[:, 6:, :, :, 12:(12+30)] = truth_19[:,6:,...]
        npf_modified = truth_modified[:, 5:, :, :, 20:]
        context_modified = truth_modified[:, :, :, :, :20]
        pred1, _, _ = model1(x = context, 
                        prediction_count = int((2/3)*truth.shape[-1]), 
                        non_pred_feat = npf)
        pred2, _, _ = model2(x = context, 
                        prediction_count = int((2/3)*truth.shape[-1]), 
                        non_pred_feat = npf)
        pred_mo, _, _ = model1(x = context_modified, 
                        prediction_count = int((2/3)*truth.shape[-1]), 
                        non_pred_feat = npf_modified)
        plot_ndvi(truth = truth, 
                    preds = [pred_mo,pred1,pred2], 
                    dates_bound = dates[dataset], 
                    model_names=["2019 weather","SGConvLSTM","SGEDConvLSTM"], 
                    filename = "demos/visualizations/"+dataset+"_ndvi.pdf",
                    prec_temp = False)

        # take out cloudy days
        #splits = np.linspace(0.1,1,10)

    else:
        model1 = load_model()
        #model2 = load_model("trained_models/top_performant_autoenc.ckpt")
        model2 = load_model("trained_models/SGEDConvLSTM.ckpt")
        dataset = "extreme"

        test_context_dataset = "Data/extreme_data/"+dataset+"_data_context_data_paths.pkl"
        with open(join(os.getcwd(), test_context_dataset),'rb') as f:
            test_context_path_list = pickle.load(f)
            if args.tile is not None:
                test_context_path_list = [path for path in test_context_path_list if args.tile in path]

        no_extreme_cubes = len(test_context_path_list)
        print("There are " + str(no_extreme_cubes) + " datacubes!")

        truth, context, target, npf = load_data_point(test_context_dataset = "Data/small_data/"+dataset+"_context_data_paths.pkl", 
                                                      test_target_dataset = "Data/small_data/"+dataset+"_target_data_paths.pkl",
                                                      index=index)
        npf_19 = truth_19[:, 5:, ...]

        # Just to get a placeholder of correct values
        pred1, _, _ = model1(x = context, 
                        prediction_count = int((2/3)*truth.shape[-1]), 
                        non_pred_feat = npf)
        #truth_full = torch.zeros_like(truth)
        #pred1_full = torch.zeros_like(pred1)
        #pred2_full = torch.zeros_like(pred1)
        #pred_mo_full = torch.zeros_like(pred1)

        #no_extreme_cubes = 5

        for j in range(0, no_extreme_cubes):

            if (j % 100) == 0:
                print(j)

            truth, context, target, npf = load_data_point(test_context_dataset = "Data/extreme_data/"+dataset+"_data_context_data_paths.pkl", 
                                                          test_target_dataset = "Data/extreme_data/"+dataset+"_data_target_data_paths.pkl",
                                                          index=j)
            
            truth_modified = copy.deepcopy(truth)
            truth_modified[:, 6:, :, :, 12:(12+30)] = truth_19[:, 6:, ...]
            npf_modified = truth_modified[:, 5:, :, :, 20:]
            context_modified = truth_modified[:, :, :, :, :20]
            pred1, _, _ = model1(x = context, 
                                 prediction_count = int((2/3)*truth.shape[-1]), 
                                 non_pred_feat = npf)
            pred2, _, _ = model2(x = context, 
                                 prediction_count = int((2/3)*truth.shape[-1]), 
                                 non_pred_feat = npf)
            pred_mo, _, _ = model1(x = context_modified, 
                                   prediction_count = int((2/3)*truth.shape[-1]), 
                                   non_pred_feat = npf_modified)

            #plot_ndvi(truth = truth, 
            #        preds = [pred_mo,pred1,pred2], 
            #        dates_bound = dates[dataset], 
            #        model_names=["2019 weather","SGConvLSTM","SGEDConvLSTM"], 
            #        filename = "demos/visualizations/"+dataset+"_"+str(j)+"_ndvi.pdf",
            #        prec_temp = False)
            
            prepare_ndvi(truth = truth, 
                         preds = [pred_mo, pred1, pred2], 
                         dates_bound = dates[dataset], 
                         model_names=["2019 weather", "SGConvLSTM", "SGEDConvLSTM"], 
                         filename = "demos/visualizations/"+dataset+"_"+str(j)+"_ndvi.pdf",
                         prec_temp = False,
                         idx = j)

            #torch.save(truth, "demos/visualizations/saved_vals/truth_" + str(j) + ".pt")
            #torch.save(pred1, "demos/visualizations/saved_vals/pred1_" + str(j) + ".pt")
            #torch.save(pred2, "demos/visualizations/saved_vals/pred2_" + str(j) + ".pt")
            #torch.save(pred_mo, "demos/visualizations/saved_vals/pred_mo_" + str(j) + ".pt")

            #truth_full += truth
            #pred1_full += pred1
            #pred2_full += pred2
            #pred_mo_full += pred_mo

        '''truth_full /= no_extreme_cubes
        pred1_full /= no_extreme_cubes
        pred2_full /= no_extreme_cubes
        pred_mo_full /= no_extreme_cubes

        plot_ndvi(truth = truth_full, 
                    preds = [pred_mo_full,pred1_full,pred2_full], 
                    dates_bound = dates[dataset], 
                    model_names=["2019 weather","SGConvLSTM","SGEDConvLSTM"], 
                    filename = "demos/visualizations/full_extreme_ndvi.pdf",
                    prec_temp = False)'''

    print("DONE")

def prepare_ndvi(truth, preds, dates_bound = None, filename = None, model_names = None, prec_temp = True, idx=0):
    ndvi_truth = ((truth[:, 3, ...] - truth[ :, 2, ...]) / (
            truth[:, 3, ...] + truth[:, 2, ...] + 1e-6))
    cloud_mask = 1 - truth[:, 4, ...]
    ndvi_truth = ndvi_truth*cloud_mask

    if not isinstance(preds, list):
        preds = [preds]

    if model_names is None:
        model_names = []
        for i in range(len(preds)):
            model_names.append("model {0}".format(i+1))
    
    colors = ["b","r","g","c","m","y"]
    colors = colors[:len(preds)]

    ndvi_preds = []
    for pred in preds:
        ndvi_preds.append(((pred[:, 3, ...] - pred[ :, 2, ...]) / (
                    pred[:, 3, ...] + pred[:, 2, ...] + 1e-6)))
        
    torch.save(ndvi_truth, "demos/visualizations/saved_vals/ndvi_truth_" + str(idx) + ".pt")
    torch.save(ndvi_preds, "demos/visualizations/saved_vals/ndvi_preds_" + str(idx) + ".pt")

    def date_linspace(start, end, steps):
        delta = (end - start) / steps
        increments = range(0, steps) * np.array([delta]*steps)
        return start + increments
    
    confidence = .5
    splits = [(1-confidence)/2, 0.5, 1 - (1-confidence)/2]
    q_t = np.quantile(ndvi_truth, splits, axis = (0,1,2))

    valid_ndvi_time = q_t[0]!=0
    #q_t[0,valid_ndvi_time]

    for i in range(q_t.shape[0]):
        xnew = np.arange(len(q_t[i]))
        zero_idx = np.where(q_t[i]==0)

        xold = np.delete(xnew, zero_idx)
        yold = np.delete(q_t[i], zero_idx)

        f = interp1d(xold, yold, fill_value="extrapolate")
        q_t[i] = np.clip(f(xnew), 0, 1)

    q_ps = []
    for ndvi_pred in ndvi_preds:
        q_ps.append(np.quantile(ndvi_pred.detach().numpy(), splits, axis = (0,1,2)))


    x_t = np.arange(truth.shape[-1])#[valid_ndvi_time] We've now interpolated, so all x's are valid
    x_p = np.arange(truth.shape[-1] - pred.shape[-1], truth.shape[-1])

    cur_data = [x_t, q_t, x_p, q_ps]
    save_object(cur_data, 'demos/visualizations/ndvi_pickles/ndvi_' + str(idx) + '.pkl')
    
    print("prepared sample!")

    '''confidence = .5
    splits = [(1-confidence)/2, 0.5, 1 - (1-confidence)/2]
    q_t = np.quantile(ndvi_truth, splits, axis = (0,1,2))

    valid_ndvi_time = q_t[0]!=0
    q_t[0,valid_ndvi_time]

    x_t = np.arange(truth.shape[-1])[valid_ndvi_time]
    x_p = np.arange(truth.shape[-1] - pred.shape[-1], truth.shape[-1])

    def date_linspace(start, end, steps):
        delta = (end - start) / steps
        increments = range(0, steps) * np.array([delta]*steps)
        return start + increments

    if dates_bound is not None:
        dates_bound = [dt.datetime.strptime(dates_bound[0], '%Y-%m-%d'),dt.datetime.strptime(dates_bound[1], '%Y-%m-%d')] 
        dates = date_linspace(dates_bound[0],dates_bound[1],truth.shape[-1])

        x_t = dates[valid_ndvi_time]
        x_p = dates[int((1/3)*truth.shape[-1]):]

    q_ps = []
    for ndvi_pred in ndvi_preds:
        q_ps.append(np.quantile(ndvi_pred.detach().numpy(), splits, axis = (0,1,2)))'''

def plot_ndvi(truth, preds, dates_bound = None, filename = None, model_names = None, prec_temp = True):

    ndvi_truth = ((truth[:, 3, ...] - truth[ :, 2, ...]) / (
            truth[:, 3, ...] + truth[:, 2, ...] + 1e-6))
    cloud_mask = 1 - truth[:, 4, ...]
    ndvi_truth = ndvi_truth*cloud_mask

    if not isinstance(preds, list):
        preds = [preds]

    if model_names is None:
        model_names = []
        for i in range(len(preds)):
            model_names.append("model {0}".format(i+1))
    
    colors = ["b","r","g","c","m","y"]
    colors = colors[:len(preds)]

    ndvi_preds = []
    for pred in preds:
        ndvi_preds.append(((pred[:, 3, ...] - pred[ :, 2, ...]) / (
                    pred[:, 3, ...] + pred[:, 2, ...] + 1e-6)))

    confidence = .5
    splits = [(1-confidence)/2, 0.5, 1 - (1-confidence)/2]
    q_t = np.quantile(ndvi_truth, splits, axis = (0,1,2))

    valid_ndvi_time = q_t[0]!=0
    q_t[0,valid_ndvi_time]

    x_t = np.arange(truth.shape[-1])[valid_ndvi_time]
    x_p = np.arange(truth.shape[-1] - pred.shape[-1], truth.shape[-1])

    def date_linspace(start, end, steps):
        delta = (end - start) / steps
        increments = range(0, steps) * np.array([delta]*steps)
        return start + increments

    #if dates_bound is not None:
    #    dates_bound = [dt.datetime.strptime(dates_bound[0], '%Y-%m-%d'),dt.datetime.strptime(dates_bound[1], '%Y-%m-%d')] 
    #    dates = date_linspace(dates_bound[0],dates_bound[1],truth.shape[-1])

    #    x_t = dates[valid_ndvi_time]
    #    x_p = dates[int((1/3)*truth.shape[-1]):]

    q_ps = []
    for ndvi_pred in ndvi_preds:
        q_ps.append(np.quantile(ndvi_pred.detach().numpy(), splits, axis = (0,1,2)))

    # plot pred and truth NDVI
    if prec_temp:
        fig = plt.figure()
        # set height ratios for subplots
        gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1]) 

        # the first subplot
        ax0 = plt.subplot(gs[0])

        for q_p, mod_name, color in zip(q_ps, model_names, colors):
            ax0.plot(x_p, q_p[1,:], '--', color = color, label = mod_name)

        ax0.plot(x_t, q_t[1,valid_ndvi_time], '-',color = 'k', label = 'Ground Truth')

        ax0.legend(loc="upper right")
        for q_p, color in zip(q_ps, colors):
            ax0.fill_between(x_p, q_p[0,:],q_p[2,:], color = color, alpha=.1)
        ax0.fill_between(x_t, q_t[0,valid_ndvi_time],q_t[2,valid_ndvi_time], color = 'k', alpha=.1)


        # Set the locator
        locator = mdates.MonthLocator()  # every month
        # Specify the format - %b gives us Jan, Feb...
        fmt = mdates.DateFormatter('%b')
        # log scale for axis Y of the first subplot

        # the second subplot
        X = ax0.xaxis
        X.set_major_locator(locator)
        X.set_major_formatter(fmt)
        # shared axis X
        ax1 = plt.subplot(gs[1], sharex = ax0)
        ax2 = plt.subplot(gs[2], sharex = ax0)

        ax0.grid()
        ax1.grid()
        ax2.grid()
        if dates_bound is None:
            ax1.plot(np.arange(truth.shape[-1]-1)+1 , 50 * truth[0,6,0,0,:-1], label = 'precipitation')
            ax2.plot(np.arange(truth.shape[-1]-1)+1 , 50 *(2*truth[0,8,0,0,:-1] - 1), label = 'mean temp')
        else:
            ax1.plot(dates[1:] , 50 * truth[0,6,0,0,:-1], label = 'precipitation')
            ax2.plot(dates[1:] , 50 *(2*truth[0,8,0,0,:-1] - 1), label = 'mean temp')

        ax0.set_ylabel("NDVI")
        ax1.set_ylabel("Rain" +"\n" + "(mm)")
        ax2.set_ylabel("Temp" +"\n" + "(°C)")
        ax2.set_xlabel("Time")

        # Set the locator
        locator = mdates.MonthLocator()  # every month
        # Specify the format - %b gives us Jan, Feb...
        fmt = mdates.DateFormatter('%b')
        # log scale for axis Y of the first subplot

        # the second subplot
        X = ax2.xaxis
        X.set_major_locator(locator)
        X.set_major_formatter(fmt)

        #plt.xlim([0, truth.size()[-1]])

        plt.setp(ax0.get_xticklabels(), visible=False)
        plt.setp(ax1.get_xticklabels(), visible=False)
        # remove last tick label for the second subplot
        '''
        yticks = ax1.yaxis.get_major_ticks()
        yticks[-1].label1.set_visible(False)'''

        # put legend on first subplot
        #ax0.legend((line0, line1), ('red line', 'blue line'), loc='lower left')

        # remove vertical gap between subplots
        plt.subplots_adjust(hspace=.0)

    else:
        fig, ax0 = plt.subplots()
        for q_p, mod_name, color in zip(q_ps, model_names, colors):
            ax0.plot(x_p, q_p[1,:], '--',color = color, label = mod_name)

        ax0.plot(x_t, q_t[1,valid_ndvi_time], '-', color = 'k', label = '2018 ground truth')

        ax0.legend(loc="upper right")
        for q_p, color in zip(q_ps, colors):
            ax0.fill_between(x_p, q_p[0,:],q_p[2,:], color = color, alpha=.1)
        ax0.fill_between(x_t, q_t[0,valid_ndvi_time],q_t[2,valid_ndvi_time], color = 'k', alpha=.1)

        #plt.axvline(dt.datetime.strptime("2018-06-05", '%Y-%m-%d'), color = "k", linestyle = ":")
        # Set the locator
        #locator = mdates.MonthLocator()  # every month
        # Specify the format - %b gives us Jan, Feb...
        #fmt = mdates.DateFormatter('%b')
        # log scale for axis Y of the first subplot

        # the second subplot
        X = ax0.xaxis
        #X.set_major_locator(locator)
        #X.set_major_formatter(fmt)
        ax0.grid()

        ax0.set_ylabel("NDVI (unitless)")
        ax0.set_xlabel("Time")

        # New edits Oto 8.4.
        days = [4, 32, 63, 93, 124, 154, 185, 216, 246, 277]
        days = [d/5 for d in days]
        plt.xticks(days, ['Feb', 'March', 'April', 'May', 'June', 'July', 'Aug', 'Sep', 'Oct', 'Nov'])
        all_x_times = np.arange(truth.shape[-1])

        plt.xlim([all_x_times[0], all_x_times[-1]])
        plt.ylim(0,1)

        plt.show()


    #plt.xlim([x_t[0],x_t[-1]]) # only to see the legal part

    if filename == None:
        plt.savefig('visualizations/NDVI_time_series.pdf', format="pdf")
        plt.show()
    else:
        filename = 'demos/visualizations/extreme_ndvi_single.pdf'
        plt.savefig(filename)
    
if __name__ == "__main__":
    main()