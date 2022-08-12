# Satellite Image Forecasting - EarthNet2021

[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/) [![DOI](https://zenodo.org/badge/412524927.svg)](https://zenodo.org/badge/latestdoi/412524927) 

Climate change has a large impact on our environment :earth_americas:. We notice that all around the world, catastrophic events such as droughts occur more and more frequently as the years pass. In this repository, you can find three deep learning models that we developed for the [EarthNet2021 challenge](https://www.earthnet.tech/), where the task is to predict future satellite images from past ones using features such as precipitation and elevation maps. With one of our models, a Peephole [Convolutional LSTM](https://proceedings.neurips.cc/paper/2015/file/07563a3fe3bbe7e3ba84431ad9d055af-Paper.pdf), we beat the current best model on the EarthNet challenge leaderboard. 

In all of our models, we employ a technique we refer to as *Baseline Framework*. Here, the model only predicts the deviation (*delta*) onto a precomputed baseline instead of predicting the satellite image directly. In our experiments, this simple trick leads to notably faster convergence. Here is a visualization of how it works:

<p align="center">
<img src="/assets/DS_lab_vis_github.svg" width="400">
</p>

We use [PyTorch](https://pytorch.org/) for the implementation.

## User Guide

### Training, testing and evaluating the model

Prerequisites: Create a conda environment from ```config/dif_env.yml```. Optionally, you may create a pip environment from ```config/dif_env.txt```.

1. Download the dataset (train/iid/ood/extreme/seasonal splits):

    ```python scripts/data_retrieval.py directory/to/store/data split```

2. Collect the paths to the data:

    ```python scripts/data_collection.py -s training/dir -tf test/dir -d dir/to/store/paths -td no/of/training/samples/ -v1 no/of/val1/samples -v2 no/of/val2/samples```

3. Train the model. Use the -mt flag to specify the model type (ConvLSTM, AutoencLSTM, ConvTransformer). Use ```config/Training.json``` and relevant ```<Model_Name.json>``` to edit tunable parameters:

    ```python scripts/train.py -mt ConvLSTM```

4. Validate the model on the val2 set (the wandb run name can be found in the wandb/run-XYZ/files/run_name.txt file):

    ```python scripts/validate.py -rn wandb/run/name -e epoch/to/validate/on```

5. Test on the iid test set:

    ```python scripts/validate.py -rn wandb/run/name -e epoch/to/validate/on -ts iid_test_split```

6. Evaluate your (ensemble of) model(s):

    ```python scripts/ensemble_score.py```

### Demos and Visualizations

Our model also comes with several notebooks/scripts for data visualization, diagnostics, etc.

```demos/model_demo.ipynb``` for exploring the dataset

```demos/data_observation_demo.ipynb``` for visualizing the dataset

```demos/draw_forecast.py ``` for visualizing predictions against ground truth

```demos/time_ndvi_plot.py``` for vizualizing the evolution of NDVI over time

```scripts/diagnosticate.py -rn wandb/run/name -e epoch/to/validate/on``` for visualizing the model's predictions

```scripts/optimize.py``` for optimizing hyperparameters (define your search space within script)


***Feel free to reach out to us if you still have any questions!***
