# Drought Impact Forecasting - EarthNet 2021

Climate change has a large impact on our environment :earth_americas:. We notice this all around the world, catastrophic events such as droughts occur more and more frequently. In this project, we develop various deep learning models for the EarthNet2021 challenge (https://www.earthnet.tech/), where our task is to predict future satellite images from past ones using features such as weather and elevation maps. With one of our models, a Peephole Convolutional LSTM, we manage to beat the current best model on the EarthNet challenge leaderboard. 

Feel free to reach out to us if you have any questions!


## User Guide

### Training, testing and evaluating the model

Prerequisites: Create a python environment as defined in ```config/dif_env.yml```

1. Dowload the dataset (train/iid/ood/extreme/seasonal splits):

    ```python scripts/data_retrieval.py directory/to/store/data split```

2. Collect the paths to the data:

    ```python scripts/data_collection.py -s training/dir -tf test/dir -d dir/to/store/paths -td no/of/training/samples/ -v1 no/of/val1/samples -v2 no/of/val2/samples```

3. Train the model (use ```config/Training.json``` to edit tunable parameters):

    ```python scripts/train.py ```

4. Validate the model on the val2 set (the wandb run name can be found in the wandb/run-XYZ/files/run_name.txt file):

    ```python scripts/validate.py -rn wandb/run/name -e epoch/to/validate/on```

5. Test on the iid test set:

    ```python scripts/validate.py -rn wandb/run/name -e epoch/to/validate/on -ts iid_test_split```

6. Evaluate you (ensemble of) model:

    ```python scripts/ensemble_score.py```

### Demos and Visualizations

Our model also comes with several notebooks/scripts for data visualization, diagnostics, etc.

```demos/model_demo.ipynb``` for exploring the dataset

```demos/data_observation_demo.ipynb``` for visualizing the dataset

```scripts/daignosticate.py -rn wandb/run/name -e epoch/to/validate/on``` for visualizing the model's predictions

