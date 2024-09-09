from pathlib import Path
import pandas as pd
from utils.dataframe_utils import split_columns
from  vae import TVAE
from generate_latent import generate_and_save_latent

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import tomli
import toml
import subprocess
import json

# import typer
# from loguru import logger
# from tqdm import tqdm

# from VAE.config import MODELS_DIR, PROCESSED_DATA_DIR

# app = typer.Typer()


# @app.command()
# def main(
#     # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
#     features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
#     model_path: Path = MODELS_DIR / "model.pkl",
#     predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
#     # -----------------------------------------
# ):
#     # ---- REPLACE THIS WITH YOUR OWN CODE ----
#     logger.info("Performing inference for model...")
#     for i in tqdm(range(10), total=10):
#         if i == 5:
#             logger.info("Something happened for iteration 5.")
#     logger.success("Inference complete.")
#     # -----------------------------------------


if __name__ == "__main__":
    data = pd.read_csv("../data/raw/bank.csv", sep=",")
    _, label = split_columns(data)
    no_label_data = pd.read_csv("../data/interim/bank_no_label.csv", sep=",")
    model = TVAE(embedding_dim=no_label_data.shape[1], compress_dims=(32,1024), decompress_dims=(1024,32))
    generate_and_save_latent(model)

    dataset = "bank_latent"
    data = pd.read_csv("../../data/processed/bank.csv")
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    idx = np.arange(0, X.shape[0])
    train_idx, test_idx = train_test_split(idx, test_size =0.3, random_state=42)
    val_idx, test_idx,  = train_test_split(test_idx, test_size =0.5, random_state=42)
    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    train_size, valu_size, test_size = X_train.shape[0], X_val.shape[0], X_test.shape[0]
    print(train_size, valu_size, test_size)

    np.save(f'../data/{dataset}/X_num_train.npy', X_train)
    np.save(f'../data/{dataset}/X_num_val.npy', X_val)
    np.save(f'../data/{dataset}/X_num_test.npy', X_test)

    np.save(f'../data/{dataset}/y_train.npy', y_train)
    np.save(f'../data/{dataset}/y_val.npy', y_val)
    np.save(f'../data/{dataset}/y_test.npy', y_test)

    num_feature_size, cat_features_size = 14, 0 # These are default considering latent size of each client is 3 in VAE
    
    info_file = {
    "task_type": "binclass",
    "name": f"{dataset}",
    "id": f"{dataset}--id",
    "train_size": train_size,
    "val_size": valu_size,
    "test_size": test_size,
    "n_num_features": num_feature_size,
    "n_cat_features": cat_features_size
    }
    
    os.makedirs(f"../data/{dataset}", exist_ok=True)

    config_file = {
        'seed': 0,
        'parent_dir': f'../exp/{dataset}/',
        'real_data_path': f'../data/{dataset}/',
        'model_type': 'mlp',
        'num_numerical_features': num_feature_size,   # Set the number of numerical features here
        'device': 'cpu',  
        'model_params': {       # Change the denoising architecture here as per your liking
            'd_in': 15, #ASK What is this.
            'num_classes': 2,
            'is_y_cond': True,
            'rtdl_params': {
                'd_layers': [
                    128,
                    512
                ],
                'dropout': 0.0
            }
        },
        'diffusion_params': {
            'num_timesteps': 1000,
            'gaussian_loss_type': 'mse'
        },
        'train': {
            'main': {
                'steps': 30000,
                'lr': 1.1510940031144828e-05,
                'weight_decay': 0.0,
                'batch_size': 4096
            },
            'T': {
                'seed': 0,
                'normalization': 'quantile',
                'num_nan_policy': '__none__',
                'cat_nan_policy': '__none__',
                'cat_min_frequency': '__none__',
                'cat_encoding': '__none__',
                'y_policy': 'default'
            }
        },
        'sample': {
            'num_samples': 5000,
            'batch_size': 500,
            'seed': 0
        },
        'eval': {
            'type': {
                'eval_model': 'catboost',
                'eval_type': 'synthetic'
            },
            'T': {
                'seed': 0,
                'normalization': '__none__',
                'num_nan_policy': '__none__',
                'cat_nan_policy': '__none__',
                'cat_min_frequency': '__none__',
                'cat_encoding': '__none__',
                'y_policy': 'default'
            }
        }
    }

    with open(f"../exp/{dataset}/config.toml", 'w') as toml_file:
        toml.dump(config_file, toml_file)

    command = [
    "python", 
    "pipeline.py", 
    "--config", f"../exp/{dataset}/config.toml", 
    "--train", 
    "--sample"
    ]
    result = subprocess.run(command, capture_output=True, text=True)

