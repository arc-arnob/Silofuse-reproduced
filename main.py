
from pathlib import Path
import pandas as pd
from VAE.utils.dataframe_utils import split_columns
from  VAE.modeling.vae import TVAE
from VAE.modeling.generate_latent import generate_and_save_latent

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import tomli
import toml
import subprocess
import json
import argparse


def main(dataset=None, trained_model=None):

    if dataset is None:
        # The default block of code to run if no dataset is specified
        data = pd.read_csv("data/raw/bank.csv", sep=",")  # Raw Data
        _, label = split_columns(data, save_path="data/interim/bank_no_label.csv")
        no_label_data = pd.read_csv("data/interim/bank_no_label.csv", sep=",")
        model = TVAE(embedding_dim=no_label_data.shape[1], compress_dims=(32, 1024), decompress_dims=(1024, 32))
        generate_and_save_latent(model, source="data/interim/bank_no_label.csv", path="data/processed/bank_latent.csv")
        
        latent_check = pd.read_csv("data/processed/bank_latent.csv")
        print("CSV Read...")
        combined_df = pd.concat([latent_check, label], axis=1)
        combined_df.to_csv("data/processed/bank_latent_w_label.csv", index=False)

        return model

    if dataset == "bank_latent":
        # This block will run only if the dataset argument is provided
        data = pd.read_csv("data/processed/bank_latent_w_label.csv")
        X = data.iloc[:, :-1].values
        print("Debug1: Tabddpm latent feature", X.shape)
        y = data.iloc[:, -1].values
        print("Debug1: Tabddpm label latent feature", y.shape)
        # Perform further processing on X and y as needed
        print("Dataset processing completed.")

        idx = np.arange(0, X.shape[0])
        train_idx, test_idx = train_test_split(idx, test_size=0.3, random_state=42)
        val_idx, test_idx = train_test_split(test_idx, test_size=0.5, random_state=42)
        X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
        y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
        
        print("Train, Test, Val Split Complete")
        
        train_size, val_size, test_size = X_train.shape[0], X_val.shape[0], X_test.shape[0]
        print(train_size, val_size, test_size)

        # Ensure the directory exists
        output_dir = f'data/interim/{dataset}/'
        os.makedirs(output_dir, exist_ok=True)

        # Save X arrays
        np.save(os.path.join(output_dir, 'X_num_train.npy'), X_train)
        np.save(os.path.join(output_dir, 'X_num_val.npy'), X_val)
        np.save(os.path.join(output_dir, 'X_num_test.npy'), X_test)
        
        print("Interim Data X Saved in .npy format for Tabddpm")

        # Save y arrays
        np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(output_dir, 'y_val.npy'), y_val)
        np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
        
        print("Interim Data Y Saved in .npy format for Tabddpm")

        num_feature_size, cat_features_size = 13, 0 # These are default considering latent size of each client is 3 in VAE
        
        info_file = {
        "task_type": "binclass",
        "name": f"{dataset}",
        "id": f"{dataset}--id",
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "n_num_features": num_feature_size,
        "n_cat_features": cat_features_size
        }
        
        os.makedirs(f"data/interim/{dataset}", exist_ok=True)
        with open(f"data/interim//{dataset}/info.json", "w") as info_writer:
            json.dump(info_file, info_writer)

        os.makedirs(f"data/external/{dataset}", exist_ok=True)

        config_file = {
            'seed': 0,
            'parent_dir': f'data/external/{dataset}/',
            'real_data_path': f'data/interim/{dataset}/',
            'model_type': 'mlp',
            'num_numerical_features': num_feature_size,   # Set the number of numerical features here
            'device': 'cpu',  
            'model_params': {       # Change the denoising architecture here as per your liking
                'd_in': 13,
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

        with open(f"data/external/{dataset}/config.toml", 'w') as toml_file:
            toml.dump(config_file, toml_file)

        command = [
        "python", 
        "tabddpm/scripts/pipeline.py", 
        "--config", f"data/external/{dataset}/config.toml", 
        "--train", 
        "--sample"
        ]
        print("Starting subprocess now...")
        result = subprocess.run(command, capture_output=True, text=True)
        print(result.stderr)
    
    if dataset == "generate":
        if trained_model is None:
            raise ValueError("Model is not available. Ensure you have trained the model before generating data.")
        generated_latent = np.load("data/external/bank_latent/X_num_unnorm.npy")
        df = pd.DataFrame(generated_latent)
        df.to_csv("data/external/bank_synth_latent.csv", index=False)
        generated_data = trained_model.sample(5000, path="data/external/bank_synth_latent.csv")
        generated_data.to_csv("data/external/bank_synth_data.csv", index=False)

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to process data or run the dataset code.")
    parser.add_argument("--dataset", type=str, help="Specify the dataset to run, e.g., 'bank_latent'")
    args = parser.parse_args()

    trained_model = main(args.dataset)
    main("bank_latent")
    main("generate", trained_model=trained_model)

