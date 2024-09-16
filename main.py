
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

    # Bank Dataset

    if dataset == "bank":
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

        # Make num_feature_size a input parameter
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
        with open(f"data/interim/{dataset}/info.json", "w") as info_writer:
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
                'd_in': 13, # Make this Input parameter
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
    
    if dataset == "bank_generate":
        if trained_model is None:
            raise ValueError("Model is not available. Ensure you have trained the model before generating data.")
        generated_latent = np.load("data/external/bank_latent/X_num_unnorm.npy")
        df = pd.DataFrame(generated_latent)
        df.to_csv("data/external/bank_synth_latent.csv", index=False)
        generated_data = trained_model.sample(5000, path="data/external/bank_synth_latent.csv")
        generated_data.to_csv("data/external/bank_synth_data.csv", index=False)

    # Diabetes Dataset
    if dataset == "diabetes":
        print("Setting up and running for Diabetes Data...")
        # The default block of code to run if no dataset is specified
        data = pd.read_csv("data/raw/diabetes.csv", sep=",")  # Raw Data
        _, label = split_columns(data, save_path="data/interim/diabetes_no_label.csv", diabetes_data=True)
        no_label_data = pd.read_csv("data/interim/diabetes_no_label.csv", sep=",")
        model = TVAE(embedding_dim=no_label_data.shape[1], compress_dims=(32, 1024), decompress_dims=(1024, 32))
        generate_and_save_latent(model, source="data/interim/diabetes_no_label.csv", path="data/processed/diabetes_latent.csv")
        
        latent_check = pd.read_csv("data/processed/diabetes_latent.csv")
        print("CSV Read...")
        combined_df = pd.concat([latent_check, label], axis=1)
        combined_df.to_csv("data/processed/diabetes_latent_w_label.csv", index=False)

        return model

    if dataset == "diabetes_latent":
        # This block will run only if the dataset argument is provided
        data = pd.read_csv("data/processed/diabetes_latent_w_label.csv")
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

        # Make num_feature_size a input parameter
        num_feature_size, cat_features_size = 8, 0 # These are default considering latent size of each client is 3 in VAE
        
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
        with open(f"data/interim/{dataset}/info.json", "w") as info_writer:
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
                'd_in': 8, # Make this Input parameter
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
    
    if dataset == "diabetes_generate":
        print("Generating synthetic data now...")
        if trained_model is None:
            raise ValueError("Model is not available. Ensure you have trained the model before generating data.")
        generated_latent = np.load("data/external/diabetes_latent/X_num_unnorm.npy")
        df = pd.DataFrame(generated_latent)
        df.to_csv("data/external/diabetes_synth_latent.csv", index=False)
        generated_data = trained_model.sample(5000, path="data/external/diabetes_synth_latent.csv")
        generated_data.to_csv("data/external/diabetes_synth_data.csv", index=False)

    
    # Abalone Dataset
    if dataset == "abalone":
        print("Setting up and running for Abalone Data...")
        # The default block of code to run if no dataset is specified
        data = pd.read_csv("data/raw/abalone.csv", sep=",")  # Raw Data
        _, label = split_columns(data, save_path="data/interim/abalone_no_label.csv")
        no_label_data = pd.read_csv("data/raw/abalone.csv", sep=",")
        model = TVAE(embedding_dim=data.shape[1], compress_dims=(32, 1024), decompress_dims=(1024, 32))
        generate_and_save_latent(model, source="data/raw/abalone.csv", path="data/processed/abalone_latent.csv")
        
        latent_check = pd.read_csv("data/processed/abalone_latent.csv")
        latent_check.to_csv("data/processed/abalone_latent_w_label.csv", index=False)
        # print("CSV Read...")
        # combined_df = pd.concat([latent_check, label], axis=1)
        # combined_df.to_csv("data/processed/abalone_latent_w_label.csv", index=False)

        return model

    if dataset == "abalone_latent":
        # This block will run only if the dataset argument is provided
        data = pd.read_csv("data/processed/abalone_latent_w_label.csv")
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

        # Make num_feature_size a input parameter
        num_feature_size, cat_features_size = 8, 0 # These are default considering latent size of each client is 3 in VAE
        
        info_file = {
        "task_type": "regression",
        "name": f"{dataset}",
        "id": f"{dataset}--id",
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "n_num_features": num_feature_size,
        "n_cat_features": cat_features_size
        }
        
        os.makedirs(f"data/interim/{dataset}", exist_ok=True)
        with open(f"data/interim/{dataset}/info.json", "w") as info_writer:
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
                'd_in': 8, # Make this Input parameter
                'num_classes': 0,
                'is_y_cond': False,
                'rtdl_params': {
                    'd_layers': [
                        256,
                        128
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
                    'lr': 0.00027761965839603165,
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
                'batch_size': 10000,
                'seed': 0
            },
            'eval': {
                'type': {
                    'eval_model': 'mlp',
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
    
    if dataset == "abalone_generate":
        print("Generating synthetic data now...")
        if trained_model is None:
            raise ValueError("Model is not available. Ensure you have trained the model before generating data.")
        generated_latent = np.load("data/external/abalone_latent/X_num_unnorm.npy")
        df = pd.DataFrame(generated_latent)
        df.to_csv("data/external/abalone_synth_latent.csv", index=False)
        generated_data = trained_model.sample(5000, path="data/external/abalone_synth_latent.csv")
        generated_data.to_csv("data/external/abalone_synth_data.csv", index=False)

    # Cardio
    if dataset == "cardio":
        print("Setting up and running for cardio Data...")
        # The default block of code to run if no dataset is specified
        data = pd.read_csv("data/raw/cardio.csv", sep=";")  # Raw Data
        _, label = split_columns(data, save_path="data/interim/cardio_no_label.csv")
        no_label_data = pd.read_csv("data/interim/cardio_no_label.csv", sep=",")
        model = TVAE(embedding_dim=no_label_data.shape[1], compress_dims=(32, 1024), decompress_dims=(1024, 32))
        generate_and_save_latent(model, source="data/interim/cardio_no_label.csv", path="data/processed/cardio_latent.csv")
        
        latent_check = pd.read_csv("data/processed/cardio_latent.csv")
        print("CSV Read...")
        combined_df = pd.concat([latent_check, label], axis=1)
        combined_df.to_csv("data/processed/cardio_latent_w_label.csv", index=False)

        return model

    if dataset == "cardio_latent":
        # This block will run only if the dataset argument is provided
        data = pd.read_csv("data/processed/cardio_latent_w_label.csv")
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

        # Make num_feature_size a input parameter
        num_feature_size, cat_features_size = 12, 0 # These are default considering latent size of each client is 3 in VAE
        
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
        with open(f"data/interim/{dataset}/info.json", "w") as info_writer:
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
                'd_in': 12, # Make this Input parameter
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
    
    if dataset == "cardio_generate":
        print("Generating synthetic data now...")
        if trained_model is None:
            raise ValueError("Model is not available. Ensure you have trained the model before generating data.")
        generated_latent = np.load("data/external/cardio_latent/X_num_unnorm.npy")
        df = pd.DataFrame(generated_latent)
        df.to_csv("data/external/cardio_synth_latent.csv", index=False)
        generated_data = trained_model.sample(5000, path="data/external/cardio_synth_latent.csv")
        generated_data.to_csv("data/external/cardio_synth_data.csv", index=False)
    
    # Adult Dataset
    if dataset == "adult":
        print("Setting up and running for adult Data...")
        # The default block of code to run if no dataset is specified
        data = pd.read_csv("data/raw/adult.csv", sep=",")  # Raw Data
        _, label = split_columns(data, save_path="data/interim/adult_no_label.csv", adult=True)
        no_label_data = pd.read_csv("data/interim/adult_no_label.csv", sep=",")
        model = TVAE(embedding_dim=no_label_data.shape[1], compress_dims=(32, 1024), decompress_dims=(1024, 32))
        generate_and_save_latent(model, source="data/interim/adult_no_label.csv", path="data/processed/adult_latent.csv")
        
        latent_check = pd.read_csv("data/processed/adult_latent.csv")
        print("CSV Read...")
        combined_df = pd.concat([latent_check, label], axis=1)
        combined_df.to_csv("data/processed/adult_latent_w_label.csv", index=False)

        return model

    if dataset == "adult_latent":
        # This block will run only if the dataset argument is provided
        data = pd.read_csv("data/processed/adult_latent_w_label.csv")
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

        # Make num_feature_size a input parameter
        num_feature_size, cat_features_size = 12, 0 # These are default considering latent size of each client is 3 in VAE
        
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
        with open(f"data/interim/{dataset}/info.json", "w") as info_writer:
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
                'd_in': 14, # Make this Input parameter
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
    
    if dataset == "adult_generate":
        print("Generating synthetic data now...")
        if trained_model is None:
            raise ValueError("Model is not available. Ensure you have trained the model before generating data.")
        generated_latent = np.load("data/external/adult_latent/X_num_unnorm.npy")
        df = pd.DataFrame(generated_latent)
        df.to_csv("data/external/adult_synth_latent.csv", index=False)
        generated_data = trained_model.sample(5000, path="data/external/adult_synth_latent.csv")
        generated_data.to_csv("data/external/adult_synth_data.csv", index=False)

    # Churn Dataset
    # Had to remove "Surname" for 86% match
    if dataset == "churn":
        print("Setting up and running for churn Data...")
        # The default block of code to run if no dataset is specified
        data = pd.read_csv("data/raw/churn.csv", sep=",")  # Raw Data
        _, label = split_columns(data, save_path="data/interim/churn_no_label.csv")
        no_label_data = pd.read_csv("data/interim/churn_no_label.csv", sep=",")
        model = TVAE(embedding_dim=no_label_data.shape[1], compress_dims=(32, 1024), decompress_dims=(1024, 32))
        generate_and_save_latent(model, source="data/interim/churn_no_label.csv", path="data/processed/churn_latent.csv")
        
        latent_check = pd.read_csv("data/processed/churn_latent.csv")
        print("CSV Read...")
        combined_df = pd.concat([latent_check, label], axis=1)
        combined_df.to_csv("data/processed/churn_latent_w_label.csv", index=False)

        return model

    if dataset == "churn_latent":
        # This block will run only if the dataset argument is provided
        data = pd.read_csv("data/processed/churn_latent_w_label.csv")
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

        # Make num_feature_size a input parameter
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
        with open(f"data/interim/{dataset}/info.json", "w") as info_writer:
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
                'd_in': 13, # Make this Input parameter
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
    
    if dataset == "churn_generate":
        print("Generating synthetic data now...")
        if trained_model is None:
            raise ValueError("Model is not available. Ensure you have trained the model before generating data.")
        generated_latent = np.load("data/external/churn_latent/X_num_unnorm.npy")
        df = pd.DataFrame(generated_latent)
        df.to_csv("data/external/churn_synth_latent.csv", index=False)
        generated_data = trained_model.sample(5000, path="data/external/churn_synth_latent.csv")
        generated_data.to_csv("data/external/churn_synth_data.csv", index=False)
    
    # Covtype Dataset
    if dataset == "covtype":
        print("Setting up and running for covtype Data...")
        # The default block of code to run if no dataset is specified
        data = pd.read_csv("data/raw/covtype.csv", sep=",")  # Raw Data
        _, label = split_columns(data, save_path="data/interim/covtype_no_label.csv")
        no_label_data = pd.read_csv("data/interim/covtype_no_label.csv", sep=",")
        model = TVAE(embedding_dim=no_label_data.shape[1], compress_dims=(32, 1024), decompress_dims=(1024, 32))
        generate_and_save_latent(model, source="data/interim/covtype_no_label.csv", path="data/processed/covtype_latent.csv")
        
        latent_check = pd.read_csv("data/processed/covtype_latent.csv")
        print("CSV Read...")
        combined_df = pd.concat([latent_check, label], axis=1)
        combined_df.to_csv("data/processed/covtype_latent_w_label.csv", index=False)

        return model

    if dataset == "covtype_latent":
        # This block will run only if the dataset argument is provided
        data = pd.read_csv("data/processed/covtype_latent_w_label.csv")
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

        # Make num_feature_size a input parameter
        num_feature_size, cat_features_size = 54, 0 # These are default considering latent size of each client is 3 in VAE
        
        info_file = {
        "task_type": "multiclass",
        "name": f"{dataset}",
        "id": f"{dataset}--id",
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "n_num_features": num_feature_size,
        "n_cat_features": cat_features_size
        }
        
        os.makedirs(f"data/interim/{dataset}", exist_ok=True)
        with open(f"data/interim/{dataset}/info.json", "w") as info_writer:
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
                'd_in': 54, # Make this Input parameter
                'num_classes': 7,
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
    
    if dataset == "covtype_generate":
        print("Generating synthetic data now...")
        if trained_model is None:
            raise ValueError("Model is not available. Ensure you have trained the model before generating data.")
        generated_latent = np.load("data/external/covtype_latent/X_num_unnorm.npy")
        df = pd.DataFrame(generated_latent)
        df.to_csv("data/external/covtype_synth_latent.csv", index=False)
        generated_data = trained_model.sample(5000, path="data/external/covtype_synth_latent.csv")
        generated_data.to_csv("data/external/covtype_synth_data.csv", index=False)

    # Heloc Dataset
    if dataset == "heloc":
        print("Setting up and running for heloc Data...")
        # The default block of code to run if no dataset is specified
        data = pd.read_csv("data/raw/heloc.csv", sep=",")  # Raw Data
        _, label = split_columns(data, save_path="data/interim/heloc_no_label.csv", heloc=True)
        no_label_data = pd.read_csv("data/interim/heloc_no_label.csv", sep=",")
        model = TVAE(embedding_dim=no_label_data.shape[1], compress_dims=(32, 1024), decompress_dims=(1024, 32))
        generate_and_save_latent(model, source="data/interim/heloc_no_label.csv", path="data/processed/heloc_latent.csv")
        
        latent_check = pd.read_csv("data/processed/heloc_latent.csv")
        print("CSV Read...")
        combined_df = pd.concat([latent_check, label], axis=1)
        combined_df.to_csv("data/processed/heloc_latent_w_label.csv", index=False)

        return model

    if dataset == "heloc_latent":
        # This block will run only if the dataset argument is provided
        data = pd.read_csv("data/processed/heloc_latent_w_label.csv")
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

        # Make num_feature_size a input parameter
        num_feature_size, cat_features_size = 23, 0 # These are default considering latent size of each client is 3 in VAE
        
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
        with open(f"data/interim/{dataset}/info.json", "w") as info_writer:
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
                'd_in': 23, # Make this Input parameter
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
    
    if dataset == "heloc_generate":
        print("Generating synthetic data now...")
        if trained_model is None:
            raise ValueError("Model is not available. Ensure you have trained the model before generating data.")
        generated_latent = np.load("data/external/heloc_latent/X_num_unnorm.npy")
        df = pd.DataFrame(generated_latent)
        df.to_csv("data/external/heloc_synth_latent.csv", index=False)
        generated_data = trained_model.sample(5000, path="data/external/heloc_synth_latent.csv")
        generated_data.to_csv("data/external/heloc_synth_data.csv", index=False)

    # Intrusion Dataset
    if dataset == "intrusion":
        print("Setting up and running for intrusion Data...")
        # The default block of code to run if no dataset is specified
        data = pd.read_csv("data/raw/intrusion.csv", sep=",")  # Raw Data
        _, label = split_columns(data, save_path="data/interim/intrusion_no_label.csv", intrusion=True)
        no_label_data = pd.read_csv("data/interim/intrusion_no_label.csv", sep=",")
        model = TVAE(embedding_dim=no_label_data.shape[1], compress_dims=(32, 1024), decompress_dims=(1024, 32))
        generate_and_save_latent(model, source="data/interim/intrusion_no_label.csv", path="data/processed/intrusion_latent.csv")
        
        latent_check = pd.read_csv("data/processed/intrusion_latent.csv")
        print("CSV Read...")
        combined_df = pd.concat([latent_check, label], axis=1)
        combined_df.to_csv("data/processed/intrusion_latent_w_label.csv", index=False)

        return model

    if dataset == "intrusion_latent":
        # This block will run only if the dataset argument is provided
        data = pd.read_csv("data/processed/intrusion_latent_w_label.csv")
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

        # Make num_feature_size a input parameter
        num_feature_size, cat_features_size = 40, 0 # These are default considering latent size of each client is 3 in VAE
        
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
        with open(f"data/interim/{dataset}/info.json", "w") as info_writer:
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
                'd_in': 40, # Make this Input parameter
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
    
    if dataset == "intrusion_generate":
        print("Generating synthetic data now...")
        if trained_model is None:
            raise ValueError("Model is not available. Ensure you have trained the model before generating data.")
        generated_latent = np.load("data/external/intrusion_latent/X_num_unnorm.npy")
        df = pd.DataFrame(generated_latent)
        df.to_csv("data/external/intrusion_synth_latent.csv", index=False)
        generated_data = trained_model.sample(5000, path="data/external/intrusion_synth_latent.csv")
        generated_data.to_csv("data/external/intrusion_synth_data.csv", index=False)


        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to process data or run the dataset code.")
    parser.add_argument("--dataset", type=str, help="Specify the dataset to run, e.g., 'bank_latent'")
    args = parser.parse_args()

    # Run: Done
    # Resemblance: Matched
    if args.dataset == 'bank':
        trained_model = main(args.dataset)
        main("bank_latent")
        main("bank_generate", trained_model=trained_model)
    
    # Run: Done
    # Resemblance: Matched
    # tabsyn: 91
    elif args.dataset == 'diabetes':
        trained_model = main(args.dataset)
        main("diabetes_latent")
        main("diabetes_generate", trained_model=trained_model)
    
    # Run: Done
    # Resemblance: 90% Matched
    # Tabsyn: TBD
    elif args.dataset == 'abalone':
        trained_model = main(args.dataset)
        main("abalone_latent")
        main("abalone_generate", trained_model=trained_model)

    # Run: Done
    # Resemblance: Matched
    elif args.dataset == 'cardio':
        trained_model = main(args.dataset)
        main("cardio_latent")
        main("cardio_generate", trained_model=trained_model)
    
    # Run: Running...
    # Resemblance: TBD
    # AttributeError: 'NoneType' object has no attribute 'inverse_transform'
    # Tabsyn: 98%
    elif args.dataset == 'adult':
        trained_model = main(args.dataset)
        main("adult_latent")
        main("adult_generate", trained_model=trained_model)
    
    # Run: Done
    # Resemblance: Matched, correlation_similarity X
    # Tabsyn..
    elif args.dataset == 'churn':
        trained_model = main(args.dataset)
        main("churn_latent")
        main("churn_generate", trained_model=trained_model)

    # Run: Running... 4-6hrs Issue with multiclass...
    # Resemblance: TBD 
    # Tabsyn:  Multiclass not Supported
    elif args.dataset == 'covtype':
        trained_model = main(args.dataset)
        main("covtype_latent")
        main("covtype_generate", trained_model=trained_model)

    # Run: Done... but column similarity doesnt work...
    # Resemblance: 83
    elif args.dataset == 'heloc':
        trained_model = main(args.dataset)
        main("heloc_latent")
        main("heloc_generate", trained_model=trained_model)

    # Run: Done
    # Resemblance: 
    # Tabsyn: 97
    elif args.dataset == 'intrusion':
        trained_model = main(args.dataset)
        main("intrusion_latent")
        main("intrusion_generate", trained_model=trained_model)


