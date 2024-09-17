import numpy as np
import pandas as pd
import os

# Dictionary to map dataset names to their types (Bin Class or others)
DATASET_TYPES = {
    "cardio": "Bin Class",
    "churn": "Bin Class",
    "diabetes": "Bin Class",
    "heloc": "Bin Class",
    "intrusion": "Bin Class",
    "adult": "Bin Class",
    # Add more datasets as necessary
}

def process_bin_class_data(dataset_name, latent_path, synth_data_path, output_path):
    """
    Function to process binary classification datasets.
    
    Parameters:
        dataset_name (str): Name of the dataset (e.g., 'cardio', 'churn').
        latent_path (str): Path to the latent label file (e.g., y_train.npy).
        synth_data_path (str): Path to the synthetic data CSV file.
        output_path (str): Path to save the final output CSV with labels.
    
    Returns:
        None: Saves the processed file to the output path.
    """
    # Check if dataset is binary classification
    if DATASET_TYPES.get(dataset_name, "") != "Bin Class":
        print(f"Skipping {dataset_name} as it is not a binary classification dataset.")
        return
    
    # Load the latent labels (y_train)
    data_np = np.load(latent_path)
    np_df = pd.DataFrame(data_np, columns=[dataset_name])
    
    # Load the synthetic data
    synth_data = pd.read_csv(synth_data_path)
    
    # Concatenate the latent labels with the synthetic data
    final_data = pd.concat([synth_data, np_df], axis=1)
    
    # Save the final data to CSV
    final_data.to_csv(output_path, index=False)
    print(f"Processed and saved: {output_path}")