import argparse
import subprocess
from VAE.utils import resemblance, postprocess
import pandas as pd
import numpy as np

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run resemblance measure between synthetic and real data.")
    parser.add_argument('--syn_data', type=str, required=True, help="Path to synthetic data file")
    parser.add_argument('--real_data', type=str, required=True, help="Path to real data file")
    parser.add_argument('--dataset', type=str, required=True, help="Dataset type, e.g., 'churn', 'other'")
    
    # Parse the arguments
    args = parser.parse_args()

    # Use the paths
    syn_data_path = args.syn_data
    real_data_path = args.real_data
    dataset_type = args.dataset

    # Load the data
    real_data = pd.read_csv(real_data_path)
    syn_data = pd.read_csv(syn_data_path)

    # If the dataset type is 'churn', apply the encoding logic
    if dataset_type.lower() == 'churn':
        # Apply get_dummies for 'Gender' and 'Geography' columns in both datasets
        real_data = pd.get_dummies(real_data, columns=['Gender', 'Geography'], dtype=int)
        syn_data = pd.get_dummies(syn_data, columns=['Gender', 'Geography'], dtype=int)

        # Apply the iloc logic to remove the last column in real_data (based on your notebook code)
        real_data = real_data.iloc[:, :-1]

    # If column count is mismatched between real and synthetic data, adjust real_data
    if real_data.shape[1] != syn_data.shape[1]:
        real_data = real_data.iloc[:, :-1]

    # Convert to numpy arrays and pass to the resemblance measure function
    _, n = resemblance.resemblance_measure(syn_data.to_numpy(), real_data.to_numpy())
    if n != 5:
        print("Some Resemblance Tests Failed running more advanced test...")
        latent_path = f"data/external/{dataset_type}_latent/y_train.npy"
        synth_data_path = f"data/external/{dataset_type}_synth_data.csv"
        output_path = f"tabsyn/synthetic/{dataset_type}/silo_{dataset_type}_gen.csv"
        # This has to be cleaned, this is a temp code as pwd would be set in subprocess
        tabsyn_path = f"synthetic/{dataset_type}/silo_{dataset_type}_gen.csv"
        postprocess.process_bin_class_data(dataset_type, latent_path, synth_data_path, output_path)

        # Run the eval_density script using subprocess
        try:
            result = subprocess.run(
                ['python', 'eval/eval_density.py', '--dataname', dataset_type, '--model', 'tabsyn', '--path', tabsyn_path],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd='tabsyn'
            )
            print(result.stdout.decode('utf-8'))
        except subprocess.CalledProcessError as e:
            print(f"Error while running eval_density.py: {e.stderr.decode('utf-8')}")
if __name__ == '__main__':
    main()