from sdv.metadata import SingleTableMetadata
import pandas as pd

def categorical_column_indices(metadata_dict):
    categorical_indices = []
    columns = metadata_dict.get('columns', {})
    print("Columns" ,columns)
    column_names = list(columns.keys()) # Exclude the last key
    for index, column_name in enumerate(column_names):
        print("Column name:::" ,column_name)
        column_data = columns[column_name]
        if column_data.get('sdtype') == 'categorical' or column_data.get('sdtype') == 'unknown':
            categorical_indices.append(index)
    return categorical_indices

def generate_and_save_latent(model, source="../data/interim/bank_no_label.csv", path="../data/processed/bank.csv"):
    try:
        DATA_PATH = source
        df = pd.read_csv(DATA_PATH, sep=",")
        actual_data = df

        latents = []
        metadata = SingleTableMetadata()
        
        # Detect metadata from CSV
        try:
            meta = metadata.detect_from_csv(source)
        except Exception as e:
            print(f"Error detecting metadata from CSV: {e}")
            return
        
        # Get discrete column indices
        try:
            discrete_columns = categorical_column_indices(metadata.to_dict())
            discrete_columns = df.columns[discrete_columns].tolist()
            print("##### Discrete Columns:", discrete_columns)
        except Exception as e:
            print(f"Error identifying discrete columns: {e}")
            return

        # Fit model
        try:
            model.fit(actual_data, discrete_columns)
        except Exception as e:
            print(f"Error fitting model", {e})
            return

        # Generate latent vectors
        try:
            latents = model.generate_latents(actual_data)
        except Exception as e:
            print(f"Error generating latents")
            return

        # Save latent vectors to CSV
        try:
            latents_df = pd.DataFrame(latents)
            latents_df.to_csv(path, index=False)
            print(f"Latent vectors saved to {path}")
        except Exception as e:
            print(f"Error saving latents to CSV: {e}")

    except FileNotFoundError as fnf_error:
        print(f"File not found: {fnf_error}")
    except pd.errors.ParserError as pe_error:
        print(f"Parsing error: {pe_error}")
    except Exception as e:
        print(f"An error occurred: {e}")


# if __name__ == "__main__":
#     generate_and_save_latent(model, source)
