from sdv.metadata import SingleTableMetadata
import pandas as pd

def categorical_column_indices(metadata_dict):
    categorical_indices = []
    columns = metadata_dict.get('columns', {})
    column_names = list(columns.keys())[:-1]  # Exclude the last key
    for index, column_name in enumerate(column_names):
        column_data = columns[column_name]
        if column_data.get('sdtype') == 'categorical':
            categorical_indices.append(index)
    return categorical_indices

def generate_and_save_latent(model, source="../data/interim/bank_no_label.csv", path="../data/processed/bank,.csv"):
    DATA_PATH = source
    df = pd.read_csv(DATA_PATH, sep=",")
    actual_data = df #.iloc[:, :-1]
    # outcomes = df.iloc[:, -1]

    latents = []
    metadata = SingleTableMetadata()
    meta = metadata.detect_from_csv(source)

    discrete_columns = categorical_column_indices(metadata.to_dict())
    model.fit(actual_data, discrete_columns)
    latents = model.generate_latents(actual_data)
    latents_df = pd.DataFrame(latents) #(unbatched_latent)
    latents_df.to_csv(path, index=False)


# if __name__ == "__main__":
#     generate_and_save_latent(model, source)
