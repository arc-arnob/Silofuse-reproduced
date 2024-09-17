# Silofuse-Reproduced

# To-Do List

## Features to Implement
- [ ] **Rectify Issues in Silofuse with Adult & Covtype dataset**
- [ ] **Add a function to preprocess final data to run SDMetrics**
- [ ] **Add Theil's U test to replicate categorical scores**
  
## Error Handling
- [ ] **Add try-except blocks** to handle missing or misaligned columns between real and synthetic data.
- [ ] **Log exceptions** for resemblance functions to ensure robust execution.

## Installing Dependencies

Python version: 3.10

Create environment

```
conda create -n tabsyn python=3.10
conda activate tabsyn
```

Install pytorch
```
pip install torch torchvision torchaudio
```

or via conda
```
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
```

Install other dependencies

```
pip install -r requirements.txt

```

## Silofuse Setup
Place dataset in `data\raw`

## Training Models and generating synthetic data

```
python main.py --dataset [NAME_OF_DATSET]
```
For this POC [NAME_OF_DATSET] can be `bank`, `diabetes`, `abalone`, `cardio`, `adult`, `churn`, `covtype`, `heloc`, `intrusion`

The Generated data can be found in `data\external\[NAME_OF_DATSET]_synth_data.csv` without label.


## Evaluation of Generated Data

```
python evaluation.py --syn_data [SYNTHETIC_DATA_PATH] --real_data [REAL_DATA_PATH] --dataset diabetes
```
**Note: This is not perfect and some components fails to find coorelation.**

For confirmation I also used:
#### Density estimation of single column and pair-wise correlation ([link](https://docs.sdv.dev/sdmetrics/reports/quality-report/whats-included))

```
python eval/eval_density.py --dataname [NAME_OF_DATASET_IN_TAB_SYN] --model [METHOD_NAME] --path [PATH_TO_SYNTHETIC_DATA]
```
**Note to use this, follow the TabSyn tutorial and use real dataset from tabsyn subdirectory. You will need to preprocess the data as shown in Tabsyn section.**

## TabSyn Setup
## Preparing Datasets
### Using the datasets adopted in the paper

Download raw dataset:

```
python download_dataset.py
```

Process dataset:

```
python process_dataset.py
```

### Using your own dataset

First, create a directory for you dataset [NAME_OF_DATASET] in ./data:
```
cd data
mkdir [NAME_OF_DATASET]
```

Put the tabular data in .csv format in this directory ([NAME_OF_DATASET].csv). **The first row should be the header** indicating the name of each column, and the remaining rows are records.

Then, write a .json file ([NAME_OF_DATASET].json) recording the metadata of the tabular, covering the following information:
```
{
    "name": "[NAME_OF_DATASET]",
    "task_type": "[NAME_OF_TASK]", # binclass or regression
    "header": "infer",
    "column_names": null,
    "num_col_idx": [LIST],  # list of indices of numerical columns
    "cat_col_idx": [LIST],  # list of indices of categorical columns
    "target_col_idx": [list], # list of indices of the target columns (for MLE)
    "file_type": "csv",
    "data_path": "data/[NAME_OF_DATASET]/[NAME_OF_DATASET].csv"
    "test_path": null,
}
```
Put this .json file in the `Info` directory.

## Training Models

For baseline methods, use the following command for training:

```
python main.py --dataname [NAME_OF_DATASET] --method [NAME_OF_BASELINE_METHODS] --mode train
```

Options of [NAME_OF_DATASET]: adult, default, shoppers, magic, beijing, news
Options of [NAME_OF_BASELINE_METHODS]: smote, goggle, great, stasy, codi, tabddpm

For Tabsyn, use the following command for training:

```
# train VAE first
python main.py --dataname [NAME_OF_DATASET] --method vae --mode train

# after the VAE is trained, train the diffusion model
python main.py --dataname [NAME_OF_DATASET] --method tabsyn --mode train
```

## Tabular Data Synthesis

For Tabsyn, use the following command for synthesis:

```
python main.py --dataname [NAME_OF_DATASET] --method tabsyn --mode sample --save_path [PATH_TO_SAVE]

```

The default save path is "synthetic/[NAME_OF_DATASET]/[METHOD_NAME].csv"

## Evaluation
We evaluate the quality of synthetic data using metrics from various aspects.

#### Density estimation of single column and pair-wise correlation ([link](https://docs.sdv.dev/sdmetrics/reports/quality-report/whats-included))

```
python eval/eval_density.py --dataname [NAME_OF_DATASET] --model [METHOD_NAME] --path [PATH_TO_SYNTHETIC_DATA]
```
--------

