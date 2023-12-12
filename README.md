# XGBoost in Compressed Space for Single-Cell Perturbations
This repository implements the solution for the "Open Problems – Single-Cell Perturbations" Kaggle Competition
using XGBoost model. 

# Context
Competition context: https://www.kaggle.com/competitions/open-problems-single-cell-perturbations/overview

Data context: https://www.kaggle.com/competitions/open-problems-single-cell-perturbations/data


# Installation

## Docker

To use a docker container, use the following commands

```
docker build -t xgb_scp_container . 
```
and then

```
docker run -p 8888:8888 xgb_scp_container
```

## Requirements 


Python 3.9 with the following packages

```
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.2.2
category_encoders==2.6.3
fastparquet==2023.10.1
pyarrow==14.0.1
xgboost==2.0.1
scipy==1.11.3
matplotlib==3.7.3
seaborn==0.12.2
```
You can use the command `pip install -r requirements.txt` to install the dependencies from the file `requirements.txt` and recreate the environment.

# Using 

The main file with results is `scp-xgboost-in-compressed-space.ipynb`  (copy of the kaggle notebook). 
To get only prediction using training and test files, run `xgb_script.py`. 

To run `.py` or `.ipynb` one must have `id_map.csv` and `de_train.parquet` in the same directory.


`python3 xgb_script.py` command gives the following output 


```
Files de_train.parquet and id_map.csv are imported successfully.
Data preprocessing is completed.
CPU Execution time: 7.2 seconds
Modeling complete.
CPU Execution time: 288.4 seconds
The prediction of the model is in submission.csv.
```

The output `submission.csv` file contains the model's prediction for `id_map.csv`. 





