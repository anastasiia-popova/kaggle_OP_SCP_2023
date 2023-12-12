# Anastasiia Popova ppva.nastya@proton.me

# --------------------- Libraries --------------------- 
import numpy as np 
import pandas as pd 
import random
import time
import os

from sklearn.model_selection import KFold
from sklearn.decomposition import TruncatedSVD 

import xgboost as xgb
import category_encoders as ce

import scipy.stats
from scipy.stats import norm, multinomial 

import warnings
warnings.filterwarnings('ignore')

# --------------------- Data Import --------------------- 

fn_1 = 'de_train.parquet'
fn_2 = 'id_map.csv'
df = pd.read_parquet(fn_1) 
df_id_map = pd.read_csv(fn_2)

features = ["cell_type", "sm_name" ]
target = [value for value in df.columns if value not in ["sm_lincs_id", "SMILES", "control"] + features ]

print(f"Files {fn_1} and {fn_2} are imported successfully.")

# --------------------- Metric  --------------------- 

def metric(y, y_hat):
    """
    Calculates the Mean Row-wise Root Mean Squared Error (MR_RMSE) between two matrices.

    Args:
    - y (array-like): Ground truth matrix.
    - y_hat (array-like): Predicted matrix.

    Returns:
    float: Mean Row-wise Root Mean Squared Error (MR_RMSE) between y and y_hat.

    Raises:
    ValueError: If the input matrices y and y_hat do not have the same shape.

    """
    y = np.array(y)
    y_hat = np.array(y_hat)
    
    # Check if the input matrices have the same shape
    if y_hat.shape != y.shape:
        raise ValueError("Input matrices must have the same shape")

    # Calculate the squared differences element-wise
    squared_diff = (y - y_hat)**2

    # Calculate the mean row-wise RMSE
    rowwise_rmse = np.sqrt(np.mean(squared_diff, axis=1))

    # Calculate the mean of row-wise RMSE values
    mr_rmse = np.mean(rowwise_rmse, axis=0)
            
    return mr_rmse


# --------------------- Data Prepocessing ---------------------

def obtain_reduced_data(df, reducer, encoder, t_score = False):
    
    """
    Obtains reduced data for a given DataFrame using dimensionality reduction and encoding techniques.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data.
        reducer: An instance of a dimensionality reduction model.
        encoder: An instance of an encoding model for categorical features.

    Returns:
        pd.DataFrame: Reduced and encoded feature matrix.
        pd.DataFrame: Original reduced feature matrix.
    """
    
    # Extracting the target variable for dimensionality reduction
    if t_score:
        X = de_to_t_score(df[target])
    else: 
        X = df[target] 

    # Defining the aggregation feature for grouping
    aggr_feature = "sm_name"
    
    # Transforming the data using the provided dimensionality reduction technique
    Xr = reducer.transform(X)
    
    # Extracting the number of components after dimensionality reduction
    n_components = Xr.shape[1]
    
    # Extracting unique drug and cell type names
    drugs = list(df[aggr_feature].unique())
    cells = list(df["cell_type"].unique())
    
    one_hot_names = drugs + cells 
    
    # Creating names for zip columns
    zip_names = [i for i in range(n_components)]

    X_zip = pd.DataFrame(np.zeros((len(df),len(one_hot_names)+len(zip_names))),
                                columns = one_hot_names + zip_names)
    
    # Initializing DataFrames for the reduced and encoded features
    y_zip = pd.DataFrame(Xr)
    
    # Looping through drugs and cells to calculate the mean of the cell types for 
    # each drug in compressed space
    ind = 0
    for i,d in enumerate(drugs):

        df_one_drug = df[df[aggr_feature]==d].copy()
        observations = df_one_drug.index.to_list()

        cells_ = list(df_one_drug["cell_type"].unique())

        for j,c in enumerate(cells_):
            
            X_zip.loc[ind, d] = 1 
            X_zip.loc[ind, [k for k in cells_ if k!= c]] = 1
   
            rest_indexes = df_one_drug[df_one_drug["cell_type"] != c].index.to_list()

            X_zip.loc[ind, zip_names] = np.mean(Xr[rest_indexes,:], axis=0)

            ind+=1
            
    # Extracting categorical features for encoding
    X_categ = df[features]
    
    # Initializing the array for the encoded categorical features
    X_categ_trained = np.zeros((len(df),2*n_components))
    
    # Looping through components for encoding categorical features
    for n in range(n_components):

        cat_encoding_one_component = encoder.fit_transform(X_categ, Xr[:,n])

        X_categ_trained[:,2*n] = cat_encoding_one_component["cell_type"]
        X_categ_trained[:,2*n+1] = cat_encoding_one_component["sm_name"]
    
    # Concatenating the reduced and encoded features into a single DataFrame
    X_zip_ = pd.concat([X_zip ,pd.DataFrame(X_categ_trained, columns=[f"c_{i}" for i in range(2*n_components)])], axis=1)
        
    return X_zip_, y_zip

# from https://www.kaggle.com/code/ambrosm/scp-eda-which-makes-sense
def de_to_t_score(de):
    """Convert log10pvalues to t-scores
    
    Parameter:
    de: array or DataFrame of log10pvalues
    
    Return value:
    t_score: array or DataFrame of t-scores
    """
    p_value = 10 ** (-np.abs(de))
    return - scipy.stats.t.ppf(p_value / 2, df=420) * np.sign(de)
#     return - norm.ppf(p_value / 2) * np.sign(de)


# from https://www.kaggle.com/code/ambrosm/scp-eda-which-makes-sense
def t_score_to_de(t_score):
    """Convert t-scores to log10pvalues (inverse of de_to_t_score)
    
    Parameter:
    t_score: array or DataFrame of t-scores
    
    Return value:
    de: array or DataFrame of log10pvalues
    """
    p_value = scipy.stats.t.cdf(- np.abs(t_score), df=420) * 2
#     p_value = norm.cdf(- np.abs(t_score)) * 2
    p_value = p_value.clip(1e-180, None)
    return - np.log10(p_value) * np.sign(t_score)

def drug_av_zip(df, idtab, reducer, encoder, t_score=False):
    """
    Create a feature encoding, applying a drug-specific averaging in compressed space to a given dataframe.

    Parameters:
    - df (pd.DataFrame): Input training dataframe.
    - idtab (pd.DataFrame): Reference dataframe for encoding features.
    - reducer: A dimensionality reduction model used for compression.

    Returns:
    pd.DataFrame: A new dataframe containing feature encoding.
    """
    
    # Extracting the target variable for dimensionality reduction
    if t_score:
        X = de_to_t_score(df[target])
    else:
        X = df[target] 
    
    # Defining the aggregation feature for grouping
    arrg_feature = "sm_name"
    
    # Transforming the data using the provided dimensionality reduction technique
    Xr = reducer.transform(X)
    
    # Extracting the number of components after dimensionality reduction
    n_components = Xr.shape[1]
    
    # Extracting unique drug names
    drugs = list(df[arrg_feature].unique())
    
    X_av_drug = np.zeros((len(drugs), n_components))
    
    # Creating names for zip columns
    zip_names = [i for i in range(n_components)]
    
    df_ = pd.DataFrame( columns= [arrg_feature] + zip_names)
    
    df_[arrg_feature] = drugs
    
    # Looping through drugs to compute the mean of the cell types for 
    # each drug in compressed space
    
    for i,d in enumerate(drugs):
        observations = df[df[arrg_feature]==d].index.to_list()
    
        X_av_drug[i,:] =  np.mean(Xr[observations,:], axis=0)

        
    df_[zip_names] =  X_av_drug.copy()
    
    X_zip = idtab[features].merge(df_, how='left', on="sm_name")
            
    #---------- categorical features encoding --------------------------#
    X_categ = idtab[features]

    X_categ_trained = np.zeros((len(idtab),2*n_components))
    
    # Looping through components for encoding categorical features
    for n in range(n_components):

        encoder.fit(df[features], Xr[:,n])

        cat_encoding_one_component = encoder.transform(X_categ, X_zip[n])

        X_categ_trained[:,2*n] = cat_encoding_one_component["cell_type"]
        X_categ_trained[:,2*n+1] = cat_encoding_one_component["sm_name"]

     # Concatenating the reduced and encoded features into a single DataFrame
    X_zip_ =  pd.concat([X_zip, pd.DataFrame(X_categ_trained,  columns=[f"c_{i}" for i in range(2*n_components)])], axis=1) 
    
    cells = list(df["cell_type"].unique())
    
    one_hot_names = drugs + cells
    
    # This loop iterates over each element in one_hot_names and inserts a new 
    # column with that name filled with zeros into the DataFrame X_zip_. 
    # The "+2" in the insert function suggests that the new columns are inserted 
    # starting from the third column.
    for i in range(len(one_hot_names)):
        X_zip_.insert(i+2, one_hot_names[i], np.zeros(X_zip_.shape[0]))
        
    # One-Hot Encoding for Drugs and Cell Types
    
    # It then sets the value in the DataFrame at the intersection of the current
    # row and the column with the name d to 1, indicating the presence of that drug.
    # Similarly, it sets the values in columns corresponding to cell types not equal 
    # to c for the same drug d to 1, indicating the presence of that drug for those other cell types.
    for ind in X_zip_.index:
        d = X_zip_.loc[ind, "sm_name"]
        X_zip_.loc[ind, d] = 1 
        
        c = X_zip_.loc[ind, "cell_type"]
        
        cells_in_train = df[(df["cell_type"]!=c) & (df["sm_name"]==d)]["cell_type"]
        X_zip_.loc[ind, cells_in_train] = 1 
        
    X_zip_.drop(columns=features, inplace=True)
    
    return X_zip_

# -------------------------------------------------------

n_components = 36
reducer = TruncatedSVD(n_components=n_components, n_iter=7, random_state=42)
encoder = ce.TargetEncoder(smoothing=8)
t_score = False

start = time.process_time()

if t_score:
    X = de_to_t_score(df[target])
else: 
    X = df[target] 

reducer.fit(X)

X_test = drug_av_zip(df, df_id_map, reducer, encoder, t_score=t_score)

X_zip, y_zip = obtain_reduced_data(df, reducer, encoder, t_score=t_score)

end = time.process_time()

print("Data preprocessing is completed.")
print("CPU Execution time:", round(end - start,1), "seconds")

# --------------------- Modelling ---------------------

def tsvd_drug_av_inv(df_train, df_test, reducer):
    """
    Applies dimensionality reduction to a dataset, computes the average of each drug's reduced representation,
    inversely transforms the averaged representation, and predicts target values for a test dataset.

    Args:
    - df_train (pandas.DataFrame): Training dataset.
    - df_test (pandas.DataFrame): Test dataset.
    - reducer: Dimensionality reduction model with 'transform' and 'inverse_transform' methods.

    Returns:
    pandas.Series: Predicted target values for the test dataset.
    """
    X = df[target] 
    aggr_feature = "sm_name"
    
    # Apply dimensionality reduction
    Xr = reducer.transform(X)
    
    n_components = Xr.shape[1]
    
    # Get unique drugs from the dataset
    drugs = df[aggr_feature].unique()
    
    # Initialize an array to store the average representation for each drug
    X_av_drug = np.zeros((len(drugs), n_components))
    
    # Create a DataFrame to store averaged drug representations
    df_ = pd.DataFrame(columns=[aggr_feature] + target)
    df_[aggr_feature] = drugs
    
    # Compute the average representation for each drug
    for i, d in enumerate(drugs):
        observations = df[df[aggr_feature] == d].index.to_list()
        X_av_drug[i, :] = np.mean(Xr[observations, :], axis=0)

    # Inversely transform the averaged drug representations
    X_av_drug_inv = reducer.inverse_transform(X_av_drug)
    
    # Update the DataFrame with the averaged and inversely transformed drug representations
    df_[target] = X_av_drug_inv.copy()
    
    # Merge averaged drug representations with the test dataset to get predictions
    y_pred = df_test[features].merge(df_, how='left', on=aggr_feature)[target]
    
    return y_pred

    
def xgb_prediction(X_train, y_train, X_test, reducer):
    """
    Perform XGBoost regression for dimensionality-reduced data.

    Args:
        X_train (pd.DataFrame): Training data features.
        y_train (pd.Series): Training data labels.
        X_test (pd.DataFrame): Test data features.
        reducer: An instance of a dimensionality reduction model.
    Note:
        The function uses XGBoost to predict values in the original feature space
        for the given test data, assuming dimensionality reduction has been applied
        using the provided reducer.
    """
    
    params = {
            'tree_method': 'hist',
            'eval_metric': 'rmse',
            'max_depth': 2, 
            'learning_rate': 0.2,
            'silent': 1.0,      
            'n_estimators': 1000,
            'random_state': 42}

    xgb_regressor = xgb.XGBRegressor(**params)

    xgb_regressor.fit(X_train,y_train) 
    y_pred = xgb_regressor.predict(X_test)
    
    Y_pred = pd.DataFrame(reducer.inverse_transform(y_pred), columns=target, index=X_test.index)
    Y_pred.index.name = 'id' # for submission

    return Y_pred

def cross_validation_xgb(df, prediction=xgb_prediction, n_folds=6, n_components = 36, print_folds=True, t_score=False):
    """
    Perform cross-validation using XGBoost with dimensionality reduction.

    Args:
    - df (pd.DataFrame): The input DataFrame containing features and target variable.
    - prediction (function): The prediction function (default is xgb_prediction).
    - n_folds (int): Number of folds for cross-validation (default is 6).
    - n_components (int): Number of components for dimensionality reduction (default is 36).
    - print_folds (bool): Flag to print fold-wise results (default is True).

    Returns:
    - float: Mean metric value across all folds.
    """
    # Set up KFold for cross-validation    
    kfold = KFold(n_splits=n_folds,random_state=42, shuffle=True)
    
    # Dimensionality reduction using TruncatedSVD
    reducer = TruncatedSVD(n_components=n_components, n_iter=7, random_state=42) #20 and 7 are the best on CV
    
    # Target encoding using TargetEncoder
    encoder = ce.TargetEncoder(smoothing=8) # 8 and 12 
    
    # Extract target variables (genes) from the DataFrame
    if t_score:
        X = de_to_t_score(df[target]) 
    else:
        X = df[target] 
    
    # Fit reducer on the target variables
    reducer.fit(X)
    
    # Obtain reduced data using dimensionality reduction and target encoding
    X_zip, y_zip = obtain_reduced_data(df, reducer, encoder, t_score=t_score)

    # Initialize lists for storing metric values and a counter for fold number
    metric_av = [] 
    counter = 0 
    
    # Loop through each fold in KFold
    for train, test in kfold.split(X_zip):
        # Split data into training and testing sets
        X_train_fold = X_zip.iloc[train] 
        y_train_fold = y_zip.iloc[train]
        
        X_test_fold = X_zip.iloc[test]
        y_test_fold = df[target].iloc[test].copy()
        
        # Make predictions using the specified prediction function
        y_pred = prediction(X_train_fold, y_train_fold, X_test_fold, reducer)
        
        if t_score:
            y_pred = t_score_to_de(y_pred)
            
        # Calculate and store the metric value for the fold
        metric_av.append(metric(y_test_fold,y_pred))
        
        # Print fold-wise results if specified
        if print_folds:
            print(f"Fold #{counter} MRRMSE:", round(metric(y_test_fold,y_pred),4))
            counter +=1
    # Return the mean of the metric values across all folds
    return round(np.mean(metric_av),4)

def validate_cells(df, prediction=xgb_prediction, n_components = 36, print_folds=True):
    """
    Perform cross-validation of a model using for training dataset without one cell type from a list
    ["NK cells", "T cells CD4+", "T cells CD8+", "T regulatory cells"]

    Args:
    - df (pd.DataFrame): The input DataFrame containing features and target variable.
    - prediction (function): The prediction function (default is xgb_prediction).
    - n_components (int): Number of components for dimensionality reduction (default is 36).
    - print_folds (bool): Flag to print fold-wise results (default is True).

    Returns:
    - float: Mean metric value across all folds.
    """
    # Set up KFold for cross-validation    
    kfold = KFold(n_splits=5,random_state=42, shuffle=True)
    
    # Dimensionality reduction using TruncatedSVD
    reducer = TruncatedSVD(n_components=n_components, n_iter=7, random_state=42) #20 and 7 are the best on CV
    
    # Target encoding using TargetEncoder
    encoder = ce.TargetEncoder(smoothing=8) # 8 and 12 
    
    # Extract target variables (genes) from the DataFrame
    X = df[target] 
    
    cell_types = ["NK cells", "T cells CD4+", "T cells CD8+", "T regulatory cells"]
    
    cell_exclude_indexes = [ df[df["cell_type"] == c].index.to_list() for c in cell_types ]
    
    # Fit reducer on the target variables
    reducer.fit(X)
    
    # Obtain reduced data using dimensionality reduction and target encoding
    X_zip, y_zip = obtain_reduced_data(df, reducer, encoder)

    # Initialize lists for storing metric values and a counter for fold number
    metric_av = [] 
    
    counter = 0
    
    for c in cell_exclude_indexes:
        
        metric_cell_av = []
        X_exclude_cell = X_zip.drop(c)
        
        for train, test in kfold.split(X_exclude_cell):
            # Split data into training and testing sets
            X_train_fold = X_zip.iloc[train] 
            y_train_fold = y_zip.iloc[train]

            X_test_fold = X_zip.iloc[test]
            y_test_fold = df[target].iloc[test].copy()

            # Make predictions using the specified prediction function
            y_pred = prediction(X_train_fold, y_train_fold, X_test_fold, reducer)

            # Calculate and store the metric value for the fold
            metric_cell_av.append(metric(y_test_fold,y_pred))
        
        metric_av.append(np.mean(metric_cell_av))

        # Print fold-wise results if specified
            
        if print_folds:
            print(f"{cell_types[counter]} excluded MRRMSE:", round(metric_av[counter],4))
            counter +=1
            
    # Return the mean of the metric values across all cells
    return round(np.mean(metric_av),4)

# --------------------------------------------------------

cross_val = False
val_cells = False 

start = time.process_time()

if cross_val:
    print("CV MRRMSE:", 
      cross_validation_xgb(df,  print_folds=False, t_score=t_score))

if val_cells:
    print("Average MRRMSE:", 
      validate_cells(df,  print_folds=True))
    

Y_pred = xgb_prediction(X_zip, y_zip, X_test, reducer) 

if t_score:
    Y_pred = t_score_to_de(Y_pred)
    
end = time.process_time()

print("Modeling complete.")
print("CPU Execution time:", round(end - start,1), "seconds")

# --------------------- Data Export  ---------------------

fn_3 = 'submission.csv'
Y_pred.to_csv(fn_3)

print(f"The prediction of the model is in {fn_3}.")