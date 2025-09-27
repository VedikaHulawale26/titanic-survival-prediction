import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import  root_mean_squared_error
import os
import joblib

# 1.store mode and pipeline train in this file
MODEL_FILE="model.pkl"
PIPELINE_FILE='pipeline.pkl'

# 2.Built the Pipline
def built_pipeline(data_num,data_cat):
    #created number pipeline
    num_pipeline=Pipeline([
        ("imputer",SimpleImputer(strategy="median")),
        ("scale",StandardScaler())
    ])

    #created categorical pipeline
    cat_pipeline=Pipeline([
    ('imputer',SimpleImputer(strategy='most_frequent')),
    ('onehot',OneHotEncoder(handle_unknown='ignore'))])

    #construct full pipeline
    full_pipeline=ColumnTransformer([
        ("num",num_pipeline,data_num.columns),
        ("cat",cat_pipeline,data_cat.columns)
    ])

    return full_pipeline
