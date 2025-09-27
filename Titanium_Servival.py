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
# Using the GradientBoostingClassifier model to predict the survival of Titanic passengers

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

# 3.MODEL_FILE is not created (mean model not train and not store in model.pkl)
if not os.path.exists(MODEL_FILE):
    
    # Let Train Model
    data=pd.read_csv("train.csv")
    
    # Seperate the Servived
    data_survived=data["Survived"].copy()
    data_train=data.drop("Survived",axis=1).copy()

    # seperated the numerical and categorical features
    data_num=data_train.select_dtypes(include=[np.number])
    data_cat=data_train.select_dtypes(exclude=[np.number])

    # call Method built_pipeline
    pipeline =built_pipeline(data_num, data_cat) 
    data_prepared = pipeline.fit_transform(data_train)

    #Define model
    model=GradientBoostingClassifier()
    
    #fit model
    model.fit(data_prepared,data_survived)
    
    #store the model in model file
    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)
    print("Model is trained. Congrats!")
else:
    # Lets do inference
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)
    
    test_data = pd.read_csv('test.csv')
    transformed_input = pipeline.transform(test_data)
    predictions = model.predict(transformed_input)
    test_data['Survived'] = predictions

    test_data.to_csv("output_Allinfo.csv", index=False)
    output = test_data[['PassengerId', 'Survived']]
    output.to_csv('output.csv', index=False)
    print("Inference is complete, results saved to output.csv Enjoy!")







