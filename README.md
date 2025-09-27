# Titanic Survival Prediction

This repository contains a Python script that builds a machine learning model to predict whether a passenger on the Titanic survived or not. The project demonstrates a complete end-to-end workflow, from data preprocessing and model training to inference on new data.

## Features

- **Automated Data Preprocessing**: Utilizes a `scikit-learn` `Pipeline` and `ColumnTransformer` to create a robust and reusable preprocessing workflow.
- **Handles Mixed Data Types**: Automatically separates numerical and categorical features and applies appropriate transformations:
    - **Numerical**: Missing values are imputed with the median, followed by feature scaling (`StandardScaler`).
    - **Categorical**: Missing values are imputed with the most frequent value, followed by one-hot encoding.
- **Model Training**: Employs a `GradientBoostingClassifier`, a powerful ensemble method, to make predictions.
- **Model Persistence**: The trained model and the entire preprocessing pipeline are saved using `joblib`, allowing for quick and repeatable inference without retraining.
- **Dual-Mode Execution**: The script is designed to:
    1.  **Train**: If no model exists, it trains one using `train.csv`.
    2.  **Infer**: If a model is found, it automatically loads it and generates predictions for `test.csv`.

## Dataset

This project uses the classic "Titanic - Machine Learning from Disaster" dataset from Kaggle. You will need to download the `train.csv` and `test.csv` files and place them in the root of this repository.

[Link to Dataset on Kaggle](https://www.kaggle.com/c/titanic)

## Technologies Used

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Joblib
