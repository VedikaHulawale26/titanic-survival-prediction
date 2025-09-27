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

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Install dependencies:**
    It's recommended to create a virtual environment first.
    ```bash
    pip install pandas numpy scikit-learn joblib
    ```

3.  **Place Data**: Download `train.csv` and `test.csv` from the Kaggle competition and place them in the project directory.

4.  **Execute the script:**
    *(**Note**: Remember to rename your script from `Titanium_Servival.py` to `titanic_survival.py` for clarity)*.

    -   **To train the model (first run):**
        ```bash
        python titanic_survival.py
        ```
        This will process `train.csv` and create two files: `model.pkl` and `pipeline.pkl`.

    -   **To generate predictions (subsequent runs):**
        Running the same command again will now skip training.
        ```bash
        python titanic_survival.py
        ```
        The script will load `model.pkl` and `pipeline.pkl`, make predictions on `test.csv`, and save the results to `output.csv`.

## File Structure
