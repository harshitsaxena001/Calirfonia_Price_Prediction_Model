# Calirfonia_Price_Prediction_Model



-----

# California Housing Price Prediction 🏡

This project builds and uses a machine learning model to predict median house values in California districts based on various features. The workflow is automated: it first trains a **Random Forest Regressor** model if one doesn't exist, and then uses the trained model for inference on new data.

## 🚀 Project Overview

The core script performs two main functions depending on the existence of a saved model file (`model.pkl`):

1.  **Training Mode**: If no pre-trained model is found, the script loads the `housing.csv` dataset, preprocesses the data, trains a `RandomForestRegressor` model, and saves both the model and the data processing pipeline.
2.  **Inference Mode**: If a model already exists, the script loads the saved model and pipeline, reads data from `input.csv`, predicts the median house values, and saves the results to `output.csv`.

The project uses a stratified split based on median income to ensure the test set is representative of the overall dataset.

-----

## 📂 File Structure

```
.
├── housing.csv         # Raw dataset for training
├── train.py            # Main script for training and inference (or your script's name)
├── model.pkl           # Saved trained model (generated after first run)
├── pipeline.pkl        # Saved data processing pipeline (generated after first run)
├── input.csv           # Test data for inference (generated after first run)
└── output.csv          # Predictions on the input data (generated after inference)
```

-----

## 🛠️ Key Components

### Data Preprocessing

A `ColumnTransformer` pipeline is used to handle both numerical and categorical features separately:

  * **Numerical Features**: Missing values are filled using the **median** (`SimpleImputer`), and then the data is scaled using `StandardScaler`.
  * **Categorical Features**: The `ocean_proximity` column is transformed into numerical format using `OneHotEncoder`.

### Model

The project uses a **`RandomForestRegressor`** from Scikit-learn, a powerful ensemble model that generally provides high accuracy.

### Workflow Automation

The script automatically detects whether to train a new model or perform inference with an existing one by checking for the `model.pkl` file. This creates a simple yet effective MLOps pipeline.

-----

## 📋 Requirements

You'll need Python 3 and the following libraries. You can install them using pip:

```bash
pip install pandas numpy scikit-learn joblib
```

-----

## ▶️ How to Run

1.  **Place your data:** Ensure the `housing.csv` file is in the same directory as the script.

2.  **Run for the first time (Training):**
    Execute the Python script. On the first run, it will:

      * Load `housing.csv`.
      * Create a stratified test set and save it as `input.csv`.
      * Train the Random Forest model on the remaining data.
      * Save the trained model as `model.pkl` and the preprocessing pipeline as `pipeline.pkl`.
      * You will see the message: `Model is trained. Congrats!`

    <!-- end list -->

    ```bash
    python your_script_name.py
    ```

    *(Replace `your_script_name.py` with the actual name of your Python file)*

3.  **Run subsequently (Inference):**
    If you run the script again, it will:

      * Load the existing `model.pkl` and `pipeline.pkl`.
      * Load the `input.csv` file.
      * Perform predictions on the data in `input.csv`.
      * Save the results (input features + predicted median house value) to `output.csv`.
      * You will see the message: `Inference is complete, results saved to output.csv Enjoy!`
