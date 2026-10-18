# E-Commerce Return Prediction Using Machine Learning

## 1. Project Overview

E-Commerce Return Prediction is a machine learning project that predicts whether an e-commerce product is likely to be returned based on customer information and purchase details.

The project focuses on implementing, training, comparing, and evaluating machine learning models to identify a suitable model for return prediction. A trained model is then integrated into a Python Tkinter GUI so users can enter purchase details and view the predicted return probability.

The goal is to demonstrate the complete machine learning workflow, from data preprocessing and model development to model evaluation and application integration.

## 2. Objectives

- Analyze e-commerce purchase data for return prediction.
- Preprocess numerical and categorical features.
- Implement and train multiple machine learning models.
- Compare models using suitable evaluation metrics.
- Identify the best-performing model based on experimental results.
- Save the trained model for reuse.
- Develop a Tkinter GUI for making predictions.
- Display prediction probabilities and model comparison results.

## 3. Project Workflow

1. Dataset collection and exploration.
2. Data cleaning and preprocessing.
3. Feature engineering and preparation.
4. Training and testing multiple machine learning models.
5. Model evaluation using suitable metrics.
6. Comparison of model performance.
7. Selection of the best-performing model.
8. Saving the trained model and GUI input schema.
9. Integrating the model with the Tkinter application.
10. Testing predictions using different purchase inputs.

## 4. Machine Learning Models

The project focuses on comparing multiple machine learning models for predicting product returns.

The final list of models and their evaluation results should be taken from the actual model-training notebook.

The supplied `gui_schema.json` identifies **Random Forest (Tuned)** as the best model.

### Model Comparison

The models can be compared using appropriate classification metrics, such as:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- ROC-AUC, if calculated in the training notebook

These metrics help evaluate the models from different perspectives. The final model should be selected based on the actual evaluation results and project requirements.

**Note:** Actual model names, scores, and comparison results should be added from the completed training notebook. No performance values are assumed in this README.

## 5. Technologies Used

- **Python:** Main programming language.
- **Pandas:** Data handling and preprocessing.
- **Scikit-learn:** Machine learning models, preprocessing pipelines, and evaluation.
- **Joblib:** Saving and loading the trained model.
- **Tkinter and ttk:** Desktop graphical user interface.
- **JSON:** Storing the GUI schema and feature information.
- **Matplotlib / Seaborn:** Data visualization, if used in the training notebook.
- **Jupyter Notebook / Google Colab:** Model development and experimentation.
- **Git and GitHub:** Version control and collaboration.

## 6. Input Features

The prediction application uses the following 12 features:

1. Product Category
2. Product Price
3. Quantity
4. Total Purchase Amount
5. Payment Method
6. Customer Age
7. Gender
8. Purchase Month
9. Purchase Day of Week
10. Purchase Hour
11. Is Weekend
12. Price per Unit

Some features are calculated automatically by the application:

- **Total Purchase Amount:** Product Price × Quantity
- **Is Weekend:** 1 for Saturday or Sunday; otherwise 0
- **Price per Unit:** Product Price

The categorical input options and expected feature order are obtained from `gui_schema.json`.

## 7. Model Saving and Integration

The trained machine learning pipeline is saved as:

`return_model.joblib`

The GUI configuration is stored in:

`gui_schema.json`

The application loads the saved model using Joblib and prepares the input data in the feature order expected by the model.

The model pipeline may include preprocessing steps along with the trained estimator. The GUI uses the pipeline to generate the return prediction.

## 8. Graphical User Interface

The Tkinter application provides:

- Dropdowns for categorical inputs.
- Input fields for numerical values.
- Purchase date and time-related selections.
- Input validation and error messages.
- A Predict button.
- A Clear button.
- An Exit button.
- Return probability displayed as a percentage.
- A result message indicating whether the product is likely to be returned.
- A Model Comparison window when evaluation results are available.

The prediction threshold used by the interface is 50%:

- Probability greater than or equal to 50%: Likely to be returned.
- Probability below 50%: Unlikely to be returned.

This threshold is a project decision and does not guarantee the actual outcome.

## 9. Project Structure

```text
ecommerce-return-prediction-ml/
├── app.py
├── gui_schema.json
├── return_model.joblib
├── model_comparison.csv
├── requirements.txt
├── README.md
└── .gitignore
```

The model comparison CSV is optional and should contain the actual evaluation results exported from the training notebook.

The model-training notebook and dataset may also be included in the repository if permitted by the dataset licence and the team's repository policy.

## 10. Installation and Setup

### Prerequisites

- Python installed on the system.
- The trained model and GUI schema files.
- A compatible Python and scikit-learn environment.
- Git, if cloning the repository.

### Step 1: Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/ecommerce-return-prediction-ml.git
cd ecommerce-return-prediction-ml
```

Replace `YOUR-USERNAME` with your GitHub username.

### Step 2: Create a virtual environment

On Windows:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

Python 3.11 is a suggested starting point for compatibility testing, not a confirmed requirement of every model dependency.

### Step 3: Install dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 4: Verify the required files

Ensure that these files are present in the project folder:

- `app.py`
- `gui_schema.json`
- `return_model.joblib`

### Step 5: Run the GUI

```bash
python app.py
```

## 11. Requirements

The main Python dependencies are:

```text
pandas
scikit-learn==1.5.1
joblib
```

The scikit-learn version should match the environment used to save the model. Update the pinned version only after verifying compatibility with the training environment.

Tkinter is generally included with standard Windows Python installations.

## 12. Model Evaluation Results

The final model evaluation table should be added here using the actual results from the training notebook.

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Model 1 | Add result | Add result | Add result | Add result |
| Model 2 | Add result | Add result | Add result | Add result |
| Random Forest (Tuned) | Add result | Add result | Add result | Add result |

Replace the placeholders with real values. Include only the models and metrics actually evaluated in the project.

## 13. Testing

The application should be tested with different valid inputs and invalid values.

- Test different product categories.
- Change the product price and quantity.
- Select different payment methods and genders.
- Test different customer ages and purchase times.
- Verify automatic feature calculations.
- Check that invalid inputs produce appropriate error messages.
- Verify that predictions are displayed correctly.
- Verify the Clear and Exit buttons.
- Check that model comparison results load when the CSV is available.

Document actual test results and screenshots in the project report.

## 14. Limitations

- Predictions depend on the quality and representativeness of the training dataset.
- The predicted probability is not a guarantee of an actual product return.
- The model file requires a compatible software environment.
- The GUI depends on the supplied trained model and schema.
- Model comparison results are available only when the evaluation data is supplied.

## 15. Future Scope

- Improve model performance through hyperparameter tuning.
- Add additional machine learning models for comparison.
- Add feature importance analysis.
- Visualize evaluation metrics and confusion matrices.
- Store prediction history.
- Develop a web-based application or API.
- Monitor model performance as new data becomes available.

## 16. Conclusion

This project demonstrates the application of machine learning to e-commerce product return prediction. It emphasizes model development, performance evaluation, comparison of alternative models, selection of a suitable trained model, and integration with a graphical user interface.

The project provides a foundation for understanding how machine learning models can be applied to real-world e-commerce problems.

