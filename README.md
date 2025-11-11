# Customer Churn Prediction: Model Development, Validation, and Deployment

## Project Overview

This repository contains the solution for the Case Study Assignment for the course **Inferential Statistics and Predictive Analytics (21AIC401T)**.

The objective of this project was to:
1.  Develop a predictive model to identify customers likely to churn in a telecommunications company.
2.  Apply statistical inference and predictive modeling concepts, including model validation, comparison, evaluation, and deployment design.
3.  Use a real-world dataset (Telco Customer Churn).

## Assignment Tasks Completed

The project addresses all four main tasks of the assignment:

1.  **Data Preparation and Introduction:** Download, cleaning, and Exploratory Data Analysis (EDA) with visualizations.
2.  **Model Development and Rule Induction:** Implementation of a Decision Tree (as a proxy for CHAID) for rule extraction and a Logistic Regression model.
3.  **Model Comparison and Evaluation:** Comparison of models using Accuracy, ROC-AUC, and discussion of Lift/Gains Charts.
4.  **Model Deployment and Updating:** Explanation of the deployment process (using `joblib` serialization) and a strategy for model updating and meta-level automation.

## Repository Contents

| File/Folder | Description |
| :--- | :--- |
| `churn_prediction_script.py` | The main Python script containing all code for data processing, EDA, model training, and evaluation. |
| `report_assets/` | Directory containing all generated assets: |
| `report_assets/cleaned_telco_customer_churn.csv` | The cleaned dataset used for modeling. |
| `report_assets/logistic_regression_model.joblib` | Serialized Logistic Regression model for deployment. |
| `report_assets/scaler.joblib` | Serialized feature scaler for deployment. |
| `report_assets/*.png` | All generated charts (EDA, ROC Curves) and the Decision Tree visualization. |
| `telco_customer_churn.csv` | The raw downloaded dataset. |

## Setup and Execution

### Prerequisites

*   Python 3.8+
*   `pip` (Python package installer)

### Installation

It is highly recommended to use a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install pandas numpy scikit-learn matplotlib seaborn graphviz
```

### Running the Script

The main analysis can be executed with the following command:

```bash
python churn_prediction_script.py
```

This script will:
1.  Load and clean the `telco_customer_churn.csv` file.
2.  Perform EDA and save plots to the `report_assets/` folder.
3.  Train the Logistic Regression and Decision Tree models.
4.  Evaluate the models and save comparison metrics.
5.  Save the trained model and scaler for deployment.

## Key Findings

*   **Best Predictive Model:** Logistic Regression (ROC-AUC: 0.8449).
*   **Key Churn Drivers (from Decision Tree):** Contract Type (Month-to-month), Internet Service (Fiber optic), and low Tenure.
*   **Deployment:** The Logistic Regression model and its scaler are saved using `joblib` for easy integration into a production API.
