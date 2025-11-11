import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
import joblib
import os

# --- Configuration ---
DATA_FILE = 'telco_customer_churn.csv'
REPORT_DIR = 'report_assets'
CLEANED_DATA_FILE = f'{REPORT_DIR}/cleaned_telco_customer_churn.csv'

# Create directory for report assets
os.makedirs(REPORT_DIR, exist_ok=True)

# --- 1. Data Preparation and Introduction (Task 1) ---

print("--- Task 1: Data Preparation and Introduction ---")

# Load the dataset
df = pd.read_csv(DATA_FILE)

# Initial description
print(f"Initial Dataset Shape: {df.shape}")
print("\nInitial Data Types:")
print(df.dtypes)

# Handle 'TotalCharges' missing values and type conversion
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
# Missing values in TotalCharges are likely customers with 0 tenure, so fill with 0
df['TotalCharges'].fillna(0, inplace=True)

# Drop 'customerID' as it's an identifier and not useful for modeling
df.drop('customerID', axis=1, inplace=True)

# Convert 'SeniorCitizen' from int to object for categorical treatment
df['SeniorCitizen'] = df['SeniorCitizen'].astype(object)

# Check for duplicates
print(f"\nNumber of duplicate rows: {df.duplicated().sum()}")

# Target variable encoding
df['Churn'] = df['Churn'].replace({'Yes': 1, 'No': 0})

# Define predictor and target variables
target = 'Churn'
categorical_cols = df.select_dtypes(include='object').columns.tolist()
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_cols.remove('Churn') # Remove target from numerical list

print("\nDefined Variables:")
print(f"Target Variable: {target}")
print(f"Categorical Predictors: {categorical_cols}")
print(f"Numerical Predictors: {numerical_cols}")

# Save cleaned dataset
df.to_csv(CLEANED_DATA_FILE, index=False)
print(f"\nCleaned dataset saved to {CLEANED_DATA_FILE}")

# --- Exploratory Data Analysis (EDA) with Visualizations ---

# 1. Churn Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x=target, data=df)
plt.title('Churn Distribution (0: No, 1: Yes)')
plt.savefig(f'{REPORT_DIR}/churn_distribution.png')
plt.close()

# 2. Tenure vs Churn
plt.figure(figsize=(8, 6))
sns.histplot(data=df, x='tenure', hue=target, multiple='stack', bins=30)
plt.title('Tenure Distribution by Churn')
plt.savefig(f'{REPORT_DIR}/tenure_churn_hist.png')
plt.close()

# 3. Monthly Charges vs Churn
plt.figure(figsize=(8, 6))
sns.histplot(data=df, x='MonthlyCharges', hue=target, multiple='stack', bins=30)
plt.title('Monthly Charges Distribution by Churn')
plt.savefig(f'{REPORT_DIR}/monthly_charges_churn_hist.png')
plt.close()

# 4. Churn by Contract Type
plt.figure(figsize=(10, 6))
sns.countplot(x='Contract', hue=target, data=df)
plt.title('Churn by Contract Type')
plt.savefig(f'{REPORT_DIR}/contract_churn_bar.png')
plt.close()

print("EDA visualizations saved to report_assets directory.")

# --- Data Preprocessing for Modeling ---

# One-hot encode categorical variables for both models
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Separate features and target
X = df_encoded.drop(target, axis=1)
y = df_encoded[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Scale numerical features for Logistic Regression (and Decision Tree, though less critical)
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])

# --- 2. Model Development and Rule Induction using Decision Tree (CHAID Proxy) (Task 2) ---

print("\n--- Task 2: Decision Tree Model Development and Rule Induction (CHAID Proxy) ---")

# Decision Tree is a suitable proxy for CHAID as both are tree-based classification algorithms
# and can be used for rule induction and interpretation.
model_dt = DecisionTreeClassifier(max_depth=4, random_state=42)
model_dt.fit(X_train, y_train) # Using unscaled data for easier rule interpretation

# Rule Interpretation: We will export the tree structure for visualization and interpretation
from sklearn.tree import export_graphviz
import graphviz

# Export as dot file
dot_data = export_graphviz(
    model_dt,
    out_file=None,
    feature_names=X_train.columns,
    class_names=['No Churn', 'Churn'],
    filled=True,
    rounded=True,
    special_characters=True
)

# Convert to graphviz object and save as PNG
graph = graphviz.Source(dot_data)
graph.render(f'{REPORT_DIR}/decision_tree', format='png', cleanup=True)
print("Decision Tree visualization saved to report_assets/decision_tree.png")

# --- 3. Model Comparison and Evaluation (Task 3) ---

print("\n--- Task 3: Model Comparison and Evaluation ---")

# --- Logistic Regression Model ---
model_lr = LogisticRegression(random_state=42, solver='liblinear')
model_lr.fit(X_train_scaled, y_train)
y_pred_lr = model_lr.predict(X_test_scaled)
y_proba_lr = model_lr.predict_proba(X_test_scaled)[:, 1]

# --- Decision Tree Model Prediction ---
y_pred_dt = model_dt.predict(X_test)
y_proba_dt = model_dt.predict_proba(X_test)[:, 1]

# --- Evaluation Metrics ---

# Function to calculate and print metrics
def evaluate_model(model_name, y_test, y_pred, y_proba):
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    print(f"\n--- {model_name} Evaluation ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.savefig(f'{REPORT_DIR}/{model_name.lower().replace(" ", "_")}_roc_curve.png')
    plt.close()
    
    return accuracy, roc_auc

# Evaluate Logistic Regression
acc_lr, auc_lr = evaluate_model("Logistic Regression", y_test, y_pred_lr, y_proba_lr)

# Evaluate Decision Tree
acc_dt, auc_dt = evaluate_model("Decision Tree", y_test, y_pred_dt, y_proba_dt)

# --- Comparison Summary ---
comparison_data = {
    'Model': ['Logistic Regression', 'Decision Tree (CHAID Proxy)'],
    'Accuracy': [acc_lr, acc_dt],
    'ROC-AUC': [auc_lr, auc_dt]
}
comparison_df = pd.DataFrame(comparison_data)
print("\n--- Model Comparison Summary ---")
print(comparison_df)

# Save model comparison data
comparison_df.to_csv(f'{REPORT_DIR}/model_comparison_summary.csv', index=False)

# --- 4. Model Deployment and Updating (Task 4) ---

print("\n--- Task 4: Model Deployment and Updating (Conceptual) ---")

# Save the Logistic Regression model and scaler using joblib (as an example for deployment)
joblib.dump(model_lr, f'{REPORT_DIR}/logistic_regression_model.joblib')
joblib.dump(scaler, f'{REPORT_DIR}/scaler.joblib')
print(f"Logistic Regression model and scaler saved for deployment to {REPORT_DIR}/")

print("\nScript execution complete. Assets saved to report_assets directory.")
