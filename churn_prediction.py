import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from langchain.prompts import PromptTemplate

# Load the dataset
df = pd.read_csv("telcom_m.csv")

# Handle missing values
# Numerical columns: impute with median
numerical_cols = [
    'Age', 'Number of Dependents', 'Number of Referrals', 'Tenure in Months',
    'Avg Monthly Long Distance Charges', 'Avg Monthly GB Download', 'Monthly Charge',
    'Total Charges', 'Total Refunds', 'Total Extra Data Charges', 'Total Long Distance Charges',
    'Total Revenue', 'Satisfaction Score', 'Churn Score', 'CLTV'
]
for col in numerical_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col].fillna(df[col].median(), inplace=True)   
     # Categorical columns: fill missing with 'Unknown'
categorical_cols = [
    'Gender', 'Under 30', 'Senior Citizen', 'Married', 'Dependents', 'Country',
    'State', 'City', 'Zip Code', 'Quarter', 'Referred a Friend', 'Offer',
    'Phone Service', 'Multiple Lines', 'Internet Service', 'Internet Type',
    'Online Security', 'Online Backup', 'Device Protection Plan', 'Premium Tech Support',
    'Streaming TV', 'Streaming Movies', 'Streaming Music', 'Unlimited Data',
    'Contract', 'Paperless Billing', 'Payment Method', 'Churn Category', 'Churn Reason'
]
for col in categorical_cols:
    df[col] = df[col].astype(str).fillna('Unknown')

# Handle target variable 'Churn Label'
df['Churn Label'] = df['Churn Label'].map({'Yes': 1, 'No': 0})
df = df.dropna(subset=['Churn Label'])
if not df['Churn Label'].isin([0, 1]).all():
    raise ValueError("Non-binary values found in 'Churn Label' column")

# Encode categorical variables
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Define features and target
drop_cols = [
    'Customer ID', 'Latitude', 'Longitude', 'Population', 'Zip Code', 'Country', 'State', 'City',
    'Churn Label', 'Churn Category', 'Churn Reason', 'Customer Status'
]
X = df.drop(drop_cols, axis=1)
y = df['Churn Label'].astype(int)  

 # Ensure all features are numeric
X = X.astype(float)

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Random Forest model with class weighting
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)

# Predict and evaluate
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values(by='importance', ascending=False)

# Define dynamic PromptTemplate
prompt_template = PromptTemplate(
    input_variables=["accuracy", "precision", "recall", "f1", "top_features", "question"],
    template="""
    The telecom churn prediction model achieved the following performance metrics:
    - Accuracy: {accuracy:.2f}
    - Precision: {precision:.2f}
    - Recall: {recall:.2f}
    - F1-Score: {f1:.2f}

    The top features contributing to customer churn are:
    {top_features}

    {question}
    """
)  

 # Format top features for the prompt
top_features_str = "\n".join([f"- {row['feature']}: {row['importance']:.4f}" for _, row in feature_importance.head(3).iterrows()])

# Input question
input_question = """
Based on these results, provide a concise summary of the model's performance and suggest actionable insights to reduce customer churn. For example, focus on customers with high values in the top features or improve services related to these features.
"""

# Generate prompt output
prompt_output = prompt_template.format(
    accuracy=accuracy,
    precision=precision,
    recall=recall,
    f1=f1,
    top_features=top_features_str,
    question=input_question
)

# Print results
print("\nModel Performance:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
print("\nTop Features:")
print(feature_importance.head(3))
print("\nPrompt for Insights:")
print(prompt_output)