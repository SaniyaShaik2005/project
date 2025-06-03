import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Step 1: Preprocessing (Already Done - Let's Recap and Proceed)
# Your preprocessed data is in the variable `data` from the previous step
# Let's ensure it's ready for the next steps

# Define categorical and numerical columns
categorical_cols = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
numerical_cols = ['age_years', 'height', 'weight', 'ap_hi', 'ap_lo']

# Preprocessing pipeline for scaling and encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
    ])

# Prepare features and target
X = data.drop(['id', 'cardio'], axis=1)
y = data['cardio']
X_preprocessed = preprocessor.fit_transform(X)

# Get feature names after one-hot encoding
cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
feature_names = numerical_cols + list(cat_feature_names)
X_preprocessed_df = pd.DataFrame(X_preprocessed, columns=feature_names)

print("\nPreprocessed Data (first 5 rows):")
print(X_preprocessed_df.head())

# Step 2: Data Analysis and Visualization
# Summary statistics
print("\nSummary Statistics:")
print(data.describe())

# Histograms for numerical features
for col in numerical_cols:
    plt.figure(figsize=(8, 5))
    sns.histplot(data[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

# Box plots for numerical features
plt.figure(figsize=(10, 6))
data[numerical_cols].boxplot()
plt.title('Box Plot of Numerical Features')
plt.xticks(rotation=45)
plt.show()

# Bar charts for categorical features
for col in categorical_cols:
    plt.figure(figsize=(8, 5))
    sns.countplot(x=col, hue='cardio', data=data)
    plt.title(f'Frequency of {col} by Heart Disease')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.show()

# Box plots for numerical features vs. target
for col in numerical_cols:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='cardio', y=col, data=data)
    plt.title(f'{col} vs. Heart Disease')
    plt.xlabel('Heart Disease (0 = No, 1 = Yes)')
    plt.ylabel(col)
    plt.show()

# Step 3: Correlation Matrix
plt.figure(figsize=(12, 8))
correlation_matrix = X_preprocessed_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Features')
plt.show()

# Fix for Correlation with Target: Select only numeric columns
numeric_data = data.select_dtypes(include=['float64', 'int64'])
print("\nCorrelation with Target:")
print(numeric_data.corr()['cardio'].sort_values(ascending=False))

# Step 4: Machine Learning Model Training
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Define models
models = {
    'SVM': SVC(),
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier()
}

# Train and evaluate models
results = {}
for name, model in models.items():
    model.fit(X_train_preprocessed, y_train)
    y_pred = model.predict(X_test_preprocessed)
    results[name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred)
    }
    print(f"\nClassification Report for {name}:")
    print(classification_report(y_test, y_pred))

# Display results
print("\nModel Performance Summary:")
results_df = pd.DataFrame(results).T
print(results_df)

# Plot model performance
plt.figure(figsize=(10, 6))
results_df.plot(kind='bar')
plt.title('Model Performance Comparison')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.show()

# Step 5: Model Building
# Optimize Random Forest
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}
rf = RandomForestClassifier()
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='f1')
grid_search.fit(X_train_preprocessed, y_train)

# Best model
best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# Train final model on entire dataset
X_preprocessed_full = preprocessor.fit_transform(X)
best_model.fit(X_preprocessed_full, y)

# Prediction function
def shape_input_data(input_data):
    expected_columns = numerical_cols + categorical_cols
    input_df = pd.DataFrame([input_data], columns=expected_columns)
    for col in numerical_cols:
        input_df[col] = input_df[col].astype(float)
    for col in categorical_cols:
        input_df[col] = input_df[col].astype(int)
    return input_df

def predict_heart_disease(input_data):
    input_df = shape_input_data(input_data)
    input_preprocessed = preprocessor.transform(input_df)
    prediction = best_model.predict(input_preprocessed)
    probability = best_model.predict_proba(input_preprocessed)[0]
    return {
        'Prediction': 'Heart Disease' if prediction[0] == 1 else 'No Heart Disease',
        'Probability': probability
    }

# Example input
example_input = {
    'age_years': 55,
    'height': 170,
    'weight': 70,
    'ap_hi': 130,
    'ap_lo': 80,
    'cholesterol': 1,
    'gluc': 1,
    'smoke': 0,
    'alco': 0,
    'active': 1,
    'gender': 2
}
print("\nPrediction for Example Input:")
print(predict_heart_disease(example_input))

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': best_model.feature_importances_
}).sort_values(by='Importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance)

# Visualize feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance for Random Forest')
plt.show()