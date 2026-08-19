# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 12:38:18 2026

@author: Muhammed
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

#DATA LOADING

file_path = r"C:\Users\Muhammed\Desktop\Makine Öğrenmesi\Titanic-Dataset.csv"
df = pd.read_csv(file_path)

#(EDA)

print(df.head())
print(df.info())

#Survival Rate by Gender
plt.figure(figsize=(6,4))
sns.countplot(x='Survived', hue='Sex', data=df, palette='Set2')
plt.title('Survival Count by Gender')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Passenger Count')
plt.show()

#Survival Count by Passenger Class
plt.figure(figsize=(6,4))
sns.countplot(x='Survived', hue='Pclass', data=df, palette='Set1')
plt.title('Survival Count by Passenger Class')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Passenger Count')
plt.show()

# DATA CLEANING & IMPUTATION

# Handling Missing Values for 'Embarked' 
# Since only 2 rows are missing, we fill them with the most frequent value.
most_frequent_port = df['Embarked'].mode()[0]
df['Embarked'] = df['Embarked'].fillna(most_frequent_port)

# Handling Missing Values for 'Age' 
# Instead of a flat average, we fill missing ages with the median age of the dataset.
median_age = df['Age'].median()
df['Age'] = df['Age'].fillna(median_age)

# Dropping 'Cabin' column due to high missing percentage (~77%)
# It contains too much noise and too little data to be useful.
df = df.drop(columns=['Cabin'])

# Verify that missing values are handled
print("\n--- Missing Values Check After Cleaning ---")
print((df.isnull().sum()))

# --- CATEGORICAL ENCODING & FEATURE ENGINEERING ---

# Convert 'Sex' into binary numerical values (0 and 1)
# Female will be 1, Male will be 0
df['Sex'] = df['Sex'].map({'female': 1, 'male': 0})

# Convert 'Embarked' into dummy variables using One-Hot Encoding
# This avoids giving an artificial numerical order (like 0, 1, 2) to ports
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True, dtype=int)

# Drop columns that are text-heavy and won't contribute to the model directly
# PassengerId, Name, and Ticket have too high cardinality (too many unique values)
df = df.drop(columns=['PassengerId', 'Name', 'Ticket'])

# Display the processed dataset structure
print("\n--- Final Cleaned and Encoded Dataset Preview ---")
print(df.head())
print("\n--- Final Column Names ---")
print(df.columns.tolist())

from sklearn.model_selection import train_test_split

# --- DATA SPLITTING (TRAIN-TEST SPLIT) ---

# Separate Features (X) and Target Variable (y)
# X contains all the input features, y contains the survival output we want to predict

X = df.drop(columns=['Survived'])
y=df['Survived']

# Split the dataset into training set (80%) and testing set (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Print the shapes of the sub-datasets to verify the split
print("\n--- Data Split Statistics ---")
print(f"Total Features Shape: {X.shape}")
print(f"Training Features Shape (X_train): {X_train.shape}")
print(f"Testing Features Shape (X_test): {X_test.shape}")
print(f"Training Target Shape (y_train): {y_train.shape}")
print(f"Testing Target Shape (y_test): {y_test.shape}")

#--- MODELS ---

# =============================================================================
# --- LOGISTIC REGRESSION MODELING & EVALUATION ---
# =============================================================================
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# max_iter=1000 ensures the optimization algorithm has enough iterations to converge
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)

y_pred_log = log_reg.predict(X_test)
y_prob_log = log_reg.predict_proba(X_test)[:, 1]

# Calculate Performance Metrics
accuracy_log = accuracy_score(y_test, y_pred_log)
cm_log = confusion_matrix(y_test, y_pred_log)

print("\n=============================================")
print("         LOGISTIC REGRESSION RESULTS         ")
print("=============================================")
print(f"Test Accuracy: {accuracy_log:.2%}")
print("\nConfusion Matrix (Raw Counts):")
print(cm_log)
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred_log))
print("=============================================")

# 5. Plot 1: Confusion Matrix Visual
plt.figure(figsize=(5, 4))
sns.heatmap(cm_log, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted Dead (0)', 'Predicted Survived (1)'],
            yticklabels=['Actual Dead (0)', 'Actual Survived (1)'])
plt.title('Logistic Regression - Confusion Matrix')
plt.ylabel('Actual Labels')
plt.xlabel('Predicted Labels')
plt.show()

# 6. Plot 2: Model Fit & Probability Distribution
prob_df = pd.DataFrame({
    'Actual': y_test,
    'Probability': y_prob_log
})

plt.figure(figsize=(8, 5))
sns.histplot(data=prob_df[prob_df['Actual'] == 0], x='Probability', color='red', label='Actual Dead (0)', kde=True, alpha=0.5, bins=20)
sns.histplot(data=prob_df[prob_df['Actual'] == 1], x='Probability', color='green', label='Actual Survived (1)', kde=True, alpha=0.5, bins=20)

plt.axvline(x=0.5, color='black', linestyle='--', label='Decision Threshold (0.5)')
plt.title('Logistic Regression - Prediction Probability Distribution')
plt.xlabel('Predicted Probability of Survival')
plt.ylabel('Count of Passengers')
plt.legend()
plt.show()


# =============================================================================
# --- K-NEAREST NEIGHBORS (KNN) WITH PIPELINE & SCALING ---
# =============================================================================
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

knn_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=5))
])

knn_pipeline.fit(X_train, y_train)
y_pred_knn = knn_pipeline.predict(X_test)
y_prob_knn = knn_pipeline.predict_proba(X_test)[:, 1]

# Calculate Performance Metrics
accuracy_knn = accuracy_score(y_test, y_pred_knn)
cm_knn = confusion_matrix(y_test, y_pred_knn)

print("\n=============================================")
print("     KNN WITH PIPELINE (SCALED) RESULTS      ")
print("=============================================")
print(f"Test Accuracy: {accuracy_knn:.2%}")
print("\nConfusion Matrix (Raw Counts):")
print(cm_knn)
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred_knn))
print("=============================================")

# Plot 1: Confusion Matrix Visual
plt.figure(figsize=(5, 4))
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Oranges', 
            xticklabels=['Predicted Dead (0)', 'Predicted Survived (1)'],
            yticklabels=['Actual Dead (0)', 'Actual Survived (1)'])
plt.title('KNN (Scaled) - Confusion Matrix')
plt.ylabel('Actual Labels')
plt.xlabel('Predicted Labels')
plt.show()

# Create a DataFrame for KNN probability visualization
prob_df_knn = pd.DataFrame({
    'Actual': y_test,
    'Probability': y_prob_knn
})

plt.figure(figsize=(8, 5))
# Plot distribution for those who actually died
sns.histplot(data=prob_df_knn[prob_df_knn['Actual'] == 0], x='Probability', color='red', label='Actual Dead (0)', alpha=0.5, bins=6)
# Plot distribution for those who actually survived
sns.histplot(data=prob_df_knn[prob_df_knn['Actual'] == 1], x='Probability', color='green', label='Actual Survived (1)', alpha=0.5, bins=6)

plt.axvline(x=0.5, color='black', linestyle='--', label='Decision Threshold (0.5)')
plt.title('KNN (Scaled) - Prediction Probability Distribution')
plt.xlabel('Predicted Probability of Survival (Based on 5 Neighbors)')
plt.ylabel('Count of Passengers')
plt.legend()
plt.show()

# =============================================================================
# --- SUPPORT VECTOR MACHINE (SVM) WITH PIPELINE & SCALING ---
# =============================================================================
from sklearn.svm import SVC

svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(probability=True, random_state=42))
])


svm_pipeline.fit(X_train, y_train)

y_pred_svm = svm_pipeline.predict(X_test)
y_prob_svm = svm_pipeline.predict_proba(X_test)[:, 1]

# Calculate Performance Metrics
accuracy_svm = accuracy_score(y_test, y_pred_svm)
cm_svm = confusion_matrix(y_test, y_pred_svm)

print("\n=============================================")
print("     SVM WITH PIPELINE (SCALED) RESULTS      ")
print("=============================================")
print(f"Test Accuracy: {accuracy_svm:.2%}")
print("\nConfusion Matrix (Raw Counts):")
print(cm_svm)
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred_svm))
print("=============================================")

# 6. Plot 1: Confusion Matrix Visual
plt.figure(figsize=(5, 4))
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Purples', 
            xticklabels=['Predicted Dead (0)', 'Predicted Survived (1)'],
            yticklabels=['Actual Dead (0)', 'Actual Survived (1)'])
plt.title('SVM (Scaled) - Confusion Matrix')
plt.ylabel('Actual Labels')
plt.xlabel('Predicted Labels')
plt.show()

# 7. Plot 2: Model Fit & Probability Distribution
prob_df_svm = pd.DataFrame({
    'Actual': y_test,
    'Probability': y_prob_svm
})

plt.figure(figsize=(8, 5))
sns.histplot(data=prob_df_svm[prob_df_svm['Actual'] == 0], x='Probability', color='red', label='Actual Dead (0)', kde=True, alpha=0.5, bins=20)
sns.histplot(data=prob_df_svm[prob_df_svm['Actual'] == 1], x='Probability', color='green', label='Actual Survived (1)', kde=True, alpha=0.5, bins=20)
plt.axvline(x=0.5, color='black', linestyle='--', label='Decision Threshold (0.5)')
plt.title('SVM (Scaled) - Prediction Probability Distribution')
plt.xlabel('Predicted Probability of Survival')
plt.ylabel('Count of Passengers')
plt.legend()
plt.show()

# =============================================================================
# --- GAUSSIAN NAIVE BAYES MODELING & EVALUATION ---
# =============================================================================
from sklearn.naive_bayes import GaussianNB

nb_clf = GaussianNB()
nb_clf.fit(X_train, y_train)
y_pred_nb = nb_clf.predict(X_test)
y_prob_nb = nb_clf.predict_proba(X_test)[:, 1]

# Calculate Performance Metrics
accuracy_nb = accuracy_score(y_test, y_pred_nb)
cm_nb = confusion_matrix(y_test, y_pred_nb)

print("\n=============================================")
print("          GAUSSIAN NAIVE BAYES RESULTS        ")
print("=============================================")
print(f"Test Accuracy: {accuracy_nb:.2%}")
print("\nConfusion Matrix (Raw Counts):")
print(cm_nb)
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred_nb))
print("=============================================")

# 5. Plot 1: Confusion Matrix Visual
plt.figure(figsize=(5, 4))
sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['Predicted Dead (0)', 'Predicted Survived (1)'],
            yticklabels=['Actual Dead (0)', 'Actual Survived (1)'])
plt.title('Naive Bayes - Confusion Matrix')
plt.ylabel('Actual Labels')
plt.xlabel('Predicted Labels')
plt.show()

# 6. Plot 2: Model Fit & Probability Distribution
prob_df_nb = pd.DataFrame({
    'Actual': y_test,
    'Probability': y_prob_nb
})

plt.figure(figsize=(8, 5))
sns.histplot(data=prob_df_nb[prob_df_nb['Actual'] == 0], x='Probability', color='red', label='Actual Dead (0)', kde=True, alpha=0.5, bins=20)
sns.histplot(data=prob_df_nb[prob_df_nb['Actual'] == 1], x='Probability', color='green', label='Actual Survived (1)', kde=True, alpha=0.5, bins=20)
plt.axvline(x=0.5, color='black', linestyle='--', label='Decision Threshold (0.5)')
plt.title('Naive Bayes - Prediction Probability Distribution')
plt.xlabel('Predicted Probability of Survival')
plt.ylabel('Count of Passengers')
plt.legend()
plt.show()

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# =============================================================================
# --- RANDOM FOREST HYPERPARAMETER TUNING (GRID SEARCH) ---
# =============================================================================

param_grid_rf = {
    'n_estimators': [50, 100, 200],          # Number of trees in the forest
    'max_depth': [3, 5, 7, None],            # Maximum depth of each tree (None means unlimited)
    'min_samples_split': [2, 5, 10],         # Minimum samples required to split an internal node
    'min_samples_leaf': [1, 2, 4]            # Minimum samples required to be at a leaf node
}

base_rf = RandomForestClassifier(random_state=42)

# cv=5 means 5-Fold Cross Validation
# n_jobs=-1 uses all available CPU cores to speed up the process
grid_search_rf = GridSearchCV(estimator=base_rf, param_grid=param_grid_rf, 
                              cv=5, scoring='accuracy', n_jobs=-1, verbose=1)

print("\n--- Tuning Random Forest Hyperparameters ---")

grid_search_rf.fit(X_train, y_train)


best_rf_model = grid_search_rf.best_estimator_
best_rf_params = grid_search_rf.best_params_


y_pred_best_rf = best_rf_model.predict(X_test)
y_prob_best_rf = best_rf_model.predict_proba(X_test)[:, 1]

# Calculate metrics
accuracy_best_rf = accuracy_score(y_test, y_pred_best_rf)
cm_best_rf = confusion_matrix(y_test, y_pred_best_rf)

print("\n=============================================")
print("        OPTIMIZED RANDOM FOREST RESULTS       ")
print("=============================================")
print(f"Best Parameters Found: {best_rf_params}")
print(f"Optimized Test Accuracy: {accuracy_best_rf:.2%}")
print("\nConfusion Matrix (Raw Counts):")
print(cm_best_rf)
print("\nOptimized Detailed Classification Report:")
print(classification_report(y_test, y_pred_best_rf))
print("=============================================")

# 7. Plot 1: Confusion Matrix Visual
plt.figure(figsize=(5, 4))
sns.heatmap(cm_best_rf, annot=True, fmt='d', cmap='Reds', 
            xticklabels=['Predicted Dead (0)', 'Predicted Survived (1)'],
            yticklabels=['Actual Dead (0)', 'Actual Survived (1)'])
plt.title('Optimized Random Forest - Confusion Matrix')
plt.ylabel('Actual Labels')
plt.xlabel('Predicted Labels')
plt.show()

# 8. Plot 2: Model Fit & Probability Distribution
prob_df_best_rf = pd.DataFrame({
    'Actual': y_test,
    'Probability': y_prob_best_rf
})

plt.figure(figsize=(8, 5))
# Plot distribution for those who actually died
sns.histplot(data=prob_df_best_rf[prob_df_best_rf['Actual'] == 0], x='Probability', color='red', label='Actual Dead (0)', kde=True, alpha=0.5, bins=20)
# Plot distribution for those who actually survived
sns.histplot(data=prob_df_best_rf[prob_df_best_rf['Actual'] == 1], x='Probability', color='green', label='Actual Survived (1)', kde=True, alpha=0.5, bins=20)

plt.axvline(x=0.5, color='black', linestyle='--', label='Decision Threshold (0.5)')
plt.title('Optimized Random Forest - Prediction Probability Distribution')
plt.xlabel('Predicted Probability of Survival')
plt.ylabel('Count of Passengers')
plt.legend()
plt.show()

# =============================================================================
# --- FINAL OPTIMIZED ENSEMBLE LEARNING (SOFT VOTING) ---
# =============================================================================
from sklearn.ensemble import VotingClassifier

# Create the Ensemble Model using Soft Voting
# Combining diverse perspectives: Linear (LR), Distance-based (KNN Pipeline), and Scaled Tree (Best RF)
final_ensemble_clf = VotingClassifier(
    estimators=[
        ('lr', log_reg),
        ('knn', knn_pipeline),  # Using the pipeline version to ensure scaling inside the ensemble
        ('rf', best_rf_model)   # Using the pruned, healthy, optimized Random Forest
    ],
    voting='soft'               # Smart probability-based weighted voting
)

final_ensemble_clf.fit(X_train, y_train)

y_pred_final = final_ensemble_clf.predict(X_test)
y_prob_final = final_ensemble_clf.predict_proba(X_test)[:, 1]

# 4. Calculate Performance Metrics
accuracy_final = accuracy_score(y_test, y_pred_final)
cm_final = confusion_matrix(y_test, y_pred_final)

print("\n=============================================")
print("       FINAL OPTIMIZED ENSEMBLE RESULTS       ")
print("=============================================")
print(f"Final Test Accuracy: {accuracy_final:.2%}")
print("\nConfusion Matrix (Raw Counts):")
print(cm_final)
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred_final))
print("=============================================")

# 6. Plot 1: Confusion Matrix Visual
plt.figure(figsize=(5, 4))
sns.heatmap(cm_final, annot=True, fmt='d', cmap='Dark2', 
            xticklabels=['Predicted Dead (0)', 'Predicted Survived (1)'],
            yticklabels=['Actual Dead (0)', 'Actual Survived (1)'])
plt.title('Final Ensemble - Confusion Matrix')
plt.ylabel('Actual Labels')
plt.xlabel('Predicted Labels')
plt.show()

# 7. Plot 2: Model Fit & Probability Distribution
prob_df_final = pd.DataFrame({
    'Actual': y_test,
    'Probability': y_prob_final
})

plt.figure(figsize=(8, 5))
sns.histplot(data=prob_df_final[prob_df_final['Actual'] == 0], x='Probability', color='red', label='Actual Dead (0)', kde=True, alpha=0.5, bins=20)
sns.histplot(data=prob_df_final[prob_df_final['Actual'] == 1], x='Probability', color='green', label='Actual Survived (1)', kde=True, alpha=0.5, bins=20)
plt.axvline(x=0.5, color='black', linestyle='--', label='Decision Threshold (0.5)')
plt.title('Final Ensemble - Prediction Probability Distribution')
plt.xlabel('Predicted Probability of Survival')
plt.ylabel('Count of Passengers')
plt.legend()
plt.show()










