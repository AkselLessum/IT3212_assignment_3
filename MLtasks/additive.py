import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from pygam import LogisticGAM, s
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
data = pd.read_csv('../graduation_dataset.csv')

# Prepare features and target variable
X = data.iloc[:, :-1]
y = (data['Target'] == 'Graduate').astype(int)  # Binary encoding: 1 for 'Graduate', 0 for 'Dropout'

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Generalized Additive Model (GAM)
gam = LogisticGAM(s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) +
                  s(8) + s(9) + s(10) + s(11) + s(12) + s(13) + s(14) +
                  s(15) + s(16) + s(17) + s(18) + s(19) + s(20) + s(21) +
                  s(22) + s(23) + s(24) + s(25) + s(26) + s(27) + s(28) +
                  s(29) + s(30) + s(31) + s(32) + s(33)).fit(X_train, y_train)
gam_pred = gam.predict(X_test)

print("GAM Model Accuracy:", accuracy_score(y_test, gam_pred))
print("\nGAM Model Classification Report:\n", classification_report(y_test, gam_pred))

# 2. Bagging Classifier
bagging_model = BaggingClassifier(n_estimators=100, random_state=42)
bagging_model.fit(X_train, y_train)
bagging_pred = bagging_model.predict(X_test)

print("\nBagging Model Accuracy:", accuracy_score(y_test, bagging_pred))
print("\nBagging Model Classification Report:\n", classification_report(y_test, bagging_pred))

# 3. AdaBoost Classifier
adaboost_model = AdaBoostClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
adaboost_model.fit(X_train, y_train)
adaboost_pred = adaboost_model.predict(X_test)

print("\nAdaBoost Model Accuracy:", accuracy_score(y_test, adaboost_pred))
print("\nAdaBoost Model Classification Report:\n", classification_report(y_test, adaboost_pred))

# FIGURE CODE BELOW
# Define the range of n_estimators to test
estimator_range = np.arange(2, 18, 2)  # 2, 4, 6, 8, 10, 12, 14, 16
scores_bagging = []   # To store accuracy scores for bagging
scores_boosting = []  # To store accuracy scores for boosting

# Evaluate Bagging model with different n_estimators
for n in estimator_range:
    bagging_model = BaggingClassifier(n_estimators=n, random_state=42)
    bagging_model.fit(X_train, y_train)
    bagging_pred = bagging_model.predict(X_test)
    acc = accuracy_score(y_test, bagging_pred)
    scores_bagging.append(acc)
    print(f"Bagging Model Accuracy with n_estimators={n}: {acc:.3f}")

# Evaluate AdaBoost model with different n_estimators
for n in estimator_range:
    adaboost_model = AdaBoostClassifier(n_estimators=n, learning_rate=0.1, random_state=42)
    adaboost_model.fit(X_train, y_train)
    adaboost_pred = adaboost_model.predict(X_test)
    acc = accuracy_score(y_test, adaboost_pred)
    scores_boosting.append(acc)
    print(f"AdaBoost Model Accuracy with n_estimators={n}: {acc:.3f}")

# Create a figure with 1 row and 2 columns of subplots
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Original GAM accuracy for comparison
org_acc = round(accuracy_score(y_test, gam_pred), 2)

# Plot for Bagging accuracy over n_estimators
axes[0].plot(estimator_range, scores_bagging)
axes[0].set_title("Accuracy Scores (Bagging)", fontsize=18)
axes[0].set_xlabel("n_estimators", fontsize=18)
axes[0].set_ylabel("Score", fontsize=18)
axes[0].tick_params(labelsize=16)
axes[0].axhline(y=org_acc, color='red', linestyle=':', linewidth=2, label=f'Original GAM score = {org_acc}')
axes[0].legend(fontsize=14)

# Plot for AdaBoost accuracy over n_estimators
axes[1].plot(estimator_range, scores_boosting)
axes[1].set_title("Accuracy Scores (AdaBoost)", fontsize=18)
axes[1].set_xlabel("n_estimators", fontsize=18)
axes[1].set_ylabel("Score", fontsize=18)
axes[1].tick_params(labelsize=16)
axes[1].axhline(y=org_acc, color='red', linestyle=':', linewidth=2, label=f'Original GAM score = {org_acc}')
axes[1].legend(fontsize=14)

# Adjust layout and show the plots
plt.tight_layout()
plt.show()
