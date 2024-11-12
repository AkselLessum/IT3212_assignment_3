import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from pygam import LogisticGAM, s

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

# 2. Gradient Boosting Classifier
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)

print("\nGradient Boosting Model Accuracy:", accuracy_score(y_test, gb_pred))
print("\nGradient Boosting Model Classification Report:\n", classification_report(y_test, gb_pred))

# 3. Random Forest Classifier (Bagging)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

print("\nRandom Forest Bagging Model Accuracy:", accuracy_score(y_test, rf_pred))
print("\nRandom Forest Bagging Model Classification Report:\n", classification_report(y_test, rf_pred))
