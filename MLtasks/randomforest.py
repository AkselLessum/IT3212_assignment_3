import pandas as pd
from sklearn.model_selection import train_test_split
import category_encoders as ce
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier

df = pd.read_csv('graduation_dataset.csv')

'''print(df.info())
print(df.head())'''

# Identified no missing values (dataset info also says no missing values)

# One-hot encoding of "target"
one_hot = pd.get_dummies(df['Marital status'], prefix='Marital status').astype(int)
df = df.drop('Marital status', axis=1)
df = df.join(one_hot)

# Label encode target into dropout 1 and enrolled/graduate 0
df['Dropout'] = df['Target'].map({'Dropout': 1, 'Enrolled': 0, 'Graduate': 0})
df = df.drop('Target', axis=1)

# Identify target columns and columns for target encoding (categorical columns with many unique values)
cat_cols = ['Application mode', 'Course', 'Previous qualification', 'Nacionality', "Mother's qualification",
             "Father's qualification", "Mother's occupation", "Father's occupation"]


# Split the dataset into test and train
X = df.drop('Dropout', axis=1)
y = df['Dropout']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Target encoding
# Fit the encoding on the training set as to avoid data leakage onto test set

# encoder = ce.TargetEncoder(cols=cat_cols, smoothing=0.3)
encoder = ce.TargetEncoder(cols=cat_cols)
X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols], y_train)
X_test[cat_cols] = encoder.transform(X_test[cat_cols])



# OUTLIER CODE BELOW !!!!!!!!
# Identify all numerical columns for outlier detection
outlier_columns = [
    'Age at enrollment',
    'Curricular units 1st sem (grade)',
    'Curricular units 2nd sem (grade)',
    'Unemployment rate',
    'Inflation rate',
    'GDP'
]
# Set the IQR multiplier (adjust for stricter outlier detection)
iqr_multiplier = 1.5

# Define the IQR-based outlier handling functions
def calculate_iqr_bounds(df, column, multiplier):
    """Calculate the IQR bounds for a given column with a specified multiplier."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    return lower_bound, upper_bound

def count_outliers(df, column, lower_bound, upper_bound):
    """Count the number of outliers in a column based on IQR bounds."""
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return len(outliers)

for col in outlier_columns:
    lower_bound, upper_bound = calculate_iqr_bounds(df, col, iqr_multiplier)
    outlier_count = count_outliers(df, col, lower_bound, upper_bound)
    
    '''print(f"Processing '{col}':")
    print(f"  - Lower Bound: {lower_bound}")
    print(f"  - Upper Bound: {upper_bound}")
    print(f"  - Outliers Detected: {outlier_count}")'''
    
    # Cap the outliers within the specified bounds
    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

# Save the cleaned DataFrame to a new CSV
output_path = 'cleaned_data.csv'
df.to_csv(output_path, index=False)
print(f"Outlier handling complete. Cleaned dataset saved to '{output_path}'.")


#RFE AND LDA
# Create an estimator to be used by RFE
estimator = LogisticRegression(max_iter=2000)
rfe = RFE(estimator, n_features_to_select=15)
rfe.fit(X_train, y_train)
# Select the features that RFE gets
X_train_selected = rfe.transform(X_train)
X_test_selected = rfe.transform(X_test)

'''selected_features = pd.DataFrame(X_train_selected, columns=X_train.columns[rfe.support_])
print(selected_features.info())'''

# LDA wants fewer dimensions than the number of classes: 1 class for binary classification
'''lda = LinearDiscriminantAnalysis(n_components=1)
X_train_lda = lda.fit_transform(X_train_selected, y_train)
X_test_lda = lda.transform(rfe.transform(X_test))'''

# Min-max scaling
scaler = MinMaxScaler()
X_train_pca = scaler.fit_transform(X_train_selected)
X_test_pca = scaler.transform(X_test_selected)
#print(X_train_pca)

# Do PCA to not reduce dimensionality too far
pca = PCA(n_components=8)
X_train_pca = pca.fit_transform(X_train_pca)
X_test_pca = pca.transform(X_test_pca)


# Random forest code
# Create model
model = RandomForestClassifier(n_estimators=200)
model.fit(X_train_pca, y_train)

y_pred_test = model.predict(X_test_pca)
y_pred_train = model.predict(X_train_pca)

print("Accuracy scores before bagging and boosting:")
print("Random Forest accuracy (TRAIN): ", accuracy_score(y_train, y_pred_train))
print("Random Forest accuracy (TEST): ", accuracy_score(y_test, y_pred_test))
print("---------------------------------------------------------------")


estimator_range = [2,4,6,8,10,12,14,16]
scoresBag = []
scoresBoost = []

for n_estimators in estimator_range:
    # Create the bagging classifier
    model_bag = BaggingClassifier(n_estimators=n_estimators, estimator=model, random_state=42)
    # Fit on training set
    model_bag.fit(X_train_pca, y_train)
    pred = model_bag.predict(X_test_pca)
    scoresBag.append(accuracy_score(y_test, pred))

i=2
for score in scoresBag:
    print("Random forest accuracy bagged", i, "base estimators:", score)
    i = i+2
print("---------------------------------------------------------------")

for n_estimators in estimator_range:
    adaboost_rf = AdaBoostClassifier(estimator=model, n_estimators=n_estimators, learning_rate=0.1)
    adaboost_rf.fit(X_train_pca, y_train)
    pred = adaboost_rf.predict(X_test_pca)
    scoresBoost.append(accuracy_score(y_test, pred))

i=2
for score in scoresBoost:
    print("Random forest accuracy boosted", i, "base estimators:", score)
    i = i+2
print("---------------------------------------------------------------")