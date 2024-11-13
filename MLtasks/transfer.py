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
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

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

# Min-max scaling
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train_selected)
X_test = scaler.transform(X_test_selected)
#print(X_train_pca)

#Train
X_source_train, X_target_train, y_source_train, y_target_train = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
#Test
X_source_test, X_target_test, y_source_test, y_target_test = train_test_split(X_test, y_test, test_size=0.1, random_state=42)



# Do PCA to not reduce dimensionality too far
pca = PCA(n_components=8)
X_source_train_pca = pca.fit_transform(X_source_train)
X_source_test_pca = pca.transform(X_source_test)

pca_target = PCA(n_components=4) # Changed the components to mix up the set
X_target_train_pca = pca_target.fit_transform(X_target_train)
X_target_test_pca = pca_target.transform(X_target_test)

# SVM linear
svm_source = SVC(kernel='linear', C=1, probability=True)
svm_source.fit(X_source_train_pca, y_source_train)
pred_linear = svm_source.predict(X_source_test_pca) # Predict on the test set, only big model
print("\"BIG MODEL\" SVM accuracy score (Linear): ", accuracy_score(y_source_test, pred_linear))

svm_target = svm_source
svm_target.fit(X_target_train_pca, y_target_train)
pred_final = svm_target.predict(X_target_test_pca) # Predict on the test set, only small model
print("\"Transfer\" SVM accuracy score (Linear): ", accuracy_score(y_target_test, pred_final))

#"BIG MODEL" SVM accuracy score (Linear):  0.8203517587939698
#"Transfer" SVM accuracy score (Linear):  0.7303370786516854

#TODO: plot

# Transfer learning code
# use 90% of the data for transfer training
# Train ours on the 10 remaining percent
# within themselves we need to have a training set (80%) and a test set (20%)
# Somehow use the big dataset on the smaller one
# For our case, combine the test sets to make a validation set for the full implementation. 

# 1. Train the SVM on the source (larger) dataset
#svm_source = SVC(kernel='linear', random_state=42)
#svm_source.fit(X_source_scaled, y_source)

# Evaluate the model on the source dataset (just to check how it performs on the source)
#y_pred_source = svm_source.predict(X_source_scaled)
#print("Accuracy on the source dataset:", accuracy_score(y_source, y_pred_source))

# 2. Fine-tune the SVM model on the target (smaller) dataset
#svm_target = svm_source  # Use the trained model as a starting point
#svm_target.fit(X_target_scaled, y_target)