import pandas as pd
from sklearn.model_selection import train_test_split
import category_encoders as ce
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('graduation_dataset.csv')

'''print(df.info())
print(df.head())'''

# Identified no missing values (dataset info also says no missing values)

# One-hot encoding of "target"
one_hot = pd.get_dummies(df['Marital status '], prefix='Marital status').astype(int)
df = df.drop('Marital status ', axis=1)
df = df.join(one_hot)

# Label encode target into dropout 1 and enrolled/graduate 0
df['Dropout'] = df['Target'].map({'Dropout': 1, 'Enrolled': 0, 'Graduate': 0})

# Identify target columns and columns for target encoding (categorical columns with many unique values)
cat_cols = ['Application mode ', 'Course ', 'Previous qualification ', 'Nacionality ', "Mother's qualification ",
             "Father's qualification ", "Mother's occupation ", "Father's occupation "]

print(df.info())
print(df.head())

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

print(X_train.head())



# OUTLIER CODE BELOW !!!!!!!!
# Identify all numerical columns for outlier detection
outlier_columns = df.select_dtypes(include=['float64', 'int64']).columns

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

# Outlier handling process for all numerical columns
for col in outlier_columns:
    # Calculate IQR bounds with the specified multiplier
    lower_bound, upper_bound = calculate_iqr_bounds(df, col, iqr_multiplier)
    outlier_count = count_outliers(df, col, lower_bound, upper_bound)
    
    # Display diagnostics for the current column
    print(f"Processing '{col}':")
    print(f"  - Lower Bound: {lower_bound}")
    print(f"  - Upper Bound: {upper_bound}")
    print(f"  - Outliers Detected: {outlier_count}")
    
    # If outliers are detected, apply removal or capping as needed
    if outlier_count > 0:
        print(f"Handling {outlier_count} outliers in '{col}'")
        
        # Option 1: Remove outliers
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        # Option 2: Cap outliers (if you prefer capping instead of removal)
        # df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    else:
        print(f"No outliers found in '{col}' after adjusting IQR multiplier.")

# Save the cleaned DataFrame to a new CSV
output_path = 'cleaned_data.csv'
df.to_csv(output_path, index=False)
print(f"Outlier handling complete. Cleaned dataset saved to '{output_path}'.")

# Min-max scaling
# Fit the scaler on the training set as to avoid data leakage onto test set
cols = [
    'Age at enrollment',
    'Curricular units 1st sem (grade)',
    'Curricular units 2nd sem (grade)',
    'Unemployment rate',
    'Inflation rate',
    'GDP',
    'Curricular units 1st sem (credited)',
    'Curricular units 1st sem (enrolled)',
    'Curricular units 1st sem (evaluations)',
    'Curricular units 1st sem (approved)',
    'Curricular units 1st sem (without evaluations)',
    'Curricular units 2nd sem (credited)',
    'Curricular units 2nd sem (enrolled)',
    'Curricular units 2nd sem (evaluations)',
    'Curricular units 2nd sem (approved)',
    'Curricular units 2nd sem (without evaluations)',
    'Application mode',
    'Application order',
    'Course',
    'Daytime/evening attendance',
    'Previous qualification',
    'Nacionality',
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation"
]
scaler = MinMaxScaler() 
X_train = scaler.fit_transform(X_train[cols])
X_test = scaler.transform(X_test[cols])
print(X_train[:5])# print first 5 rows of X_train

