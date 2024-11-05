import pandas as pd
from sklearn.model_selection import train_test_split
import category_encoders as ce

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

encoder = ce.TargetEncoder(cols=cat_cols, smoothing=0.3)
X_train = encoder.fit_transform(X_train[cat_cols], y_train)
X_test = encoder.transform(X_test[cat_cols])

print(X_train.head())

