import pandas as pd
from sklearn.model_selection import train_test_split
import category_encoders as ce
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt

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
df = df.drop('Target', axis=1)

# Identify target columns and columns for target encoding (categorical columns with many unique values)
cat_cols = ['Application mode ', 'Course ', 'Previous qualification ', 'Nacionality ', "Mother's qualification ",
             "Father's qualification ", "Mother's occupation ", "Father's occupation "]


# Split the dataset into test and train
X = df.drop('Dropout', axis=1)
y = df['Dropout']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Target encoding
# Fit the encoding on the training set as to avoid data leakage onto test set

encoder = ce.TargetEncoder(cols=cat_cols, smoothing=0.3)
X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols], y_train)
X_test[cat_cols] = encoder.transform(X_test[cat_cols])

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
lda = LinearDiscriminantAnalysis(n_components=1)
X_train_lda = lda.fit_transform(X_train_selected, y_train)
X_test_lda = lda.transform(rfe.transform(X_test))

'''print("Shape of LDA-transformed training data:", X_train_lda.shape)
print("Shape of LDA-transformed test data:", X_test_lda.shape)

# Optionally, you can analyze the LDA components
print("LDA Components:", lda.coef_)  # This shows the direction of the separation'''

# Visualize LDA results
lda_df = pd.DataFrame(data=X_train_lda, columns=['LDA Component'])
lda_df['Target'] = y_train.values  # Add the target variable for coloring

# Scatter plot of LDA components
plt.figure(figsize=(8, 6))
colors = ['red' if label == 1 else 'blue' for label in lda_df['Target']]
plt.scatter(lda_df['LDA Component'], [0]*len(lda_df), c=colors, alpha=0.5)
plt.title('LDA: Projected Data Points')
plt.xlabel('LDA Component 1')
plt.yticks([])  # Hide y-axis
plt.grid()
plt.show()


