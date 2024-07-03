import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

# Load the dataset
dataset = pd.read_csv('hiring.csv')

# Rename columns for simplicity
dataset.columns = ['experience', 'test_score', 'interview_score', 'salary']

# Fill missing values in 'experience' and 'test_score'
dataset['experience'] = dataset['experience'].fillna(0)
dataset['test_score'] = dataset['test_score'].fillna(dataset['test_score'].mean())

# Convert text in 'experience' to numerical values
def convert_experience(x):
    if isinstance(x, str):
        if x.isdigit():
            return int(x)
        else:
            # Handle specific cases
            word_dict = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
                         'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
                         'eleven': 11, 'twelve': 12, 'zero': 0}
            return word_dict.get(x.lower(), 0)
    return x

dataset['experience'] = dataset['experience'].apply(convert_experience)

# Separate the features and the target variable
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Encoding categorical data (if necessary)
# If the 'experience' column has been converted to numerical values, no need for LabelEncoder

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Saving the model to disk
pickle.dump(regressor, open('model.pkl', 'wb'))

# Loading the model to compare the results
model = pickle.load(open('model.pkl', 'rb'))
print(model.predict([[2, 9, 6]]))
