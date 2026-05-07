import nbformat as nbf

nb = nbf.v4.new_notebook()

text1 = """# Credit Card Fraud Detection
In this notebook, we will build a Machine Learning model to detect fraudulent credit card transactions. 
We will use Logistic Regression on an undersampled dataset to handle class imbalance.
"""

code1 = """import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib"""

text2 = """## Loading the Dataset"""

code2 = """# Loading the dataset to a Pandas DataFrame
credit_card_data = pd.read_csv('creditcard.csv')

# First 5 rows of the dataset
credit_card_data.head()"""

code3 = """# Dataset Information
credit_card_data.info()"""

code4 = """# Checking the number of missing values in each column
credit_card_data.isnull().sum()"""

text3 = """## Handling Imbalanced Data
The dataset is highly unbalanced. Let's see the distribution."""

code5 = """# Distribution of legit transactions & fraudulent transactions
credit_card_data['Class'].value_counts()"""

text4 = """0 --> Normal Transaction
1 --> fraudulent transaction

We will separate the data for analysis."""

code6 = """# Separating the data for analysis
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]

print(legit.shape)
print(fraud.shape)"""

text5 = """### Undersampling
Build a sample dataset containing similar distribution of normal transactions and Fraudulent Transactions.
Number of Fraudulent Transactions is around 492."""

code7 = """legit_sample = legit.sample(n=len(fraud), random_state=42)"""

text6 = """Concatenating two DataFrames"""

code8 = """new_dataset = pd.concat([legit_sample, fraud], axis=0)

new_dataset.head()"""

code9 = """new_dataset['Class'].value_counts()"""

text7 = """## Splitting the data into Features & Targets"""

code10 = """X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']"""

code11 = """X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)
print(X.shape, X_train.shape, X_test.shape)"""

text8 = """## Model Training
Logistic Regression"""

code12 = """model = LogisticRegression(max_iter=1000)
# Training the Logistic Regression Model with Training Data
model.fit(X_train, Y_train)"""

text9 = """## Model Evaluation"""

code13 = """# Accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on Training data : ', training_data_accuracy)"""

code14 = """# Accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy on Test data : ', test_data_accuracy)"""

text10 = """## Saving the Model"""

code15 = """import joblib
joblib.dump(model, 'model.pkl')
print("Model saved to model.pkl successfully!")"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text1),
    nbf.v4.new_code_cell(code1),
    nbf.v4.new_markdown_cell(text2),
    nbf.v4.new_code_cell(code2),
    nbf.v4.new_code_cell(code3),
    nbf.v4.new_code_cell(code4),
    nbf.v4.new_markdown_cell(text3),
    nbf.v4.new_code_cell(code5),
    nbf.v4.new_markdown_cell(text4),
    nbf.v4.new_code_cell(code6),
    nbf.v4.new_markdown_cell(text5),
    nbf.v4.new_code_cell(code7),
    nbf.v4.new_markdown_cell(text6),
    nbf.v4.new_code_cell(code8),
    nbf.v4.new_code_cell(code9),
    nbf.v4.new_markdown_cell(text7),
    nbf.v4.new_code_cell(code10),
    nbf.v4.new_code_cell(code11),
    nbf.v4.new_markdown_cell(text8),
    nbf.v4.new_code_cell(code12),
    nbf.v4.new_markdown_cell(text9),
    nbf.v4.new_code_cell(code13),
    nbf.v4.new_code_cell(code14),
    nbf.v4.new_markdown_cell(text10),
    nbf.v4.new_code_cell(code15)
]

with open('credit_card_fraud_training.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook generated successfully!")
