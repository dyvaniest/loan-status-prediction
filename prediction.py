# %% [markdown]
# Importing the Dependencies

# %%
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import joblib

# %%
# Title of the Streamlit app
st.title('Loan Prediction System')

# %% [markdown]
# Data Collection and Processing

# %%
# Importing the dataset
st.write("### Data Collection and Processing")

# %%
# loading the dataset to pandas DataFrame
@st.cache_data
def load_data():
    data = pd.read_csv('dataset.csv')
    return data

# Loading data
loan_dataset = load_data()

# %%
# Display dataset
st.write("#### Loaded Dataset")
st.dataframe(loan_dataset)

st.write("#### 5 First Loaded Dataset")
st.dataframe(loan_dataset.head())

# %%
# Data Preprocessing
st.write("### Data Preprocessing")

# %%
# Dropping missing values
loan_dataset = loan_dataset.dropna()

# %%
# Encoding categorical values
loan_dataset.replace({"Loan_Status": {'N': 0, 'Y': 1}}, inplace=True)
loan_dataset.replace({'Married': {'No': 0, 'Yes': 1}, 'Gender': {'Male': 1, 'Female': 0},
                      'Self_Employed': {'No': 0, 'Yes': 1}, 'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2},
                      'Education': {'Graduate': 1, 'Not Graduate': 0}}, inplace=True)
loan_dataset = loan_dataset.replace(to_replace='3+', value=4)

# %%
# Display preprocessed dataset
st.write("#### Preprocessed Dataset")
st.dataframe(loan_dataset.head())

# %% [markdown]
# Data Visualization

# %%
# Data Visualization
st.write("### Data Visualization")

# %%
# education & Loan Status
fig, ax = plt.subplots()
sns.countplot(x='Education', hue='Loan_Status', data=loan_dataset, ax=ax)
st.pyplot(fig)

# %%
# marital status & Loan Status
sfig, ax = plt.subplots()
sns.countplot(x='Married', hue='Loan_Status', data=loan_dataset, ax=ax)
st.pyplot(fig)

# %%
# Separating the data and labels
X = loan_dataset.drop(columns=['Loan_ID', 'Loan_Status'], axis=1)
Y = loan_dataset['Loan_Status']

# %%
# Convert Y to integer type
Y = Y.astype(int)

# %% [markdown]
# Train Test Split

# %%
# Train Test Split
@st.cache_data
def split_data(X, Y):
    return train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=2)

# %%
X_train, X_test, Y_train, Y_test = split_data(X, Y)

# %%
# Convert Y_train and Y_test to integer type
Y_train = Y_train.astype(int)
Y_test = Y_test.astype(int)

# %% [markdown]
# Training the model:
# 
# Support Vector Machine Model

# %%
# Model Training and Saving
@st.cache_data
def train_model(X_train, Y_train):
    classifier = svm.SVC(kernel='linear')
    classifier.fit(X_train, Y_train)
    joblib.dump(classifier, 'loan_predictor.joblib')
    return classifier

# %%
#training the support Vector Macine model
classifier = train_model(X_train, Y_train)

# %%
# Load trained model
@st.cache_data
def load_model():
    return joblib.load('loan_predictor.joblib')

# %%
classifier = load_model()

# %% [markdown]
# Model Evaluation

# %%
# Model Evaluation
X_train_prediction = classifier.predict(X_train)
X_test_prediction = classifier.predict(X_test)

# %%
# Convert predictions to integer type
X_train_prediction = X_train_prediction.astype(int)
X_test_prediction = X_test_prediction.astype(int)

# %%
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

# %%
st.write("### Model Evaluation")
st.write(f'Accuracy on training data: {training_data_accuracy}')
st.write(f'Accuracy on test data: {test_data_accuracy}')

# %% [markdown]
# Making a predictive system

# %%
# Predictive System
st.write("### Make a Prediction")

# %%
def user_input_features():
    Gender = st.selectbox('Gender', ('Male', 'Female'))
    Married = st.selectbox('Married', ('No', 'Yes'))
    Dependents = st.selectbox('Dependents', (0, 1, 2, 3, 4))
    Education = st.selectbox('Education', ('Graduate', 'Not Graduate'))
    Self_Employed = st.selectbox('Self Employed', ('No', 'Yes'))
    ApplicantIncome = st.number_input('Applicant Income', min_value=0)
    CoapplicantIncome = st.number_input('Coapplicant Income', min_value=0)
    LoanAmount = st.number_input('Loan Amount', min_value=0)
    Loan_Amount_Term = st.number_input('Loan Amount Term', min_value=0)
    Credit_History = st.selectbox('Credit History', (0, 1))
    Property_Area = st.selectbox('Property Area', ('Rural', 'Semiurban', 'Urban'))
    
    data = {
        'Gender': 1 if Gender == 'Male' else 0,
        'Married': 1 if Married == 'Yes' else 0,
        'Dependents': Dependents,
        'Education': 1 if Education == 'Graduate' else 0,
        'Self_Employed': 1 if Self_Employed == 'Yes' else 0,
        'ApplicantIncome': ApplicantIncome,
        'CoapplicantIncome': CoapplicantIncome,
        'LoanAmount': LoanAmount,
        'Loan_Amount_Term': Loan_Amount_Term,
        'Credit_History': Credit_History,
        'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2}[Property_Area]
    }
    features = pd.DataFrame(data, index=[0])
    return features

# %%
input_df = user_input_features()
st.write("### User Input Parameters")
st.dataframe(input_df)

# %%
# Prediction
if st.button('Predict'):
    # Ensure all inputs are provided
    if None in input_df.values or any(input_df.isnull().any()):
        st.error("Please provide all input parameters.")
    else:
        prediction = classifier.predict(input_df)
        prediction_proba = classifier.decision_function(input_df)

        st.write(f"Loan Status Prediction: {'Approved' if prediction[0] == 1 else 'Rejected'}")
        st.write(f"Prediction Confidence: {prediction_proba[0]:.2f}")

# %%
# Running the app
if __name__ == '__main__':
    st.write("Streamlit Loan Prediction App")


