import streamlit as st
import pandas as pd
import joblib

st.write("""
# China Suicide Method Prediction App
This app predicts the **Suicide Method** of the suicide victim in Shandong, China!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    dead = st.sidebar.number_input('Is the victim dead? (0 represents No, 1 represents Yes)', 0, 1, 0)
    urban = st.sidebar.number_input('Does the victim live in an urban area? (0 represents No, 1 represents Yes)', 0, 1, 0)
    year = st.sidebar.number_input('Which year did the incident take place?', 2009, 2011, 2010)
    month = st.sidebar.number_input('Which month did the incident take place?', 1, 12, 6)
    gender = st.sidebar.number_input('What is the gender of the victim? (0 represents Female, 1 represents Male)', 0, 1, 0)
    age = st.sidebar.number_input('How old was the victim?', 12, 100, 40)
    education = st.sidebar.selectbox('What is the education level of the victim?', ('Iliterate', 'Primary', 'Secondary', 'Tertiary', 'Unknown'))
    job = st.sidebar.selectbox('What is the occupation of the victim', ('household', 'farming', 'others/unknown', 'professional', 'unemployed', 'business/service', 'student', 'worker', 'others', 'retiree'))
    data = {'Died': dead,
            'Urban': urban,
            'Year': year,
            'Month': month,
            'Sex': gender,
            'Age': age,
            'Education': education,
            'Occupation': job}
    features = pd.DataFrame(data, index=[0])
    features['Education'] = features['Education'].map({'Iliterate' : 0, 'Primary' : 1, 'Secondary' : 2, 'Tertiary' : 3, 'Unknown' : 4})
    features_encoded = pd.get_dummies(features, columns=['Occupation'])  # Use one-hot encoding on 'Occupation' column
    missing_cols = set(['Occupation_' + str(i) for i in range(10)]) - set(features_encoded.columns)
    for col in missing_cols:
        features_encoded[col] = 0
        features_encoded = features_encoded[['Died', 'Urban', 'Year', 'Month', 'Sex', 'Age', 'Education'] +
                                        sorted(features_encoded.columns[2:])]
    return features_encoded

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

model = joblib.load('suicide_prediction_model.pkl') #Import trained model

prediction = model.predict(df)
prediction_proba = model.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
target = ['Cutting', 'Drowning', 'Hanging', 'Jumping', 'Other poison',  'Pesticide', 'Poison unspec', 'Unspecified', 'Others']
st.write(target)

st.subheader('Prediction')
st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)