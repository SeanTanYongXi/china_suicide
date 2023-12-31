import streamlit as st
import pandas as pd
import joblib

st.write("""
# China Suicide Method Prediction App
This app predicts the **Suicide Method** of the suicide victim in Shandong, China!
""")

st.sidebar.header('User Input Parameters')

encoded_columns = ['Occupation_business/service', 'Occupation_farming', 'Occupation_household', 'Occupation_others', 'Occupation_others/unknown', 'Occupation_professional', 'Occupation_retiree', 'Occupation_student', 'Occupation_unemployed', 'Occupation_worker']

def user_input_features():
    dead = st.sidebar.number_input('Is the victim dead? (0 represents No, 1 represents Yes)', 0, 1, 0)
    urban = st.sidebar.number_input('Does the victim live in an urban area? (0 represents No, 1 represents Yes, 2 represents unknown)', 0, 2, 0)
    year = st.sidebar.number_input('Which year did the incident take place?', 2009, 2011, 2010)
    month = st.sidebar.number_input('Which month did the incident take place?', 1, 12, 6)
    sex = st.sidebar.number_input('What is the gender of the victim? (0 represents Female, 1 represents Male)', 0, 1, 0)
    age = st.sidebar.number_input('How old was the victim?', 12, 100, 40)
    education = st.sidebar.selectbox('What is the education level of the victim?', ('Iliterate', 'Primary', 'Secondary', 'Tertiary', 'Unknown'))
    job = st.sidebar.selectbox('What is the occupation of the victim', ('business/service', 'farming', 'household', 'others', 'others/unknown', 'professional', 'retiree', 'student', 'unemployed', 'worker'))
    data = {'Died': dead,
            'Urban': urban,
            'Year': year,
            'Month': month,
            'Sex': sex,
            'Age': age,
            'Education': education,
            'Occupation': job}
    features = pd.DataFrame(data, index=[0])
    features['Education'] = features['Education'].map({'Iliterate' : 0, 'Primary' : 1, 'Secondary' : 2, 'Tertiary' : 3, 'Unknown' : 4})
    features_encoded = pd.get_dummies(features, columns=['Occupation'], prefix='Occupation')
    missing_cols = set(encoded_columns) - set(features_encoded.columns)
    for col in missing_cols:
        features_encoded[col] = 0
    
    # Ensure columns are in the correct order
    features_encoded = features_encoded[['Died', 'Urban', 'Year', 'Month', 'Sex', 'Age', 'Education'] + encoded_columns]

    return features_encoded

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

model = joblib.load('suicide_prediction_model.pkl') #Import trained model

# Reorder columns to match the order in the model training
columns_order = ['Died', 'Urban', 'Year', 'Month', 'Sex', 'Age', 'Education', 'Occupation_business/service', 'Occupation_farming', 'Occupation_household', 'Occupation_others', 'Occupation_others/unknown', 'Occupation_professional', 'Occupation_retiree', 'Occupation_student', 'Occupation_unemployed', 'Occupation_worker']
df = df[columns_order]

prediction = model.predict(df)
prediction_proba = model.predict_proba(df)

target = ['Cutting', 'Drowning', 'Hanging', 'Jumping', 'Other poison', 'Others', 'Pesticide', 'Poison unspec', 'Unspecified']
class_labels = pd.DataFrame({'Class Label': target, 'Index Number': range(len(target))})

# Display the data table using st.table
st.subheader('Class labels and their corresponding index number')
st.table(class_labels)

st.subheader('Prediction')
st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)