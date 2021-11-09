import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st

#Title for app
st.write("""
# Diabetes Detection App
Detect if someone has diabetes using machine learning and Python
""")

#open and display image
# image = Image.open('C:/Users/User/Desktop/python/Diabetes Detector/causes-of-diabetes.jfif')
# st.image(image, caption = 'Diabetes', use_column_width = True)

#Get Data
df= pd.read_csv('C:/Users/User/Desktop/python/Diabetes Detector/diabetes.csv')

#Set subheader on the web app
st.subheader('Data Information:')
#Show data as table
st.dataframe(df)
#Show statistics on the data
st.write(df.describe())
# Show the data as chart
chart = st.bar_chart(df)

# Split data into independent X and dependent Y
X = df.iloc[:, 0:8]
Y = df.iloc[:, -1].values

#Split into testing and training

X_train, X_test ,  Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state = 0)

# Get feature input for the user
def get_user_input():
    pregnancies = st.sidebar.slider('pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('glucose', 0, 199, 117)
    blood_pressure = st.sidebar.slider('blood_pressure', 0, 122, 72)
    skin_thickness = st.sidebar.slider('skin_thickness', 0, 99, 23)
    insulin = st.sidebar.slider('insulin', 0.0, 846.0, 30.0)
    bmi = st.sidebar.slider('bmi', 0.0, 67.1, 32.0)
    dpf = st.sidebar.slider('dpf', 0.078, 2.42,0.3725)
    age = st.sidebar.slider('age', 21, 81, 29)
    # Store dictionary into variable
    user_data = {
        'pregnancies' : pregnancies,
        'glucose': glucose,
        'blood_pressure': blood_pressure,
        'skin_thickness': skin_thickness,
        'insulin': insulin,
        'bmi': bmi,
        'dpf': dpf,
        'age': age,
    }
    # Transform the data into a data frame
    features = pd.DataFrame(user_data, index = [0])
    return features
#Store inputs to vairable
user_input = get_user_input()

# Set a subheader and display the user's inputs
st.subheader('User Inputs:')
st.write(user_input)

# Create and train the model
RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train, Y_train)


#show the model metrics
st.subheader('Accuracy:')
st.write(str(accuracy_score(Y_test, RandomForestClassifier.predict(X_test)) * 100)+'%')

# Store the model's predictions ina viriable
prediction = RandomForestClassifier.predict(user_input)


# Set a subheadr and display the classification
st.subheader('Classification: ')
st.write(prediction)







