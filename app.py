import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st

#Title for app
st.write("""
# ALA Student GPA Predictor
Using social and academic metrics to predict students' GPA at the African Leadership Academy in Johannesburg, South Africa \n
""")

#open and display image
image = Image.open('C:/Users/User/Desktop/diabetes_detector/ala.jpg')
st.image(image, caption = 'Diabetes', use_column_width = True)

#Get Data
df= pd.read_csv('C:/Users/User/Desktop/diabetes_detector/diabetes.csv')

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
    pregnancies = st.sidebar.slider('Hours of Sleep', 0, 24, 3)
    glucose = st.sidebar.slider('Hours of Study', 0, 24, 3)
    skin_thickness = st.sidebar.slider('Hours on Social Media**', 0, 24, 2)
    blood_pressure = st.sidebar.slider('Previous GPA', 0, 100, 72)
    skin_thickness = st.sidebar.slider('Weeks to Finals', 1, 20, 10)
    skin_thickness = st.sidebar.slider('Class Attendance Rate', 0, 100, 50)

    user_data = {
        'Hours of sleep' : pregnancies,
        'Hours of Study': glucose,
        'Previous GPA': blood_pressure,
        'Weeks to Finals': skin_thickness,
        ' ': 0,
        'bmi': 0,
        'dpf': 0,
        'age': 0,
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
st.subheader('Final GPA:')
st.write(str(accuracy_score(Y_test, RandomForestClassifier.predict(X_test)) * 100)+'%')

# Store the model's predictions ina viriable
prediction = RandomForestClassifier.predict(user_input)


# Set a subheadr and display the classification
# st.header('Classification: ')
# if prediction ==1:
#     st.header("""
#         Final GPA: C
#     """)
# else:
#     st.header("""
#            Final GPA: A
#        """)









