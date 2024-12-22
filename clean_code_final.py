import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# List of symptoms
l1 = ['back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes',
      'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise', 
      'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure', 'runny_nose',
      'congestion', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate', 'pain_during_bowel_movements', 
      'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 
      'obesity', 'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails', 
      'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech',
      'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness', 
      'spinning_movements', 'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side', 'loss_of_smell', 
      'bladder_discomfort', 'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 
      'toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 
      'belly_pain', 'abnormal_menstruation', 'watering_from_eyes', 'increased_appetite', 
      'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration', 'visual_disturbances', 
      'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 
      'distention_of_abdomen', 'history_of_alcohol_consumption', 'fluid_overload', 'blood_in_sputum', 
      'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 
      'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister', 
      'red_sore_around_nose', 'yellow_crust_ooze']

# List of diseases
disease = ['Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis', 'Drug Reaction', 'Peptic ulcer diseae', 'AIDS', 
           'Diabetes', 'Gastroenteritis', 'Bronchial Asthma', 'Hypertension', 'Migraine', 'Cervical spondylosis', 
           'Paralysis (brain hemorrhage)', 'Jaundice', 'Malaria', 'Chicken pox', 'Dengue', 'Typhoid', 'Hepatitis A', 
           'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E', 'Alcoholic hepatitis', 'Tuberculosis', 
           'Common Cold', 'Pneumonia', 'Dimorphic hemmorhoids(piles)', 'Heart attack', 'Varicose veins', 
           'Hypothyroidism', 'Hyperthyroidism', 'Hypoglycemia', 'Osteoarthristis', 'Arthritis', 
           '(vertigo) Paroymsal Positional Vertigo', 'Acne', 'Urinary tract infection', 'Psoriasis', 'Impetigo']

# Initialize list for symptoms encoding
l2 = [0] * len(l1)

# Load the datasets
df = pd.read_csv("C:/Users/Shubham/OneDrive/Desktop/PROJ/Disease-prediction-using-Machine-Learning-master/Training.csv")
tr = pd.read_csv("C:/Users/Shubham/OneDrive/Desktop/PROJ/Disease-prediction-using-Machine-Learning-master/Testing.csv")

# Replace disease names with numerical labels
disease_mapping = {disease[i]: i for i in range(len(disease))}
df.replace({'prognosis': disease_mapping}, inplace=True)
tr.replace({'prognosis': disease_mapping}, inplace=True)

# Features and labels for training and testing data
X = df[l1]
y = df[["prognosis"]].values.ravel()

X_test = tr[l1]
y_test = tr[["prognosis"]].values.ravel()

# Ensure labels are converted to numeric
y = pd.to_numeric(y, errors='coerce')
y_test = pd.to_numeric(y_test, errors='coerce')

# Drop any missing labels if found
y = np.nan_to_num(y, nan=-1).astype(int)
y_test = np.nan_to_num(y_test, nan=-1).astype(int)

# Decision Tree Classifier
def DecisionTree(psymptoms):
    clf3 = tree.DecisionTreeClassifier()
    clf3 = clf3.fit(X, y)
    
    # Accuracy
    y_pred = clf3.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Predict disease
    for k in range(len(l1)):
        if l1[k] in psymptoms:
            l2[k] = 1

    inputtest = [l2]
    predicted = clf3.predict(inputtest)[0]
    return disease[predicted], accuracy

# Random Forest Classifier
def RandomForest(psymptoms):
    clf4 = RandomForestClassifier()
    clf4 = clf4.fit(X, y)
    
    # Accuracy
    y_pred = clf4.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Predict disease
    for k in range(len(l1)):
        if l1[k] in psymptoms:
            l2[k] = 1

    inputtest = [l2]
    predicted = clf4.predict(inputtest)[0]
    return disease[predicted], accuracy

# Naive Bayes Classifier
def NaiveBayes(psymptoms):
    gnb = GaussianNB()
    gnb = gnb.fit(X, y)
    
    # Accuracy
    y_pred = gnb.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Predict disease
    for k in range(len(l1)):
        if l1[k] in psymptoms:
            l2[k] = 1

    inputtest = [l2]
    predicted = gnb.predict(inputtest)[0]
    return disease[predicted], accuracy

# Streamlit UI
st.title("Disease Predictor using Machine Learning")

st.write("Enter the symptoms to predict the disease:")

# Input fields for symptoms
symptom1 = st.selectbox("Symptom 1", options=["None"] + l1)
symptom2 = st.selectbox("Symptom 2", options=["None"] + l1)
symptom3 = st.selectbox("Symptom 3", options=["None"] + l1)
symptom4 = st.selectbox("Symptom 4", options=["None"] + l1)
symptom5 = st.selectbox("Symptom 5", options=["None"] + l1)

psymptoms = [symptom1, symptom2, symptom3, symptom4, symptom5]
psymptoms = [sym for sym in psymptoms if sym != "None"]

# Collect accuracy scores for comparison
accuracy_dict = {}

if st.button("Predict Disease with Decision Tree"):
    predicted_disease, accuracy = DecisionTree(psymptoms)
    st.write(f"Predicted Disease: {predicted_disease}")
    st.write(f"Accuracy: {accuracy * 100:.2f}%")
    accuracy_dict['Decision Tree'] = accuracy

if st.button("Predict Disease with Random Forest"):
    predicted_disease, accuracy = RandomForest(psymptoms)
    st.write(f"Predicted Disease: {predicted_disease}")
    st.write(f"Accuracy: {accuracy * 100:.2f}%")
    accuracy_dict['Random Forest'] = accuracy

if st.button("Predict Disease with Naive Bayes"):
    predicted_disease, accuracy = NaiveBayes(psymptoms)
    st.write(f"Predicted Disease: {predicted_disease}")
    st.write(f"Accuracy: {accuracy * 100:.2f}%")
    accuracy_dict['Naive Bayes'] = accuracy

# Plot the accuracy of the classifiers
if accuracy_dict:
    st.write("### Classifier Accuracy Comparison")
    
    # Plot the accuracy as a bar graph
    fig, ax = plt.subplots()
    ax.barh(list(accuracy_dict.keys()), list(accuracy_dict.values()), color=['blue', 'green', 'red'])
    ax.set_xlabel('Accuracy')
    ax.set_title('Classifier Accuracy Comparison')
    
    # Display the plot in Streamlit
    st.pyplot(fig)
