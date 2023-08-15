# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import json
import requests
from streamlit_lottie import st_lottie
import pickle
import pandas as pd
from streamlit_option_menu import option_menu
import streamlit as st
from PIL import Image
import sklearn
import numpy as np
from Function import predict_rf


heart_disease_model=pd.read_pickle('/Users/ronishkhatiwada/Downloads/Heart Disease Prediction/model_tree.pickle')

#heart_disease_model=pickle.load(open('//Users//ronishkhatiwada//Downloads//Heart Disease Prediction//rf_lib.sav'))

# sidebar for navigation
with st.sidebar:
    
    selected = option_menu('Main Menu',
                          
                          ['HomePage',
                           'Description',
                           'Diagnose',
                           'Tips',
                           'Data'],
                          icons=['house','info-circle-fill','heart','activity','graph-up-arrow'],
                          default_index=0)
      
  
# HomePage
if (selected=='HomePage'):
    # page title
    st.title('Heart Disease Prediction System')
    #st.image('/Users/ronishkhatiwada/Downloads/Heart Disease Prediction/heart.jpg')
    def load_lottiefile(filepath: str):
     with open(filepath, "r") as f:
        return json.load(f)
    
    lottie_heart=load_lottiefile("/Users/ronishkhatiwada/Downloads/Heart Disease Prediction/lottie/heart.json")
    st_lottie(
        lottie_heart,
        speed=1,
        reverse=False,
        loop=True,
        quality="high",
        height=300,
        width=500,
        key=None,
)
    #st.markdown('Welcome to our Heart Disease Prediction System, a cutting-edge tool designed to help you assess your risk of developing heart disease.') 
    
    
    st.markdown('Heart Disease Prediction System is a web app that uses machine learning to predict the risk of heart disease based on input like age, gender, blood pressure, and cholesterol. It can diagnose the disease early, enabling prompt management')
    st.markdown('It can also assist physicians in making informed decisions about the best course of treatment for their patients. The system is user-friendly and can be easily accessed by patients and healthcare professionals through  web in computers. Its accuracy and reliability can help reduce healthcare costs and improve the overall quality of patient care.')
                
    #st.markdown('Using our user-friendly interface, you can input your relevant health data and receive an instant result along with recommendations for lifestyle changes and preventative measures. Our system is based on the latest scientific research and medical guidelines, ensuring that you receive accurate and up-to-date information.')
    #st.markdown('Whether you are looking to take proactive steps to protect your heart health or are already experiencing symptoms, our Heart Disease Prediction System can provide valuable insights into your risk profile and help you make informed decisions about your health. Take the first step towards a healthier heart today by using our system to assess your risk of heart disease.')

# Description Page
    
if(selected=='Description'):
    st.title('Description')
    
    def load_lottiefile(filepath: str):
     with open(filepath, "r") as f:
        return json.load(f)
    
    lottie_description=load_lottiefile("/Users/ronishkhatiwada/Downloads/Heart Disease Prediction/lottie/description.json")
    st_lottie(
        lottie_description,
        speed=1,
        reverse=False,
        loop=True,
        quality="high",
        height=200,
        width=200,
        key=None,
)
         
    
    st.markdown(' Age(age): Age is a critical risk factor for heart attacks, as the risk doubles with increasing age. Fatty streaks that signify coronary artery disease begin to form in adults. Patients aged 65 or older account for over 80% of heart attack cases due to coronary heart disease')
    
    st.markdown(' Sex(sex): Men under 50 have a higher risk of heart attack than women, but after menopause, the risk becomes comparable. Women with diabetes have an increased risk of heart attack. (Note: 1 = male, 0 = female)')
    
    st.markdown(' Chest Pain(cp): Angina occurs when the heart muscle is deprived of oxygenated blood, causing chest pressure and discomfort in the shoulder, jaw, back, or neck. Hand pain may also occur. Chest pain is classified using a numbering system, and can be accompanied by indigestion-like symptoms.')
    st.markdown('a) 1 represents Atypical Angina')
    st.markdown('Atypical angina is a chest pain type that lacks typical symptoms such as chest pressure or discomfort but may cause a burning sensation, shortness of breath, fatigue, or nausea. Diagnosing atypical angina is challenging due to its symptoms resembling those of other conditions like acid reflux or anxiety.')
    st.markdown('b) 2 represents Typical Angina')
    st.markdown('Typical angina causes tightness, squeezing or weight in the chest, and can also affect the arms, neck, jaw, shoulder, and back. Its triggered by physical activity or emotional stress and eased by rest or medication. It results from reduced blood supply to the heart muscle due to coronary artery constriction.')
    st.markdown('c) 3 represents Non-anginal pain ')
    st.markdown('Non-anginal chest pain is discomfort in the chest not caused by decreased blood flow to the heart. It can stem from factors like heartburn, anxiety, strained muscles, or inflammation. The lungs, esophagus, or ribs may also be the source')
    st.markdown('d) 4 represents Asymptomatic ')
    st.markdown('"Asymptomatic chest pain" is chest discomfort without other symptoms like nausea or shortness of breath. Its often found by chance during medical exams, and its cause is not always known. It is important to consult a medical expert to assess potential underlying problems and determine the best course of action')
   
    
    st.markdown(' Resting Blood Pressure(tresbps): Resting blood pressure is a measurement taken while a person is relaxed and seated, without any physical activity or stimulant consumption for at least 5 minutes. It is measured using a blood pressure cuff and is a crucial indicator of cardiovascular health.')
    
    st.markdown(' Serum cholestoral(chol): Serum cholesterol is a blood-borne fatty substance that helps build cell membranes, produce hormones, and aid digestion. High levels of cholesterol can increase the risk of heart disease and stroke, while lower levels are associated with better cardiovascular health')
    
    st.markdown(' Fasting blood sugar(fbs): Fasting blood sugar (FBS) is the glucose amount in blood after an overnight fast of 8-12 hours. Itis measured in the morning before eating or drinking anything other than water. FBS levels help diagnose and monitor diabetes and prediabetes. High levels suggest impaired glucose tolerance or insulin resistance, which can lead to type 2 diabetes and other health complications [1 represents fbs>120 mg/dl and 0 represents fbs<120 mg/dl]')
    
    st.markdown(' Resting electrocardiographic results(restecg): For medium to high risk of heart attack, the present scenario is not sufficient to understand the screening disadvantages. For those having less risk of disease, the screening harmful effects including a rash or irritation on skin can balance up with exercise.')
    st.markdown('a) 1 represents Left vent hyper')
    st.markdown('b) 2 represents Normal')
    st.markdown('c) 3 represents St wave abnormality')
    
    st.markdown(' Maximum heart rate achieved(thalach): The increase in the heart rate with the enhanced risk of heart disease is being parallelized with risk increment with blood pressure enhancement. It is proven in research that if the heart rate increases by 10 bpm, then the chances of cardiac death increase by 20%. This is also the same with the enhancement in the blood pressure of 10 mm Hg.')
    
    st.markdown(' Exercise induced angina(exang): The discomfort from Angina which is an Exercise-induced makes the person feel gripped, squeezed and tight which can carry from mild to serious. The pain is usually felt in the chest’s center and it can spread up in the shoulders, back, jaw, arm or neck. Angina plays a crucial role in identifying coronary disease which makes it worthwhile to consider it a separate category for analysis.[1 represents Yes and 0 represents No]')
    
    st.markdown(' ST depression induced by exercise relative to rest(oldpeak): ST depression induced by exercise relative to rest is a type of electrocardiogram (ECG) finding that can occur during exercise stress testing. When a person exercises, their heart works harder to meet the increased demand for oxygen and nutrients. In some cases, if the blood supply to the heart is reduced or blocked, this can result in a decrease in oxygen delivery to the heart muscle, which can be seen on the ECG as ST segment depression.')
    
    
# Heart Disease Prediction Page
if (selected == 'Diagnose'):
    
    # page title
    st.title('Enter details to Check')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('#Age:')
        # st.text("")
        # st.text("")
        # st.text("")
        # st.text("")
        # st.text("")
        # st.text("")
        age = st.text_input('Enter Age')
        
    with col2:
        st.markdown('#Sex:')
        # st.text("")
        # st.text("")
        # st.text("")
        # st.text("")
        # st.text("")
        # st.text("")
        sex = st.selectbox('Select [0] for female or [1] for male','01')
        
    with col3:
        st.markdown('#ChestPain Type:')
        st.markdown('1.Typical angina')
        st.markdown('2.Atypical angina')
        st.markdown('3.Non-anginal pain')
        st.markdown('4.Asymptomatic')
        st.text("")
        cp = st.selectbox('(CP) choose between 1-4','1234')
        
    with col1:
        st.text("")
        st.text("")
        st.text("")
        st.text("")
        st.markdown('#Resting Blood Pressure')
        trestbps = st.text_input('trestbps')
        
        
    with col2:
        st.text("")
        st.text("")
        st.text("")
        st.text("")
        st.markdown('#Serum Cholestoral in mg/dl ')
        chol = st.text_input('chol')
        
    with col3:
       
        st.text("")
        st.markdown('#Fasting Blood Sugar > 120 mg/dl')
        fbs = st.selectbox('(fbs) Select 1 for True or 0 for False','01')
        
    with col1:
        st.markdown("#Resting Electrocardiographic results")
        st.markdown('0. Normal')
        st.markdown('1. ST-T wave abnormality')
        st.markdown('2. Probable or definite left ventricular hypertrophy by Estes criteria')
        restecg = st.selectbox('(restecg) Choose between 0-2','012')
        
    with col2:
        st.markdown("#Maximum Heart Rate achieved")
        # st.text("")
        # st.text("")
        # st.text("")
        # st.text("")
        # st.text("")
        # st.text("")
        # st.text("")
        # st.text("")
        # st.text("")
        thalach = st.text_input(' (thalach)')
        
    with col3:
        # st.text("")
        # st.text("")
        st.text("")
        st.text("")
        st.text("")
        st.text("")
        st.markdown('#Exercise Induced Angina')
        # st.text("")
        # st.text("")
        # st.text("")
        # st.text("")
        # st.text("")
        # st.text("")
        # st.text("")
        # st.text("")
        # st.text("")
        exang = st.selectbox('(exang) Select 1 for Yes or 0 for No','01')
        
    with col1:
        st.markdown('#ST depression induced by exercise relative to test ')
        # st.text("")
        # st.text("")
        # st.text("")
        oldpeak = st.text_input('(oldpeak)')
        
   # with col2:
       # slope = st.text_input('Slope of the peak exercise ST segment')
        
    #with col3:
       # ca = st.text_input('Major vessels colored by flourosopy')
        
   # with col1:
        #thal = st.text_input('thal: 1= normal; 2 = fixed defect; 3 = reversable defect')
        
        
     
     
    # code for Prediction
    heart_diagnosis = ''
    
    # creating a button for Prediction
    if st.button('Check'):
        try:
            test = {
              'age':int(age),
              'sex':int(sex),
              'cp':int(cp),
              'trestbps':int(trestbps),
              'chol':int(chol),
              'fbs':int(fbs),
              'restecg':int(restecg),
              'thalach':int(thalach),
              'exang':int(exang),
              'oldpeak':float(oldpeak)
             }
          
            test = pd.DataFrame(test, index = [0])
          
            heart_prediction = predict_rf(heart_disease_model,test)                        
          
            if (heart_prediction[0] == 1):
                heart_diagnosis = 'You have Heart Disease.Consult Doctor!! (Visit our Page for tips to maintain good heart.)'
            else:
                heart_diagnosis = 'You do not have any heart disease'
            st.success(heart_diagnosis)
            
        except:
            st.markdown("Please provide values for all input fields.")
    
# Tips for Healthy Heart Page
if (selected == "Tips"):
    col1,  col3 = st.columns(2)
    with col3: 
        
        def load_lottiefile(filepath: str):
          with open(filepath, "r") as f:
           return json.load(f)
        lottie_care=load_lottiefile("/Users/ronishkhatiwada/Downloads/Heart Disease Prediction/lottie/care.json")
        st_lottie(
        lottie_care,
        speed=1,
        reverse=False,
        loop=True,
        quality="high",
        height=200,
        width=200,
        key=None,
)
    
    # page title
    with col1:
        
        st.subheader("Tips for Healthy Heart")
        st.markdown('1. Maintain a healthy diet')
        st.markdown('2. Exercise Regularly')
        st.markdown('3. No Smoking and Alcohol')  
        st.markdown('4. Manage Stress')
        st.markdown('6. Manage Cholestrol Levels')
        st.markdown('7. Avoid Processed and Junk Foods')
        st.markdown('8. Stay Hydrated')
        st.markdown('9. Reduce Salt Intake')
        st.markdown('10. Monitor Caffiene Intake')
        st.markdown('11. Eat Fiber-rich Foods')
        st.markdown('12. Avoid Sitting for too long')
        st.markdown('13. Stay up-to-date on Medical Checkups')
        st.markdown('14. Reduce Exposure to air')
        st.markdown('15. Practice good Oral Hygiene')

#About Data         
if (selected=='Data'):
    st.title('About Data')
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text('Original dataset through which we trained our Model')
    st.image('/Users/ronishkhatiwada/Downloads/Heart Disease Prediction/Data Figures/Data.JPG')
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text('Heart Disease Patients in DataSet')
    st.image('/Users/ronishkhatiwada/Downloads/Heart Disease Prediction/Data Figures/Heart Disease Data.JPG')
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text('Distribution of Gender in Dataset')
    st.image('/Users/ronishkhatiwada/Downloads/Heart Disease Prediction/Data Figures/Gender Data.JPG')
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text('Outliers in Dataset')
    st.image('/Users/ronishkhatiwada/Downloads/Heart Disease Prediction/Data Figures/Outliers .JPG') 
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text('Processed and Cleaned Data')
    st.image('/Users/ronishkhatiwada/Downloads/Heart Disease Prediction/Data Figures/Cleaned Data.JPG')
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text('Comparison of different Algorithms using Library Implementation with result')
    st.image('/Users/ronishkhatiwada/Downloads/Heart Disease Prediction/Data Figures/comparison.JPG')    
            
            
            
            
            
            
            
            
            
            
            
            
            