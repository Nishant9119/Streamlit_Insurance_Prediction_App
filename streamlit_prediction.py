import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.svm import SVC
import pickle
from sklearn.cluster import KMeans
import pickle
def get_file():
    with open("C:/Users/Nikhil Sharma/PycharmProjects/StreamLit_App/Nishant_Pycharm_Pickel.pkl","rb") as file:
        data = pickle.load(file)
    return data
data= get_file()
regressor = data['model']
print(regressor)
def predict():
    st.title("Fraud Prediction Page")
    st.write("Enter the required values to get the prediction")
    deductible_amount = (300,400,500,700)
    driver_rating =(1,2,3,4)
    Accident_Area = ("Urban","Rural")
    Gender = ("Male","Female")
    Marital_Status = ("Single","Married","Widow","Divorced")
    Fault = ("Policy Holder","Third Party")
    Policy_Type = ('Sport-Liability', 'Sport-Collision', 'Sedan-Liability',
       'Utility-All Perils', 'Sedan-All Perils', 'Sedan-Collision',
       'Utility-Collision', 'Utility-Liability', 'Sport-All Perils')
    Vehicle_Category= ('Sport', 'Utility', 'Sedan')
    Vehicle_Price = ('more than 69,000', '20,000 to 29,000', '30,000 to 39,000',
       'less than 20,000', '40,000 to 59,000', '60,000 to 69,000')
    Days_Policy_Accident = ('more than 30', '15 to 30', 'none', '1 to 7', '8 to 15')
    Days_POlicy_Claim = ('more than 30', '15 to 30', '8 to 15')
    Past_Number_of_Claims = ('none', '1', '2 to 4', 'more than 4')
    Age_of_Policy_Holder = ('26 to 30', '31 to 35', '41 to 50', '51 to 65', '21 to 25',
       '36 to 40', '16 to 17', 'over 65', '18 to 20')
    Police_Report_Filed = ('No', 'Yes')
    witness_Present = ('No', 'Yes')
    Agent_Type =('External', 'Internal')
    Number_of_Suppliments = (0, 'more than 5', '3 to 5', '1 to 2')
    Address_Change_Claim = ('1 year', 'no change', '4 to 8 years', '2 to 3 years',
       'under 6 months')
    Number_of_Cars = ('3 to 4', '1 vehicle', '2 vehicles', '5 to 8', 'more than 8')
    Base_Policy =('Liability', 'Collision', 'All Perils')
    # Values which use will select
    user_deductibles = st.number_input("Enter Deductible Amount",0)
    user_Driver_Rating = st.slider("Select Driver Rating",1,5)
    user_Accident_Area = st.selectbox("Select Accident Area",Accident_Area)
    user_Gender = st.selectbox("Select Gender",Gender)
    user_Marital_Status = st.selectbox("Select Marital Status",Marital_Status)
    user_Fault = st.selectbox("Select Fault Type",Fault)
    user_Policy_Type = st.selectbox("Select Policy Type",Policy_Type)
    user_Vehicle_Category = st.selectbox("Select Vehicle Category",Vehicle_Category)
    user_Vehicle_Price = st.selectbox("Select Vehicle Price",Vehicle_Price)
    user_Days_Policy_Accident = st.selectbox("Days Policy Accident",Days_Policy_Accident)
    user_Days_POlicy_Claim = st.selectbox("Days Policy Claim",Days_POlicy_Claim)
    user_past_claims= st.number_input("Enter Number of Past Claims",1)
    user_Age_of_Policy_Holder = st.number_input("Enter Age of Policy Holder")
    user_Police_Report_Filed = st.radio("Police Report Filed ?",Police_Report_Filed)
    user_Witness_Present= st.radio("Witness Presence ?",witness_Present)
    user_Agent_Type = st.radio("Agent Type",Agent_Type)
    user_Number_of_Suppliments  =st.selectbox("Number of Suppliments",Number_of_Suppliments)
    user_Address_Change_Claim = st.selectbox("Address Change Claim",Address_Change_Claim)
    user_Number_of_Cars = st.number_input("Enter Number of Cars",0)
    user_Base_Policy = st.selectbox("Base Policy",Base_Policy)
    ok = st.button("Predict")
    if ok:
        a = user_deductibles
        b = user_Driver_Rating
        c = 1 if user_Accident_Area=='Urban' else 0
        d = 1 if user_Gender=='Male' else 0
        e = 1 if user_Marital_Status=="Married" else 0
        f = 1 if user_Marital_Status=="Single" else 0
        g = 1 if user_Marital_Status=="Widow" else 0
        h = 1 if user_Fault=="Third Party" else 0
        i = 1 if user_Policy_Type == 'Sedan-Collision' else 0
        j = 1 if user_Policy_Type =='Sedan-Liability' else 0
        k = 1 if user_Policy_Type =='Sport-All Perils' else 0
        l = 1 if user_Policy_Type =='Sport-Collision' else 0
        m = 1 if user_Policy_Type =='Sport-Liability' else 0
        n = 1 if user_Policy_Type =='Utility-All Perils' else 0
        o = 1 if user_Policy_Type =='Utility-Collision' else 0
        p = 1 if user_Policy_Type =='Utility-Liability' else 0
        q = 1 if user_Vehicle_Category =='Sport' else 0
        r = 1 if user_Vehicle_Category == 'Utility' else 0
        s = 1 if user_Vehicle_Price=='30,000 to 39,000' else 0
        t = 1 if user_Vehicle_Price == '40,000 to 59,000' else 0
        u = 1 if user_Vehicle_Price ==  '60,000 to 69,000' else 0
        v = 1 if user_Vehicle_Price == 'less than 20,000' else 0
        w = 1 if user_Vehicle_Price == 'more than 69,000' else 0
        x = 1 if user_Days_Policy_Accident ==  '15 to 30' else 0
        y = 1 if user_Days_Policy_Accident == '8 to 15' else 0
        z = 1 if user_Days_Policy_Accident == 'more than 30' else 0
        ab = 1 if user_Days_Policy_Accident == "none" else 0
        ac = 1 if user_Days_POlicy_Claim == '8 to 15' else 0
        ad = 1 if user_Days_POlicy_Claim== 'more than 30' else 0
        ae = 1 if 2<=user_past_claims<=4  else 0
        af  = 1 if user_past_claims>=4  else 0
        ag = 1 if user_past_claims == 0 else 0
        ah = 1 if 18<=user_Age_of_Policy_Holder<=20 else 0
        ai = 1 if 21<=user_Age_of_Policy_Holder<=25 else 0
        aj = 1 if 26<=user_Age_of_Policy_Holder<=30 else 0
        ak = 1 if 31<= user_Age_of_Policy_Holder<=35 else 0
        al = 1 if 36<= user_Age_of_Policy_Holder<=40 else 0
        am = 1 if 41<= user_Age_of_Policy_Holder<50 else 0
        an = 1 if 51<= user_Age_of_Policy_Holder<= 65 else 0
        ao = 1 if user_Age_of_Policy_Holder>=66 else 0
        ap = 1 if user_Police_Report_Filed=='Yes' else 0
        aq = 1 if user_Witness_Present =='Yes' else 0
        ar = 1 if user_Agent_Type=='Internal' else 0
        as1 = 1 if user_Number_of_Suppliments =='1 to 2' else 0
        at = 1 if user_Number_of_Suppliments == '3 to 5' else 0
        au = 1 if user_Number_of_Suppliments == 'more than 5' else 0
        av = 1 if user_Address_Change_Claim =='2 to 3 years' else 0
        aw = 1 if user_Address_Change_Claim == '4 to 8 years' else 0
        ax = 1 if user_Address_Change_Claim == 'no change' else 0
        ay = 1 if user_Address_Change_Claim == 'under 6 months' else 0
        az = 1 if user_Number_of_Cars ==2 else 0
        aa1 = 1 if 3<=user_Number_of_Cars <=4 else 0
        ab1 = 1 if 5<= user_Number_of_Cars<= 8 else 0
        ac1 = 1 if user_Number_of_Cars>9 else 0
        ad1 = 1 if user_Base_Policy=='Collision' else 0
        ae1 = 1 if user_Base_Policy =='Liability' else 0
        x = [[a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,ab,ac,ad,ae,af,ag,ah,ai,aj,ak,al,am,an,ao,ap,aq,ar,as1,at,au,av,aw,ax,ay,az,aa1,ab1,ac1,ad1,ae1]]
        fraud_prediction = regressor.predict(x)
        if fraud_prediction==1:
            st.subheader(f"The Person with given inputs is potentially a fraud.")
        if fraud_prediction ==0 :
            st.subheader(f"The Person with given inputs is potentially not a fraud.")











