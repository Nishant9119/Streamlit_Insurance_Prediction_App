import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score, confusion_matrix
import pickle
df = pd.read_csv("C:/Users/Nikhil Sharma/Downloads/final_smoted_claims_data.csv")
x = df.drop(["FraudFound_Yes",'X'],axis= 1)
y = df['FraudFound_Yes']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25)
rf = RandomForestClassifier(n_estimators = 100)
rf.fit(x_train,y_train)
predicted = rf.predict(x_test)
data = {"model":rf}
print(df.columns)
with open("Nishant_Pycharm_Pickel.pkl","wb") as file:
    pickle.dump(data,file)
