import pandas as pd 
import numpy as np 
 
from sklearn.model_selection import train_test_split 
import warnings 
warnings.filterwarnings("ignore") 
import pickle 
 
te2=pd.read_csv("telecom.csv") 
from sklearn.preprocessing import LabelEncoder 
le=LabelEncoder() 
te2['gender']=le.fit_transform(te2['gender']) 
te2['PhoneService']=le.fit_transform(te2['PhoneService']) 
te2['Partner']=le.fit_transform(te2['Partner']) 
te2['OnlineSecurity']=le.fit_transform(te2['OnlineSecurity']) 
te2['OnlineBackup']=le.fit_transform(te2['OnlineBackup']) 
te2['PaperlessBilling']=le.fit_transform(te2['PaperlessBilling']) 
te2['Churn']=le.fit_transform(te2['Churn']) 
te2['InternetService']=le.fit_transform(te2['InternetService']) 
te2['Contract']=le.fit_transform(te2['Contract']) 
te2['PaymentMethod']=le.fit_transform(te2['PaymentMethod'])  
te2['TechSupport']=le.fit_transform(te2['TechSupport']) 
X=te2.drop(['customerID','MonthlyCharges','Partner','SeniorCitizen','Dependents','MultipleLines','DeviceProtection','StreamingTV','StreamingMovies','TotalCharges','Churn'],axis=1)
Y=te2['Churn'] 
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=
 42) 
from sklearn.ensemble import RandomForestClassifier 
rf=RandomForestClassifier(n_estimators=1000) 
rf.fit(X_train,Y_train) 
pickle.dump(rf,open('model.pkl','wb'))