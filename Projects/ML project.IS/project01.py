import numpy as np # type: ignore
import pandas as pd # type: ignore
import matplotlib.cm as cm # type: ignore
import matplotlib.pyplot as plt # type: ignore
#importing data
data_train = pd.read_csv("train - Copy.csv")
data_train.head()

data_train.shape
#(14999,9)

#visualization
   #individual plots
plt.hist(data_train["category"])
plt.show()
plt.hist(data_train["adview"])
plt.show()

   #remoove videos with adview greater than 200000 as outlier

data_train=data_train[data_train["adview"]<2000000]

import seaborn as sns # type: ignore
 

f,ax=plt.subplots(figsize=(10,8))
corr=data_train.corr()
sns.heatmap(corr,mask=np.zeros_like(corr,dtype=np.bool),cmap=sns.diverging_palettee(220,10,as_cmap=True),square=True,ax=ax,annot=True)
plt.show()

data_train=data_train[data_train.views!='F']
data_train=data_train[data_train.likes!='F']
data_train=data_train[data_train.dislikes!='F']
data_train=data_train[data_train.comments!='F']

data_train.head()

category={'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7,'H':8}
data_train["category"]=data_train["category"].map(category)
data_train.head()

data_train["views"]=pd.to_numeric(data_train["views"])
data_train["comments"]=pd.to_numeric(data_train["comments"])
data_train["likes"]=pd.to_numeric(data_train["likes"])
data_train["dislikes"]=pd.to_numeric(data_train["dislikes"])
data_train["adview"]=pd.to_numeric(data_train["adviiew"])


column_vidid=data_train['vidid']

#encoding features like category,duration,vidid

from sklearn.preprocessing import LabelEncoder # type: ignore
data_train['duration']=LabelEncoder().fit_transform(data_train['duration']) 
data_train['vidid']=LabelEncoder().fit_transform(data_train['vidid'])
data_train['published']=LabelEncoder().fit_transform(data_train['published'])

data_train.head()

#convert time_in_sec for duration

import datetime
import time

def checki(x):
    y=x[2:]
    h=''
    m=''
    s=''
    mm=''
    P = ['H','M','S']
    for i in y:
        if i not in P:
            mm+=i
        else:
            if(i=="H"):
                h=mm
                mm=''
            elif(i=="M"):
                m=mm
                mm=''
            else:
                s=mm 
                mm=''
    if(h==''):
        h='00'
    if(m==''):
        m='00'
    if(s==''):
        s='00'
    bp = h+':'+m+':'+s
    return bp
train=pd.read_csv("train.csv")
mp=pd.read_csv(path+"train.csv")["duration"]
time=mp.apply(checki)

def func_sec(time_string):
    h,m,s=time_string.split(':')
    return int(h)*3600+int(m)*60+int(s)

time1=time.apply(func_sec)

data_train["duration"]=time1
data_train.head()


#splitdata
Y_train=pd.DataFrame(data = data_train.iloc[:, 1].values,columns=['target'])
data_train=data_train.drop(["adview"],axis=1)
data_train=data_train.drop(["vidid"],axis=1)
data_train.head()

from sklearn.model_selection import train_test_split # type: ignore
X_train,X_test,Y_train,y_test=train_test_split(data_train,Y_train,test_size=0.2,random_state=42)

X_train.shape

#normalise data
from sklearn.preprocessing import MinMaxScaler # type: ignore
scaler=MinMaxScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)   

X_train.mean()

# #evaluation metrics
from sklearn import metrics
def print_error(X_test,y_test,model_name):
    prediction=model_name.predict(X_test)
    print('Mean Absolute Erroe:',metrics.mean_absolute_error(y_test,prediction))
    print('Mean Squared Error:',metrics.mean_squared_error(y_test,prediction))
    print('Root Mean Squared Error:',np.sqrt(metrics.mean_squared_error(y_test,prediction)))


#Linear regression
from sklearn import linear_model
linear_regression = linear_model.LinearRegression()
linear_regression.fit(X_train,Y_train)
print_error(X_test,y_test,linear_regression)


#Decision tree
from sklearn.tree import DecisionTreeRegressor
decision_tree=DecisionTreeRegressor()
decision_tree.fit(X_train,Y_train)
print_error(X_test,y_test,decision_tree)

#Random Forest Regresssor
from sklearn.ensemble import RandomForestRegressor
n_estimators=200
max_depth=25
min_samples_split=15
min_samples_leaf=2
random_forest=RandomForestRegressor(n_estimators=n_estimators,max_depth=max_depth,min_samples_split=min_samples_split)
random_forest.fit(X_train,Y_train)
print_error(X_test,y_test,random_forest) 


#artificial neural network
import keras
from keras.layers import Dense
ann = keras.models.Sequential( [
                                 Dense(6,activation="relu",input_shape=X_train.shape[1:]),
                                 Dense(6,activation="relu"),
                                 Dense(1)
                                ])
optimizer=keras.optimizers.Adam()
loss=keras.losses.mean_squared_error
ann.compile(optimizer=optimizer,loss=loss,metrics=["mean_squared_error"])

history=ann.fit(X_train,Y_train,epochs=100)
ann.summary()
print_error(X_test,y_test,ann) 

#saving scikitlearn models
import joblib

joblib.dump(decision_tree,"decisiontree_youtubeadview.pkl")

#saving keras artificial meural network model
ann.save("ann_youtubeadview.h5")
