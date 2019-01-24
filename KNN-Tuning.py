import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np

# Code to find the optimal K value for KNN

# loading data into variable dataset

dataset = pd.read_csv('time_data.csv')
X = dataset.iloc[:,[1,2,3,4,5]].values
norm_data = MinMaxScaler()
X= norm_data.fit_transform(X)
labels = dataset.iloc[:,[8]].values
labels = [str(i) for i in labels]


X_train, X_test, y_train, y_test=train_test_split(X,labels, test_size=0.1)
temp=[]
for k in xrange(10,100):
    classifier = KNeighborsClassifier(n_neighbors=k,weights=lambda x: x ** -2,metric='manhattan')
    classifier.fit(X_train, y_train)
    classes =  classifier.classes_
    ff = classifier.predict_proba(X_test)
    best_n = np.argsort(ff)[:, -3:][:, ::-1]
    score = 0.0
    for i in range(0, len(best_n)-1):
        for j in range(0,len(best_n[0])-1):
            if(y_test[i] == classes[best_n[i][j]]):
                if(j==0):
                    score = score + 1
                elif(j==1):
                    score = score + 0.61
                elif(j==2):
                    score = score + (1/3)
    score = score/len(best_n)
    temp.append(score)
    print "computing kNN for k="+str(k)+" with score="+str(score)

    
maxk=1+np.argmax(np.asarray(temp))