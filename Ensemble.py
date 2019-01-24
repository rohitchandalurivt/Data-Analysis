import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# this file is to run ensembeled model on sample data set and get results

# data is stored in df variable
df = pd.read_csv("time_data1.csv")
# sc is to store scores for each grid 
sc=np.zeros((1,1))

# for grids from 0 to 10
for i in xrange(1,2):
    for j in xrange(1,2):
        # dividing the data as per the grid
        df_temp=df[(df["x"] >= float(i-1))&(df["x"] <= float(i))&(df["y"] >= float(j-1))&(df["y"] <= float(j))]
        X = df_temp.iloc[:,[1,2,3,4,5]].values
        labels = df_temp.iloc[:,[8]].values
        labels = [str(k) for k in labels]
        
        X_train, X_test, y_train, y_test=train_test_split(X,labels, test_size=0.10)
        print " Train/Test Dataset "+str(i)+str(j)+" Loaded!"
        # naive bayes classifier
        clf_pf = GaussianNB()
        clf_pf.fit(X_train, y_train)
        
        # predicting placeids with naive bayes
        ff_nb = clf_pf.predict_proba(X_test)
        print "Bayesian Score predicted!"
        
        norm_data = MinMaxScaler()
        X_train_norm= norm_data.fit_transform(X_train)
        X_test_norm= norm_data.fit_transform(X_test)
        
        # KNN classifier with a weight of 1/d2
        classifier1 = KNeighborsClassifier(n_neighbors=50,weights=lambda x: x ** -2,metric='manhattan')
        # KNN classifier with a weight of 1/d
        classifier2 = KNeighborsClassifier(n_neighbors=15,weights='distance',metric='manhattan')
        
        classifier1.fit(X_train_norm, y_train)
        
        # predicting classes with KNN 1/d2 
        ff_knn1 = classifier1.predict_proba(X_test_norm)
        print "knn1 predicted!"
        
        classifier2.fit(X_train_norm, y_train)
        
        
        # predicting classes with KNN 1/d2
        ff_knn2 = classifier2.predict_proba(X_test_norm)
        print "knn2 predicted!"
        
        classes =  clf_pf.classes_ 
        
        ff=(ff_knn1+ff_knn2+ff_nb)/3
        
        best_n = np.argsort(ff)[:, -3:][:, ::-1]
        
        score = 0.0
        for l in range(0, len(best_n)-1):
            for m in range(0,len(best_n[0])-1):
                if(y_test[l] == classes[best_n[l][m]]):
                    if(m==0):
                        score = score + 1
                    elif(m==1):
                        score = score + 0.61
                    elif(m==2):
                        score = score + (1/3)
        
        
        score = score/len(best_n)
        sc[i-1,j-1]=score
        print("the score of grid "+str(i)+str(j)+" is "+str(score))
        
            