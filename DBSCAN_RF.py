import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import csv
from sklearn.cluster import DBSCAN


# this code is to combine DBSCAN and Random Forests, as suggested by professor aditya we tried to decrease the
# number of classes by classifying those classes as cluster labels and predicting those cluster labels depending
# on the random forest classifier.

dataset = pd.read_csv('time_data.csv')
X = dataset.iloc[:,[1,2]].values


dbscan = DBSCAN(eps=0.016,min_samples=300).fit(X)
labels = dbscan.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

# Black removed and is used for noise instead.
temp=labels
values, counts = np.unique(temp, return_counts=True)
counts_sort = sorted(counts,reverse=True)
unique_labels_2=[];
counts= counts.tolist();


with open("time_data.csv", "rb") as infile:
    re1 = csv.reader(infile)
    result=[]
    for row in re1:
        result.append(row[8])

    trainclass = result[:251900]
    testclass = result[251901:279950]


with open("time_data.csv", "rb") as infile:
    re = csv.reader(infile)
    coords = [(float(d[1]), float(d[2]), float(d[3]), float(d[4]), float(d[5])) for d in re if len(d) > 0]
    train = coords[:251900]
    test = coords[251901:279950]

print "Doe splitting data into test and train data"

clf = RandomForestClassifier(n_estimators=500,max_features="log2", min_samples_split=3, min_samples_leaf=2)

clf.fit(train,trainclass)

print "Done training"
score = clf.score(test,testclass)
print "Done Testing"
print "rf accuracy"
print score