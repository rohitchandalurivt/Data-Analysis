import csv
import numpy as np
from sklearn.svm import SVC
from sklearn import preprocessing

# the code is to model data on SVM and test the testing data set on that model using SVM
# The code took more than a day to run the complete model.

with open("time_data.csv", "rb") as infile:
    re1 = csv.reader(infile)
    result=[]
    for row in re1:
        result.append(row[8])

    trainclass = result[:251900]
    testclass = result[251901:279953]


with open("time_data.csv", "rb") as infile:
    re = csv.reader(infile)
    coords = [(float(d[1]), float(d[2]), float(d[3]), float(d[4]), float(d[5])) for d in re if len(d) > 0]
    train = coords[:251900]
    test = coords[251901:279953]


clf = SVC(kernel='rbf', probability=True, cache_size=10000)
clf.fit(preprocessing.scale(train),trainclass)

Accu = clf.score(preprocessing.scale(test),testclass)

print "rbf accuracy"

print Accu

ff = clf.predict_log_proba(preprocessing.scale(test))

classes =  clf.classes_

best_n = np.argsort(ff)[:, -3:][:, ::-1]


score = 0.0

for i in range(0, 28051):
    for j in range(0,2):
        if(testclass[i] == classes[best_n[i][j]]):
            if(j==0):
                score = score + 1
            elif(j==1):
                score = score + 0.61
            elif(j==2):
                score = score + (1/3)


score = score/28052
print "Final score in rbf"
print score