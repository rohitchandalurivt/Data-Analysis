import numpy as np
import pandas as pd
import os
dataset = pd.read_csv('time_data.csv')
X = dataset.iloc[:,1:7].values
Y = dataset.iloc[:,-1].values

# This code is to run neural network model on the data set

unique_values = sorted(set(Y))
dict_y ={k: v for v, k in enumerate(unique_values)}
Y_new = np.array(Y)
for i in range(len((Y))):
    Y_new[i] =dict_y[Y[i]]


os.environ['KMP_DUPLICATE_LIB_OK']='True'

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y_new, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, TerminateOnNaN

classifier = Sequential()
classifier.add(Dense(activation="relu", input_dim=len(X[0]), units=128, kernel_initializer="uniform"))

classifier.add(Dense(kernel_initializer="uniform", activation="relu", units=512))

classifier.add(Dense(kernel_initializer="uniform", activation="relu", units=512))

classifier.add(Dense(kernel_initializer="uniform", activation="relu", units=512))

classifier.add(Dense(activation="softmax", units=len(set(Y)), kernel_initializer="uniform"))
classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='sparse_categorical_crossentropy', patience=100)
classifier.fit(X_train, y_train, batch_size = 1000, epochs = 4000,callbacks=[early_stopping, TerminateOnNaN()])
classifier.save('/home/riteshbansal/acm/model_trained1x1_new_2.h5')


##Prediction
y_pred = classifier.predict(X_test)
res = np.argmax(y_pred, axis=1)

##Result calculate
count_correct = np.count_nonzero(y_test==res)
acc = count_correct/len(res)
print(acc)

#Score
best_n = np.argsort(y_pred)[:, -3:][:, ::-1]
score = 0.0
for i in range(0, len(best_n) - 1):
    for j in range(0, len(best_n[0]) - 1):
        if (y_test[i] == best_n[i][j]):
            if (j == 0):
                score = score + 1
            elif (j == 1):
                score = score + 0.61
            elif (j == 2):
                score = score + (1 / 3)
score = score / len(best_n)