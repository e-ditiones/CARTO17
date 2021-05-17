import os, os.path
import pickle
import tensorflow
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split,learning_curve, GridSearchCV
from sklearn import datasets, preprocessing, metrics
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import numpy as np
from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from keras import backend as K

###############
# PREPARATION #
###############

#IMPORT
#import data
csv_file=os.path.join('listPlaces', 'traCom_listPlaces_plus.tsv')
data = pd.read_csv(csv_file, delimiter='\t')
#split coordinates into two columns to separate long et lat
data[['long','lat']] = data.coord.str.split(",",expand=True,)
data['lat'] = pd.to_numeric(data['lat'])
data['long'] = pd.to_numeric(data['long'])
# It is a good idea to check and make sure the data is loaded as expected.
print(data.head(5))

#GET FEATURES
# features
feature_1 = data['long']
feature_2 = data['lat']
# labels
y = data['subgenre']

#CREATE SPLITS
#creating labelEncoder
le = preprocessing.LabelEncoder()
# converting string labels into numbers
label=le.fit_transform(y)
features=list(zip(feature_1,feature_2))
X_train, X_test, y_train, y_test = train_test_split(features, label,
	test_size=0.2,# 80% training and 20% test
	random_state=1) # reproducibility

#path for saving metrics
save_report = os.path.join('models','model_best_classification.txt')

##############
# FIT MODELS #
##############

####################################
#1- NAIVE BAYES
####################################

#Without parameters
model = GaussianNB()
model.fit(X_train, y_train)
print(model)
# make predictions
expected = y_test
predicted = model.predict(X_test)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
print("Accuracy Naive Bayes: ",metrics.accuracy_score(expected, predicted))

#NB: looking for best parameters
model = GaussianNB()
params_NB = {'var_smoothing': np.logspace(0,-9, num=100)}
gs_NB = GridSearchCV(estimator=model,
                 param_grid=params_NB,
                 cv=4,   # use any cross validation technique
                 verbose=1,
                 scoring='accuracy')
gs_NB.fit(X_train, y_train)
print("Best NB parameters are: ",gs_NB.best_params_)

# fit a Naive Bayes model to the data
model = GaussianNB(var_smoothing=gs_NB.best_params_['var_smoothing'])
model.fit(X_train, y_train)
print(model)
# make predictions
expected = y_test
predicted = model.predict(X_test)
# summarize the fit of the model
classification_model = metrics.classification_report(expected, predicted)
confusion_model = metrics.confusion_matrix(expected, predicted)
accuracy_model = metrics.accuracy_score(expected, predicted)
print(classification_model)
print(confusion_model)
print("Accuracy Naive Bayes with best parameters:",accuracy_model)

#Save metrics
with open(save_report, "w") as text_file:
    text_file.write("\nAccuracy Naive Bayes with best parameters:")
    text_file.write(str(accuracy_model))
    text_file.write("\nClassification Naive Bayes with best parameters:\n")
    text_file.write(classification_model)
# save the model to disk
filename = os.path.join('models','NB_model_best.sav')
pickle.dump(model, open(filename, 'wb'))


####################################
#2-SVM
####################################

#Without parameters
model = SVC()
model.fit(X_train, y_train)
print(model)
# make predictions
expected = y_test
predicted = model.predict(X_test)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
print("Accuracy SVM: ",metrics.accuracy_score(expected, predicted))

#SVM: looking for best parameters
def svc_param_selection(X, y, nfolds):
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=nfolds)
    grid_search.fit(X_train, y_train)
    grid_search.best_params_
    return grid_search.best_params_
best_param_svm = svc_param_selection(X_train, y_train,4)
print("Best SVM parameters are: ",best_param_svm)

# fit a SVM model to the data
model = SVC(C=best_param_svm['C'],gamma=best_param_svm['gamma'])
model.fit(X_train, y_train)
print(model)
# make predictions
expected = y_test
predicted = model.predict(X_test)
# summarize the fit of the model
classification_model = metrics.classification_report(expected, predicted)
confusion_model = metrics.confusion_matrix(expected, predicted)
accuracy_model = metrics.accuracy_score(expected, predicted)
print(classification_model)
print(confusion_model)
print("Accuracy SVM with best parameters: ",accuracy_model)

#Save metrics
with open(save_report, "a") as text_file:
    text_file.write("\nAccuracy SVM with best parameters: ")
    text_file.write(str(accuracy_model))
    text_file.write("\nClassification SVM with best parameters:\n")
    text_file.write(classification_model)

# save the model to disk
filename = os.path.join('models','SVM_model_best.sav')
pickle.dump(model, open(filename, 'wb'))

####################################
#3-K.nn
####################################

model = KNeighborsClassifier()
model.fit(X_train, y_train)
print(model)
# make predictions
expected = y_test
predicted = model.predict(X_test)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
print("Accuracy K-nn: ",metrics.accuracy_score(expected, predicted))

#K-NN: looking for best parameters
knn2 = KNeighborsClassifier()#create a dictionary of all values we want to test for n_neighbors
param_grid = {'n_neighbors': np.arange(1, 25)}#use gridsearch to test all values for n_neighbors
knn_gscv = GridSearchCV(knn2, param_grid, cv=5)#fit model to data
knn_gscv.fit(X_train, y_train)
print("Best parameters for K-nn: ",knn_gscv.best_params_)

# fit a k-nearest neighbor model to the data
model = KNeighborsClassifier(n_neighbors=knn_gscv.best_params_['n_neighbors'])
model.fit(X_train, y_train)
print(model)
# make predictions
expected = y_test
predicted = model.predict(X_test)
# summarize the fit of the model
classification_model = metrics.classification_report(expected, predicted)
confusion_model = metrics.confusion_matrix(expected, predicted)
accuracy_model = metrics.accuracy_score(expected, predicted)
print(classification_model)
print(confusion_model)
print("Accuracy K-nn with best parameters: ",accuracy_model)

#Save metrics
with open(save_report, "a") as text_file:
    text_file.write("\nAccuracy K-nn with best parameters: ")
    text_file.write(str(accuracy_model))
    text_file.write("\nClassification K-nn with best parameters:\n")
    text_file.write(classification_model)

# save the model to disk
filename = os.path.join('models','KNN_model_best.sav')
pickle.dump(model, open(filename, 'wb'))


####################################
#4-Neural
####################################
#Evaluation
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
# define the keras model
model = Sequential()
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
history = model.fit(np.asarray(X_train), np.asarray(y_train), epochs=15, batch_size=1)
# evaluate the keras model
_, accuracy = model.evaluate(np.asarray(X_test), np.asarray(y_test))
print('Accuracy: %.2f' % (accuracy*100))
# make class predictions with the model
expected = np.asarray(y_test)
predicted = model.predict_classes(np.asarray(X_test))
# summarize the fit of the model
classification_model = metrics.classification_report(expected, predicted)
confusion_model = metrics.confusion_matrix(expected, predicted)
accuracy_model = metrics.accuracy_score(expected, predicted)
print(classification_model)
print(confusion_model)
print("Accuracy NN with best parameters: ",accuracy_model)

#Save metrics
with open(save_report, "a") as text_file:
    text_file.write("\nAccuracy NN with best parameters: ")
    text_file.write(str(accuracy_model))
    text_file.write("\nClassification NN with best parameters:\n")
    text_file.write(classification_model)

#Save model
filename = os.path.join('models','NN_model_best.h5')
model.save(filename)

####################
# APPLY BEST MODEL #
####################

# load best SVM
loaded_model = pickle.load(open(os.path.join('models','SVM_model_best.sav'), 'rb'))
#predict
predicted = loaded_model.predict(features)
#add to dataframe
data['prediction'] = predicted
print(data)
data.to_csv(os.path.join('listPlaces','listPlaces_classified.tsv'), sep='\t', encoding='utf-8')
