import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier # NN Classifier (well this is useless because keras)
from sklearn.linear_model import LogisticRegression # LogReg Classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree # Decision Tree Classifier
from sklearn.svm import SVC # Support vector Machines
from sklearn.cluster import KMeans # for Kmeans clustering
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.preprocessing import normalize
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import keras

# Neural Networks
# SVM with various kernels
# Logistics Regression
# Decision Trees
# Clustering maybe? We'll see how the computer defines genres


# import and clean the data: It's already pretty clean BUT
df_30_sec = pd.read_csv("C:/Users/ICD-15/Music-Classification-Project/features_30_sec.csv")
df_3_sec = pd.read_csv("C:/Users/ICD-15/Music-Classification-Project/features_3_sec.csv")

df_30_sec.drop(['filename', 'length'], axis = 1, inplace = True)
df_3_sec.drop(['filename', 'length'], axis = 1, inplace = True)
df_30_sec.label = pd.Categorical(df_30_sec.label)
df_30_sec['genre_num'] = df_30_sec.label.cat.codes
df_3_sec.label = pd.Categorical(df_3_sec.label)
df_3_sec['genre_num'] = df_3_sec.label.cat.codes


print(df_30_sec.dtypes, df_30_sec.shape)
print(df_3_sec.dtypes, df_3_sec.shape)
print(df_3_sec[['label', 'genre_num']])
# X_train, X_test, y_train, y_test = train_test_split(df_30_sec.drop(['label'], axis=1), df_30_sec['label'], test_size = 0.1)
# # Logistic Regression Model
# Log_Model = LogisticRegression(solver='liblinear', max_iter=10000).fit(X_train,y_train)
# yhat_log = Log_Model.predict(X_test)
# print(confusion_matrix(y_test, yhat_log))


X_train, X_test, y_train, y_test = train_test_split(df_3_sec.drop(['label', 'genre_num'], axis=1), df_3_sec['genre_num'], test_size = 0.1)
X_train_norm, X_test_norm = normalize(X_train), normalize(X_test)


# Logistic Regression Model (Train: 0.53 - Test: 0.52)
Log_Model = LogisticRegression(solver='liblinear', max_iter=10000).fit(X_train,y_train)
yhat_log_train = Log_Model.predict(X_train)
yhat_log = Log_Model.predict(X_test)
mat1_train = confusion_matrix(y_train, yhat_log_train)
mat1 = confusion_matrix(y_test, yhat_log)
print('Training Accuracy LogReg: ', np.trace(mat1_train)/np.sum(mat1_train))
print('Testing Accuracy LogReg: ', np.trace(mat1)/np.sum(mat1))


# KNN Classification model (Train: 0.51 - Test: 0.32)
KNN_Model = KNeighborsClassifier(p =1).fit(X_train_norm, y_train)
yhat_knn_train = KNN_Model.predict(X_train_norm)
yhat_knn = KNN_Model.predict(X_test_norm)
mat2_train = confusion_matrix(y_train,yhat_knn_train)
mat2 = confusion_matrix(y_test, yhat_knn)
print('Training Accuracy KNN: ', np.trace(mat2_train)/np.sum(mat2_train))
print('Testing Accuracy KNN:', np.trace(mat2)/np.sum(mat2))


# SVM RBF (Train: 0.52 - Test: 0.45) (This worked but only when C and gamma were adjusted. It don't like small gamma)
SVM_Model = SVC(C = 1000, kernel='rbf', decision_function_shape= 'ovo', gamma= 100).fit(X_train_norm, y_train)
yhat_svm_train = SVM_Model.predict(X_train_norm)
yhat_svm = SVM_Model.predict(X_test_norm)
mat3_train = confusion_matrix(y_train, yhat_svm_train)
mat3 = confusion_matrix(y_test, yhat_svm)
print('Training Accuracy SVM: ',np.trace(mat3_train)/np.sum(mat3_train))
print('Testing Accuracy SVM:', np.trace(mat3)/np.sum(mat3))


# SVM Sigmoid (Train: 0.27 - Test: 0.27)
SVM_Model2 = SVC(C = 1000, kernel='sigmoid', decision_function_shape= 'ovo', gamma= 0.01).fit(X_train_norm, y_train)
yhat_svm2_train = SVM_Model2.predict(X_train_norm)
yhat_svm2 = SVM_Model2.predict(X_test_norm)
mat4_train = confusion_matrix(y_train, yhat_svm2_train)
mat4 = confusion_matrix(y_test, yhat_svm2)
print('Training Accuracy SVM2 Sigmoid: ', np.trace(mat4_train)/np.sum(mat4_train))
print('Testing Accuracy SVM2 Sigmoid: ', np.trace(mat4)/np.sum(mat4))


# SVM Linear (Train: 0.34 - Test: 0.34)
SVM_Model3 = SVC(C = 1000, kernel='linear', decision_function_shape= 'ovo', gamma= 0.01).fit(X_train_norm, y_train)
yhat_svm3_train = SVM_Model3.predict(X_train_norm)
yhat_svm3 = SVM_Model3.predict(X_test_norm)
mat5_train = confusion_matrix(y_train, yhat_svm3_train)
mat5 = confusion_matrix(y_test, yhat_svm3)
print('Training Accuracy SVM3 Linear: ', np.trace(mat5_train)/np.sum(mat5_train))
print('Testing Accuracy SVM3 Linear: ', np.trace(mat5)/np.sum(mat5))


# Keras Model (Train: 0.54 - Test: 0.54) (This model can vary widely)
y_train_mat = to_categorical(y_train)

model = Sequential()
model.add(Dense(20, input_dim = len(X_train.columns), activation='tanh'))
model.add(Dense(20, activation='tanh'))
model.add(Dense(15, activation='tanh'))
model.add(Dense(12, activation='tanh'))
model.add(Dense(10, activation = 'softmax'))

opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.summary()

model.fit(X_train_norm, y_train_mat, epochs=250)

yhat_neural_train = np.argmax(model.predict(X_train_norm), axis = -1)
yhat_NN = np.argmax(model.predict(X_test_norm), axis=-1)

NN_mat_train = confusion_matrix(y_train,yhat_neural_train)
NN_mat = confusion_matrix(y_test,yhat_NN)

# print(NN_mat, np.trace(NN_mat)/np.sum(NN_mat))
# print(confusion_matrix(yhat_NN, y_test))
print('Training Accuracy Neural Net: ', np.trace(NN_mat_train)/np.sum(NN_mat_train))
print('Testing Accuracy Neural Net:', np.trace(NN_mat)/np.sum(NN_mat))

# Plotting the results

fig, axes = plt.subplots(nrows=2, ncols=3)
axes[0][0].matshow(mat1)
axes[0][0].set_title('Logistic Regression')

axes[0][1].matshow(mat2)
axes[0][1].set_title('KNeighbors')

axes[0][2].matshow(NN_mat)
axes[0][2].set_title('Deep Neural Net')

axes[1][0].matshow(mat3)
axes[1][0].set_title('SVM Radial')

axes[1][1].matshow(mat4)
axes[1][1].set_title('SVM Sigmoidal')

axes[1][2].matshow(mat5)
axes[1][2].set_title('SVM Linear')

plt.show()
