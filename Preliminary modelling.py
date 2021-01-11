import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier # NN Classifier
from sklearn.linear_model import LogisticRegression # LogReg Classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree # Decision Tree Classifier
from sklearn.svm import SVC # Support vector Machines
from sklearn.cluster import KMeans # for Kmeans clustering
from sklearn.pipeline import Pipeline # to make things easy
from sklearn.pipeline import make_pipeline
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
# KNeighbors
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

# for now we''ll do the 3 second presegmented dataset because data augmentation is cool

X_train, X_test, y_train, y_test = train_test_split(df_3_sec.drop(['label', 'genre_num'], axis=1), df_3_sec['genre_num'], test_size = 0.1)
X_train_norm, X_test_norm = normalize(X_train), normalize(X_test)


# Logistic Regression Model : accuracy- approximately 0.53
Log_Model = LogisticRegression(solver='liblinear', max_iter=10000).fit(X_train,y_train)
yhat_log = Log_Model.predict(X_test)
mat1 = confusion_matrix(y_test, yhat_log)
print(np.trace(mat1)/np.sum(mat1))

# KNN Classification model : accuracy by metric - l1 : ~0.32, l2 : ~0.27. l3 ~0.21
KNN_Model = KNeighborsClassifier(p =1).fit(X_train_norm, y_train)
yhat_knn = KNN_Model.predict(X_test_norm)
mat2 = confusion_matrix(y_test, yhat_knn)
print(np.trace(mat2)/np.sum(mat2))

# Support Vector Model : accuracy by kernel - rbf: ~0.19 , linear:~0.21, sigmoid: ~0.11 (Higher C param meant greater accuracy but only marginallly ~0.3) 
SVM_Model = SVC(kernel='rbf', decision_function_shape= 'ovo').fit(X_train_norm, y_train)
yhat_svm = SVM_Model.predict(X_test_norm)
mat3 = confusion_matrix(y_test, yhat_svm)
print(np.trace(mat3)/np.sum(mat3))

y_train_mat = to_categorical(y_train)


# Keras Model
model = Sequential()
model.add(Dense(20, input_dim = len(X_train.columns), activation='tanh'))
model.add(Dense(20, activation = 'tanh'))
model.add(Dense(15, activation= 'tanh'))
model.add(Dense(12, activation='tanh'))
model.add(Dense(10, activation = 'sigmoid'))

opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

model.summary()

model.fit(X_train_norm, y_train_mat, epochs=250)

yhat_NN = np.argmax(model.predict(X_test_norm), axis=-1)
NN_mat = confusion_matrix(yhat_NN,y_test)
print(NN_mat, np.trace(NN_mat)/np.sum(NN_mat))
plt.matshow(NN_mat)
plt.show()
