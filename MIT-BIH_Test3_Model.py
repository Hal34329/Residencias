#
# Import libraries and modules
#
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re # For regular expressions with Python

#
# Reading MIT-BIH Arrhythmia Dataset
#
data_df = pd.read_csv('C:/Users/HP/Documents/2024/Python/GitKraken/PythonWorkspace/Modelo residencias/MIT-BIH Arrhythmia Database.csv') 
print(data_df.shape)
data_df.head()

#
# Delete QRS Morphology features from Data frame
#
qrs_morph_columns = [col for col in data_df.columns if re.search(r'\d+_qrs_morph\d+', col)]
data_df_no_morph = data_df.drop(columns=qrs_morph_columns)

#
# Split the data into features and class labels
#
x_data = data_df_no_morph.iloc[:, 2:].copy()
y_label = data_df_no_morph[['type']].copy()

#
# Transform multi-class labels into binary-class (arrhythmia and normal)
#
y_label.replace(['VEB', 'SVEB', 'F', 'Q'], 'arrhythmia', inplace=True)
y_label.replace(['N'], 'normal', inplace=True)

#
# Train-test Split
#
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_data, y_label, random_state=101)

#
# Feature Scaling
#
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()
X_train_scaled = min_max_scaler.fit_transform(X_train)
X_test_scaled = min_max_scaler.transform(X_test)

print(min_max_scaler.scale_)

#
# Model Training
#
from sklearn.ensemble import RandomForestClassifier

y_train_1d = y_train.values.ravel()  # Convertir a array 1D

model = RandomForestClassifier(random_state=101, n_estimators=150)
model.fit(X_train_scaled, y_train_1d)

# Training accuracy
print('Accuracy for the train data', model.score(X_train_scaled, y_train))

#
# Model Testing
#
from sklearn import metrics
y_pred = model.predict(X_test_scaled)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print("*** Confusion Matrix ***")
print(metrics.confusion_matrix(y_test, y_pred))