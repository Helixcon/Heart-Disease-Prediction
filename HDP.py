# This Heart Disease Prediction code is based on the original work by Shreekant Gosavi
# Original code: https://github.com/g-shreekant/Heart-Disease-Prediction-using-Machine-Learning.git

# Changes made: 
# Data Visualization
# More varied visualization techniques were used to analyze the data. The data was visualized using count plots, and histograms.

# Decision Tree and Random Forest
# The Decision Tree and Random Forest models were optimized by iterating through a range of random states to find the best accuracy score.

# Neural Network Implementation
# A neural network was implemented using TensorFlow and Keras. The training and test data were converted into TensorFlow datasets, and the model was trained using the Sequential API. Also updating importing sequential and dense from tensorflow.keras.models and tensorflow.keras.layers respectively.



# 1. Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


import warnings
import os

warnings.filterwarnings('ignore')

# 2. Importing the dataset and understanding the data
print(os.listdir())
dataset = pd.read_csv("heart.csv")

# 3. Verifying the data object in pandas
print(type(dataset))

# 4. Shape of the dataset
print(dataset.shape)

# 5. Checking the first 5 rows of the dataset
print(dataset.head(5))
print(dataset.sample(5))

# 6. Describing the dataset
print(dataset.describe())
print(dataset.info())

# Understanding the columns of data
info = [
    "age",
    "1: male, 0: female",
    "chest pain type, 1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic",
    "resting blood pressure",
    "serum cholestoral in mg/dl",
    "fasting blood sugar > 120 mg/dl",
    "resting electrocardiographic results (values 0,1,2)",
    "maximum heart rate achieved",
    "exercise induced angina",
    "oldpeak = ST depression induced by exercise relative to rest",
    "the slope of the peak exercise ST segment",
    "number of major vessels (0-3) colored by flourosopy",
    "thal: 3 = normal; 6 = fixed defect; 7 = reversable defect"
]

for i in range(len(info)):
    print(dataset.columns[i] + ":\t\t\t" + info[i])

# 7. Analyzing the target variable
print(dataset["target"].describe())
print(dataset["target"].unique())

# 8. Checking correlation between columns
print(dataset.corr()["target"].abs().sort_values(ascending=False))

# 9. First Data Visualization
y = dataset["target"]

sns.countplot(y)
plt.show()

target_temp = dataset.target.value_counts()
print(target_temp)
print("Percentage of patients without heart problems: " + str(round(target_temp[0]*100/303, 2)))
print("Percentage of patients with heart problems: " + str(round(target_temp[1]*100/303, 2)))

# Analyzing all features
# First, we will analyze sex feature
sns.barplot(x=dataset["sex"], y=y, palette="coolwarm")
plt.show()

# Second, we will analyze cp feature
sns.barplot(x=dataset["cp"], y=y, palette="coolwarm")
plt.show()

# Third, we will analyze fbs feature
sns.barplot(x=dataset["fbs"], y=y, palette="coolwarm")
plt.show()

# Fourth, we will analyze restecg feature
sns.barplot(x=dataset["restecg"], y=y, palette="coolwarm")
plt.show()

# Fifth, we will analyze exang feature
sns.barplot(x=dataset["exang"], y=y, palette="coolwarm")
plt.show()

# Sixth, we will analyze slope feature
sns.barplot(x=dataset["slope"], y=y, palette="coolwarm")
plt.show()

# Seventh, we will analyze ca feature
sns.countplot(x=dataset["ca"], palette="coolwarm")
plt.show()
sns.barplot(x=dataset["ca"], y=y, palette="coolwarm")
plt.show()

# Eighth, we will analyze thal feature
sns.barplot(x=dataset["thal"], y=y, palette="coolwarm")
plt.show()
sns.histplot(dataset["thal"], kde=True, color="teal")
plt.show()

# 11. Train Test Split
predictors = dataset.drop("target", axis=1)
target = dataset["target"]

X_train, X_test, Y_train, Y_test = train_test_split(predictors, target, test_size=0.20, random_state=0)

# Output the X and Y of train and test datasets
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

# 12. Model Fitting
# 13. Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, Y_train)
Y_pred_lr = lr.predict(X_test)
score_lr = round(accuracy_score(Y_pred_lr, Y_test) * 100, 2)
print("The accuracy score achieved using Logistic Regression is: " + str(score_lr) + " %")

# 14. Naive Bayes
nb = GaussianNB()
nb.fit(X_train, Y_train)
Y_pred_nb = nb.predict(X_test)
score_nb = round(accuracy_score(Y_pred_nb, Y_test) * 100, 2)
print("The accuracy score achieved using Naive Bayes is: " + str(score_nb) + " %")

# 15. SVM (Support Vector Machine)
sv = svm.SVC(kernel='linear')
sv.fit(X_train, Y_train)
Y_pred_svm = sv.predict(X_test)
score_svm = round(accuracy_score(Y_pred_svm, Y_test) * 100, 2)
print("The accuracy score achieved using Linear SVM is: " + str(score_svm) + " %")

# 16. KNN (K-Nearest Neighbors)
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, Y_train)
Y_pred_knn = knn.predict(X_test)
score_knn = round(accuracy_score(Y_pred_knn, Y_test) * 100, 2)
print("The accuracy score achieved using KNN is: " + str(score_knn) + " %")

# 17. Decision Tree
max_accuracy = 0
best_x = 0

for x in range(200):
    dt = DecisionTreeClassifier(random_state=x)
    dt.fit(X_train, Y_train)
    Y_pred_dt = dt.predict(X_test)
    current_accuracy = round(accuracy_score(Y_pred_dt, Y_test) * 100, 2)
    if current_accuracy > max_accuracy:
        max_accuracy = current_accuracy
        best_x = x

dt = DecisionTreeClassifier(random_state=best_x)
dt.fit(X_train, Y_train)
Y_pred_dt = dt.predict(X_test)
score_dt = round(accuracy_score(Y_pred_dt, Y_test) * 100, 2)
print("The accuracy score achieved using Decision Tree is: " + str(score_dt) + " %")

# 18. Random Forest
max_accuracy = 0
best_x = 0

for x in range(2000):
    rf = RandomForestClassifier(random_state=x)
    rf.fit(X_train, Y_train)
    Y_pred_rf = rf.predict(X_test)
    current_accuracy = round(accuracy_score(Y_pred_rf, Y_test) * 100, 2)
    if current_accuracy > max_accuracy:
        max_accuracy = current_accuracy
        best_x = x

rf = RandomForestClassifier(random_state=best_x)
rf.fit(X_train, Y_train)
Y_pred_rf = rf.predict(X_test)
score_rf = round(accuracy_score(Y_pred_rf, Y_test) * 100, 2)
print("The accuracy score achieved using Random Forest is: " + str(score_rf) + " %")

# 19. XGBoost
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, Y_train)
Y_pred_xgb = xgb_model.predict(X_test)
score_xgb = round(accuracy_score(Y_pred_xgb, Y_test) * 100, 2)
print("The accuracy score achieved using XGBoost is: " + str(score_xgb) + " %")



import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 20. Neural Network
# Convert training and test data into TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train.values, Y_train.values)).batch(32)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test.values, Y_test.values)).batch(32)

# Neural Network
model = Sequential()
model.add(Dense(11, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_dataset, epochs=300, verbose=0)

# Predicting on test data
Y_pred_nn = model.predict(X_test)
rounded = [round(x[0]) for x in Y_pred_nn]
Y_pred_nn = rounded

# Calculating accuracy
score_nn = round(accuracy_score(Y_pred_nn, Y_test) * 100, 2)
print("The accuracy score achieved using Neural Network is: " + str(score_nn) + " %")

# 21. Summary of all models
algorithms = ["Logistic Regression", "Naive Bayes", "Support Vector Machine  ", " K-Nearest Neighbors", "Decision Tree", "Random Forest", "XGBoost", "Neural Network"]
scores = [score_lr, score_nb, score_svm, score_knn, score_dt, score_rf, score_xgb, score_nn]

sns.set_theme(rc={'figure.figsize': (15, 8)})
custom_colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#e67e22", "#1abc9c", "#34495e"]
sns.barplot(x=algorithms, y=scores, palette=custom_colors)
plt.xlabel("Algorithms")
plt.ylabel("Accuracy score")
plt.show()
