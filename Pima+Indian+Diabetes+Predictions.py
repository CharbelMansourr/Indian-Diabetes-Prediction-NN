
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

# Read the csv file
diabetes = pd.read_csv("diabetes.csv")


# Check for missing values
diabetes.isnull().sum(axis=0)


# Create X and Y variables
X = diabetes.iloc[:, 0:-1]
Y = diabetes.iloc[:,   -1]


# Split the data into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = \
    train_test_split(X, Y, test_size=0.2, random_state=1234, stratify=Y)


# Define the keras sequential model with three hidden layers
model = Sequential()

model.add(Dense(24, input_shape=(8,),activation='relu',kernel_initializer='RandomNormal'))


model.add(Dense(12, activation='relu',kernel_initializer='RandomNormal'))


model.add(Dense(1,activation='sigmoid',kernel_initializer='RandomNormal'))


# Compile the keras model for classification accuracy
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# Fit the model on the training dataset
model.fit(X_train, Y_train, epochs=160, batch_size=10)


# Evaluate the model
accuracy_test = model.evaluate(X_test, Y_test)


# Get the predicted probabilities and predicted classes
Y_pred_prob = model.predict(X_test)
predictions = (Y_pred_prob > 0.5).astype('int32')


# Create the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, predictions)











