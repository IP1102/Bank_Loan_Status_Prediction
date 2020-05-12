
#%%
import tensorflow as tf
import pandas as pd
import numpy as np

#%%
# Data Read
df = pd.read_csv('credit_train.csv')

df.iloc[:,3:4].values

df.columns

#%% Data Analysis
X = df
X = X.drop(columns=['Loan ID', 'Customer ID'])


X.isna().sum().sort_values(ascending = False)


X['Months since last delinquent'].count()
X = X.drop(columns=['Months since last delinquent'])

X[X['Years of Credit History'].isnull() == True] #To chech the NaN ones
X.drop(X.tail(514).index, inplace=True) # drop last 514 rows

#To drop Max Open Cred NaN
for i in X['Maximum Open Credit'][X['Maximum Open Credit'].isnull() == True].index:
    X.drop(labels=i, inplace=True)

#To drop text liens NaN
for i in X['Tax Liens'][X['Tax Liens'].isnull() == True].index:
    X.drop(labels=i, inplace=True)

#To drop Bankruptcies
for i in X['Bankruptcies'][X['Bankruptcies'].isnull() == True].index:
    X.drop(labels=i, inplace=True)

X.fillna(X.mean(), inplace = True)


#To check for Years in Current job
pd.value_counts(X['Years in current job']).plot.bar()

X['Years in current job'].fillna('10+ years', inplace=True)

#%% Encoding

categorical_subset = X[['Term', 'Years in current job', 'Home Ownership', 'Purpose']]

#One Hot Encode
categorical_subset = pd.get_dummies(categorical_subset)

#Join 
X.drop(labels=['Term', 'Years in current job', 'Home Ownership', 'Purpose'], axis=1, inplace=True)
X = pd.concat([X, categorical_subset], axis = 1)

#Label Encode Loan Status
from sklearn.preprocessing import LabelEncoder
labelencoder_loan = LabelEncoder()
X['Loan Status'] = labelencoder_loan.fit_transform(X['Loan Status'])

#%% Split 
Y = X['Loan Status']
X = X.drop(columns = 'Loan Status')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 42)

#%% Neural Network 
import keras
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

#Add Input Layer and 1st Hidden Layer
model.add(Dense(24, input_dim = 44, activation = 'relu'))
#Add 2nd Hidden Layer
model.add(Dense(24,activation = 'relu'))
#Add 3rd
model.add(Dense(24, activation = 'relu'))
#Add the final layer
model.add(Dense(1, activation = 'sigmoid'))

#Compile
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) 

#Model fit
history = model.fit(X_train, y_train,
          batch_size = 10,
          epochs = 100,
          validation_data = (X_test,y_test),
          callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)])

#Save the model
model.save('model_nn')

#%% Tuning the NN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

def my_model(optimizer):
    model = Sequential()

    #Add Input Layer and 1st Hidden Layer
    model.add(Dense(24, input_dim = 44, activation = 'relu'))
    #Add 2nd Hidden Layer
    model.add(Dense(24,activation = 'relu'))
    #Add 3rd
    model.add(Dense(24, activation = 'relu'))
    #Add the final layer
    model.add(Dense(1, activation = 'sigmoid'))
    
    #Compile
    model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    return model

classifier = KerasClassifier(build_fn = my_model)
parameters = {'batch_size':[25,32],
              'optimizer':['adam', 'rmsprop'],
              'epochs': [100,200]}

gridSearch = GridSearchCV(estimator = classifier,
                          param_grid = parameters,
                          scoring = 'accuracy',
                          cv = 10)


gridSearch = gridSearch.fit(X_train,y_train)
best_accuracy = gridSearch.best_score_

#%%
#Logistic Regression

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr.predict(X_test)
lr.score(X_test, y_test)

#Save the model
import joblib
joblib.dump(lr,'model_lr')

#%% Plot the model loss
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

