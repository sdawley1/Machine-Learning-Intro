%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, VotingRegressor
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

# Make NumPy printouts easier to read
np.set_printoptions(precision=3, suppress=True)

# URL to extract data from
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'

# Names of columns in data set
# Displacement == mileage on car
# Horsepower == how many horses 
column_names = ['MPG', 'Cylinders', 'Displacement', 
                'Horsepower', 'Weight', 'Acceleration', 
                'Model Year', 'Origin']

# Reading the data from the URL above
# Parameters are data set specific
# dtype = 'a' specifies float64 formatting
raw_dataset = pd.read_csv(url, names=column_names, na_values='?', 
                          comment='\t', sep=' ', skipinitialspace=True)

# Converting the elements of the 'Origin' column to something that makes sense,
# that is, '1' changes to 'USA', etc., so that we can see the place of origin
raw_dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})

# `cleanData` is all of the data contained in the infile that is not a NaN
# We're effectively 'cleaning' the data
cleanData = raw_dataset.dropna()
cleanDataC = cleanData.copy()

# Now we'll normalize the data using `sklearn`
scaler = StandardScaler().fit(cleanData)
temp_scaled = scaler.transform(cleanData)

# Converting the data back into a pandas dataframe
indices = [i for i in range(7)]
cleanDataC = pd.DataFrame(temp_scaled, columns=column_names)

cleanDataC = cleanData.copy()

# The training data set is chosen arbitrarily as the first 350 rows of the data frame
# Obviously the number of rows / method would have to change for larger data sets
# I managed to find an incredibly convoluted way of identifying all the columns 
# except for the MPG for x_train. Though, it does get the job done.
x_train = cleanDataC.drop(cleanDataC.columns[[0]], axis=1)[:371].to_numpy()
y_train = cleanData.MPG[:371].to_numpy()

# The test data set is what will test the ML model
# cleanData is a pd dataframe and so we can access MPG data directly as a method
x_test = cleanDataC.drop(cleanDataC.columns[[0]], axis=1)[371:].to_numpy()
y_test = cleanData.MPG[371:].to_numpy()

# Plotting the figure and making it look nice
plt.figure()
plt.title('Estimate Made by Linear Regression')
plt.xlabel('True MPG'); plt.ylabel('Predicted MPG')
lims = [0, 40]; plt.xlim(lims); plt.ylim(lims)

# This is the linear regression which will model the data
# Mathematically, this is performing an ordinary least squares method
reg = LinearRegression()
reg.fit(x_train, y_train)
print('Mean Accuracy of Linear Regression (R^2 value): {:0.7f}'.format(reg.score(x_test, y_test)))

# This is the ridge regression which will be compared to the linear regression
regR = Ridge()
regR.fit(x_train, y_train)
print('Mean Accuracy of Ridge Regression (R^2 value):  {:0.7f}'.format(regR.score(x_test, y_test)))

# Results of our model - How well the regression will predict MPG
res = reg.predict(x_test)
resM = reg.predict(x_train)

# Scatter plot of the test data 
# (and the training data to get a sense of how well the model did in general)
plt.scatter(y_test, res, c='r', label='Test Data')
plt.scatter(y_train, resM, c='k', alpha=0.05, label='Training Data')

# This is the line y=x. If our model were perfect, every point on the plot would lie on this line
# Otherwise, there is some error in our regression
plt.plot(lims, lims, c='k', lw=2, label='No Losses')

plt.legend()

# Plotting the figure and making it look nice
plt.figure()
plt.title('Estimate Made by Random Forest')
plt.xlabel('True MPG'); plt.ylabel('Predicted MPG')
lims = [0, 40]; plt.xlim(lims); plt.ylim(lims)

# Creating the random forest
# `n_estimators` is a hyperparameter than can be optimized. 4 seems to work well
rfr = RandomForestRegressor(n_estimators=4, random_state=314159)

# Fitting the random forest model on the training data
rfr.fit(x_train, y_train)
print('Mean Accuracy of Random Forest (R^2 value):  {:0.7f}'.format(rfr.score(x_test, y_test)))

# Predicting MPG from the test data
rfrTrain = rfr.predict(x_train)
rfrTest = rfr.predict(x_test)

# Scatter plot of the results
plt.scatter(y_test, rfrTest, c='c', label='Test Forest')
plt.scatter(y_train, rfrTrain, c='k', alpha=0.05, label='Training Forest')

# This is the line y=x.
plt.plot(lims, lims, c='k', lw=2, label='No Losses')

plt.legend()

# Plotting the figure and making it look nice
plt.figure()
plt.title('Estimate Made by AdaBoosted Random Forest')
plt.xlabel('True MPG'); plt.ylabel('Predicted MPG')
lims = [0, 40]; plt.xlim(lims); plt.ylim(lims)

# Initializing AdaBoost using the random forest base estimator
ABT = AdaBoostRegressor(base_estimator=rfr, random_state=314159)

# Fitting the AdaBoosted Forest to the training data
ABT.fit(x_train, y_train)
print('Mean Accuracy of AdaBoosted Random Tree (R^2 value):  {:0.7f}'.format(ABT.score(x_test, y_test)))

# Predicting MPG from the test data
ABTtrain = ABT.predict(x_train)
ABTres = ABT.predict(x_test)

# Scatter plot of the results
plt.scatter(y_test, ABTres, c='g', label='Test AdaBoost')
plt.scatter(y_train, ABTtrain, c='k', alpha=0.05, label='Training AdaBoost')

# This is the line y=x. If our model were perfect, every point on the plot would lie on this line
# Otherwise, there is some error in our regression
plt.plot(lims, lims, c='k', lw=2, label='No Losses')

plt.legend()

# Plotting the figure and making it look nice
plt.figure()
plt.title('Estimate Made by AdaBoosted Random Forest')
plt.xlabel('True MPG'); plt.ylabel('Predicted MPG')
lims = [0, 40]; plt.xlim(lims); plt.ylim(lims)

# Initializing AdaBoost using the random forest base estimator
ABT = AdaBoostRegressor(base_estimator=rfr, random_state=314159)

# Fitting the AdaBoosted Random forest to the training data
ABT.fit(x_train, y_train)
print('Mean Accuracy of AdaBoosted Random Tree (R^2 value):  {:0.7f}'.format(ABT.score(x_test_10, y_test_10)))

# Predicting MPG from the test data WITHOUT the outlier
ABTtrain = ABT.predict(x_train)
ABTres = ABT.predict(x_test_10)

# Scatter plot of the results
plt.scatter(y_test_10, ABTres, c='g', label='Test AdaBoost')
plt.scatter(y_train, ABTtrain, c='k', alpha=0.05, label='Training AdaBoost')

# This is the line y=x. If our model were perfect, every point on the plot would lie on this line
# Otherwise, there is some error in our regression
plt.plot(lims, lims, c='k', lw=2, label='No Losses')

plt.legend()

# Plotting the figure and making it look nice
plt.figure()
plt.title('Estimate Made by Voting Regression')
plt.xlabel('True MPG'); plt.ylabel('Predicted MPG')
lims = [0, 40]; plt.xlim(lims); plt.ylim(lims)

# Fitting the VotingRegressor to the training data
VR = VotingRegressor(estimators=[('rf', rfr), ('abrfr', ABT)], n_jobs=-1).fit(x_train, y_train)
print('Mean Accuracy of Voting Regression (R^2 value):  {:0.7f}'.format(VR.score(x_test_10, y_test_10)))

# Predicting MPG from the test data WITHOUT the outlier
VRtrain = VR.predict(x_train)
VRres = VR.predict(x_test_10)

# Scatter plot of the results
plt.scatter(y_test_10, VRres, c='r', label='Test Voting')
plt.scatter(y_train, VRtrain, c='k', alpha=0.05, label='Training Voting')

# This is the line y=x. If our model were perfect, every point on the plot would lie on this line
# Otherwise, there is some error in our regression
plt.plot(lims, lims, c='k', lw=2, label='No Losses')

plt.legend()

# Make NumPy printouts easier to read
np.set_printoptions(precision=3, suppress=True)

# First, download and import data using pandas
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

raw_dataset = pd.read_csv(url, names=column_names,
                          na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True)

# Make copy of raw data to 
dataset = raw_dataset.copy()

# Here we 'clean' the data, finding and omitting NaNs
dataset.isna().sum()
dataset = dataset.dropna()

# The 'Origin' column is categorical instead of numerical, so we'll change that
# I guess this is sometimes referred to as converting it to a 'one-hot'
dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})

dataset = pd.get_dummies(dataset, prefix='', prefix_sep='')
dataset

# Split dataset into training and testing sets
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# Examine overall statistics of data
train_dataset.describe().transpose()

# Seperating the target value (MPG) from the features
# `X_features' corresponds to those values which we'll use to estimate the target
# `X_labels` corresponds to the value we're trying to estimate, in this case MPG
train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('MPG')
test_labels = test_features.pop('MPG')

# The range of each piece of data is very different --> Normalization!
train_dataset.describe().transpose()[['mean', 'std']]

# The `preprocessing.Normalization` layer offers a simple way to implement preprocessing
normalizer = preprocessing.Normalization()

# After establishing the layer, we'll adapt it to the data with `adapt()` and
# Then compute the means and variances of each metric, storing the output 
normalizer.adapt(np.array(train_features))

# When the layer is called, it returns the data, properly normalized
# `first` is defined to offer a comparison when we print the data
first = np.array(train_features[:1])

# The build and compile methods are included in a single function
def build_and_compile_model(norm):
    model = keras.Sequential([
        norm,
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(0.001))
    return model

# Start DNN model with a all inputs
dnn_model = build_and_compile_model(normalizer)

# Train the model on all parameters
history = dnn_model.fit(
    train_features, train_labels,
    validation_split=0.2,
    verbose=0, epochs=100)

# Function to plot the loss of the model
def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)
    return

plot_loss(history)

test_predictions = dnn_model.predict(test_features).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions, c='r')
plt.xlabel('True MPG')
plt.ylabel('Predicted MPG')
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
plt.title('Estimate Made by Machine Learning')
_ = plt.plot(lims, lims, c='k', label='No Loess')

error = test_predictions - test_labels
plt.hist(error, bins=25, color='r')
plt.xlabel('Predicted MPG Error')
plt.title('Error Distribution')
_ = plt.ylabel('Count')
