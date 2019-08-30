import numpy as np
from sklearn import linear_model

# Create linear regression object
regr = linear_model.LinearRegression()
X_train = np.array([[1720,880,1910,1069,0],[1655,814,1907,1066,100]])
y_train = np.array([[41,40,49],[54,51,65]])
# Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the testing set
def al_level_coe(X_test):
	X_test = np.array(X_test)
	X_test = X_test.reshape(1, -1)
	y_pred = regr.predict(X_test)
	return (y_pred[0])

# The coefficients
# print('Coefficients: ', regr.coef_)
