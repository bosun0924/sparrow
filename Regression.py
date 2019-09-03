import numpy as np
from sklearn import linear_model

# Create linear regression object
regr = linear_model.LinearRegression()
X_train = np.array([[1720,880,1910,1069,0],[1655,814,1907,1066,100]])
y_train = np.array([[41,40,49],[54,51,65]])
# Train the model using the training sets
regr.fit(X_train, y_train)

# Create linear regression object (minimap on the left)
regr_l = linear_model.LinearRegression()
#X_train_l = np.array([[10,880,200,1069,0],[13,814,265,1066,100]])
X_train_l = np.array([[200,880,10,1069,0],[265,814,13,1066,100]])
y_train_l = np.array([[39,41,48],[52,53,65]])
# Train the model using the training sets
regr_l.fit(X_train_l, y_train_l)

# Make predictions using the testing set
def al_level_coe(X_test):
	X_test = np.array(X_test)
	X_test = X_test.reshape(1, -1)
	y_pred = regr.predict(X_test)
	return (y_pred[0])

def al_level_coe_left(X_test_l):
	X_test_l = np.array(X_test_l)
	X_test_l = X_test_l.reshape(1, -1)
	y_pred_l = regr_l.predict(X_test_l)
	return (y_pred_l[0])

# The coefficients
# print('Coefficients: ', regr.coef_)
