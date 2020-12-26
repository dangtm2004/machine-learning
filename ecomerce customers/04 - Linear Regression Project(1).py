#Import pandas, numpy, matplotlib, and seaborn.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

#Read in the Ecommerce Customers csv file as a DataFrame called customers.
customers = pd.read_csv('Ecommerce Customers')

#Check the head of customers, and check out its info() and describe() methods.
print(customers.head())
print(customers.info())
print(customers.describe())

#Use seaborn to create a jointplot to compare the Time on Website and Yearly Amount Spent columns. Does the correlation make sense?
sns.jointplot(x='Time on Website', y='Yearly Amount Spent', data=customers, kind='scatter')
plt.show()
print("The correlation between [Time on Website] and [Yearly Amount Spent] does not make sense") 

#Do the same but with the Time on App column instead.
sns.jointplot(x='Time on App', y='Yearly Amount Spent', data=customers, kind='scatter')
plt.show()

#Use jointplot to create a 2D hex bin plot comparing Time on App and Length of Membership.
sns.jointplot(x='Time on App', y='Length of Membership', data=customers, kind="hex", color="#4CB391")
plt.show()

#Let's explore these types of relationships across the entire data set. Use pairplot to recreate the plot below.(Don't worry about the the colors)
sns.pairplot(customers)
plt.show()

#Based off this plot what looks to be the most correlated feature with Yearly Amount Spent?
print("[Length of membership] is most correlated feature with [Yearly Amount Spent]")

#Create a linear model plot(using seaborn's lmplot) of Yearly Amount Spent vs. Length of Membership.
#customers and a variable y equal to the "Yearly Amount Spent" column.
sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data = customers)
plt.show()

#Use model_selection.train_test_split from sklearn to split the data into training and testing sets. Set test_size=0.3 and random_state=101
#Import LinearRegression from sklearn.linear_model
#Create an instance of a LinearRegression() model named lm.
X=customers[['Avg. Session Length','Time on App','Length of Membership']]
y=customers['Yearly Amount Spent']
print("[Time on Website] does not have a relationship with [Year Amount Spent], so I take it out of training set")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

#Train/fit lm on the training data.
lm = LinearRegression().fit(X_train, y_train)

#Print out the coefficients of the model
coeff_customers = pd.DataFrame(lm.coef_, X.columns, columns=['Coefficient'])
print(coeff_customers)

#Use lm.predict() to predict off the X_test set of the data.
predictions = lm.predict(X_test)

#Create a scatterplot of the real test values versus the predicted values.
plt.scatter(y_test, predictions)
plt.show()

#Calculate the Mean Absolute Error, Mean Squared Error, and the Root Mean Squared Error. Refer to the lecture or to Wikipedia for the formulas
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

#Plot a histogram of the residuals and make sure it looks normally distributed. Use either seaborn distplot, or just plt.hist().
sns.distplot((y_test-predictions), bins=50)
plt.show()

#Recreate the dataframe below.
sns.pairplot(customers)
plt.show()
print("[Time on Website] does not have a relationship with [Year Amount Spent], so I take it out of training set")

#How can you interpret these coefficients?
print("The coefficients of [Avg. Session Length], [Time on App] and [Length of Membership] show that they have an affect on the target")
print("These coefficients are positive that mean when numbers of feature increase, the target also increase")
print("The coefficient value represents the mean change in the response given a one unit change in the predictor")

#Do you think the company should focus more on their mobile app or on their website?
print("The company should focus more on mobile app")
# Ref: www.pieriandata.com
