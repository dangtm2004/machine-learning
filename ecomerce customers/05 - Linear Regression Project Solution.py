import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

#os.chdir('C:\\Users\kkhan\Documents\_CS Outreach\Python\Python Source Files')
customers = pd.read_csv("Ecommerce Customers")
customers.head()
customers.describe()
customers.info()
sns.set_palette("GnBu_d")
sns.set_style('whitegrid')
# More time on site, more money spent.
sns.jointplot(x='Time on Website', y='Yearly Amount Spent', data=customers)
sns.jointplot(x='Time on App', y='Yearly Amount Spent', data=customers)
sns.jointplot(x='Time on App', y='Length of Membership', kind='hex', data=customers)
sns.pairplot(customers)
# Length of Membership
sns.lmplot(x='Length of Membership', y='Yearly Amount Spent', data=customers)

y = customers['Yearly Amount Spent']
X = customers[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)

# The coefficients
print('Coefficients: \n', lm.coef_)
predictions = lm.predict(X_test)
plt.scatter(y_test, predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
# calculate these metrics by hand!
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
sns.distplot((y_test-predictions), bins=50)
coeffecients = pd.DataFrame(lm.coef_, X.columns)
coeffecients.columns = ['Coeffecient']
coeffecients

"""
How can you interpret these coefficients?
Interpreting the coefficients:
•   Holding all other features fixed, a 1 unit increase in Avg. Session Length is associated with an increase of 25.98 total dollars spent.
•   Holding all other features fixed, a 1 unit increase in Time on App is associated with an increase of 38.59 total dollars spent.
•   Holding all other features fixed, a 1 unit increase in Time on Website is associated with an increase of 0.19 total dollars spent.
•   Holding all other features fixed, a 1 unit increase in Length of Membership is associated with an increase of 61.27 total dollars spent.


Do you think the company should focus more on their mobile app or on their website?

This is tricky, there are two ways to think about this: Develop the Website to catch up to the performance of the mobile app,
or develop the app more since that is what is working better. This sort of answer really depends on the other factors going
on at the company, you would probably want to explore the relationship between Length of Membership and the App or
the Website before coming to a conclusion!
"""
#Ref: www.pieriandata.com
