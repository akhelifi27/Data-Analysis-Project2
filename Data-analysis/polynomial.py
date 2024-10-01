import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
# Load the data
data = pd.read_csv("NFL_Stats_Dataset.csv")
x = data["Passing Yards"].values
y = data['Offensive Points Scored (dependent variable)'].values
xsquared = x ** 2
# Adding a constant for the intercept term
x_with_const = sm.add_constant(x)

# Stack the polynomial term with the original x values and the constant
x_with_poly = np.column_stack((x_with_const, xsquared))
# Fit the linear regression model
model = sm.OLS(y, x_with_poly).fit()
# Predicted values
predictions = model.predict(x_with_poly)
# Get summary of regression
print(model.summary())
# Plotting the data and the regression line
plt.scatter(x, y, label='Data')
plt.plot(x, predictions, color='red', label='Regression Line')
plt.xlabel(' (x)')
plt.ylabel(' (y)')
plt.title('')
plt.legend()
plt.show()




