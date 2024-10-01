### Import necessary libraries
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

### Load the CSV file by copying the file path into the parenthesis
data = pd.read_csv(r"NFL_Stats_Dataset.csv")


### Drop unwanted columns
data = data.drop(['Time of Possession (minutes)', 'Injury Index'], axis=1)

### Define dependent variable
y = data['Offensive Points Scored (dependent variable)']

### Define independent variables and add a constant term
X = sm.add_constant(data.drop('Offensive Points Scored (dependent variable)', axis=1))

### Add interaction term
X['RushingY_Oppdef_Interact'] = X["Home Advantage (1 for yes)"] * X["Opponent's Defensive Rank"]

### Create the OLS model
model = sm.OLS(y, X)

### Fit the model
results = model.fit()

### Print regression summary
print(results.summary())

### Check correlation matrix
correlation_matrix = data.corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

### Check VIF for multicollinearity
vif_data = pd.DataFrame()
vif_data['feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print('\nVIF Data:')
print(vif_data)
