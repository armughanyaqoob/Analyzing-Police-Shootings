#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install xlrd')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# Load the dataset from Excel
excel_file_path = 'C:/Users/Hp/Downloads/fatal-police-shootings-data (4).xls'  
df = pd.read_excel(excel_file_path)

# Display basic information about the DataFrame
print(df.info())

# Display basic statistics
print(df.describe())

# Plot histograms for numerical columns
df.hist(figsize=(12, 10))
plt.show()

# Plot correlation matrix heatmap
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix Heatmap')
plt.show()


import pandas as pd
import numpy as np
import statsmodels.api as sm
# Load the dataset from Excel
excel_file_path = 'C:/Users/Hp/Downloads/fatal-police-shootings-data.xls'  # Replace with the actual path to your Excel file
df = pd.read_excel(excel_file_path)

# Assuming df is your DataFrame
df = df.dropna()  # Drop rows with missing values

X = df[['age', 'longitude', 'latitude']]
y = df['is_geocoding_exact']

# Check for and handle infinite values
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X = X.dropna()

X = sm.add_constant(X)  # Add a constant term to the independent variables

model = sm.OLS(y, X).fit()
print(model.summary())


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Assuming df is your DataFrame
X = df[['age', 'longitude', 'latitude']]
y = df['is_geocoding_exact']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Assuming df is your DataFrame
features = ['age', 'longitude', 'latitude']
X = df[features]

# Standardize the data
X_std = StandardScaler().fit_transform(X)

pca = PCA()
pca_result = pca.fit_transform(X_std)

print("Explained Variance Ratio:", pca.explained_variance_ratio_)

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Assuming df is your DataFrame
X = df[['age', 'longitude', 'latitude']]

kmeans = KMeans(n_clusters=3)
df['cluster'] = kmeans.fit_predict(X)

plt.scatter(df['longitude'], df['latitude'], c=df['cluster'], cmap='viridis')
plt.show()

import numpy as np

# Assuming df is your DataFrame
age_sample = df['age'].sample(frac=0.8, replace=True)

# Bootstrap resampling
bootstrap_means = [age_sample.sample(frac=1, replace=True).mean() for _ in range(1000)]

# Confidence interval
conf_int = np.percentile(bootstrap_means, [2.5, 97.5])
print("Bootstrapped 95% Confidence Interval for Age Mean:", conf_int)


from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Assuming df is your DataFrame
X = df[['age', 'longitude', 'latitude']]
y = df['is_geocoding_exact']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

import pandas as pd
import matplotlib.pyplot as plt

# Assuming df is your DataFrame with a 'date' column
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Plot time series data
plt.plot(df.index, df['age'])
plt.xlabel('Date')
plt.ylabel('Age')
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

# Assuming df is your DataFrame
plt.figure(figsize=(8, 6))
sns.countplot(x='manner_of_death', data=df)
plt.title('Number of People Shot')
plt.xlabel('Manner of Death')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(12, 8))
sns.countplot(x='race', data=df, order=df['race'].value_counts().index)
plt.title('Distribution by Race')
plt.xlabel('Race')
plt.ylabel('Count')
plt.show()


plt.figure(figsize=(10, 5))
sns.countplot(x='state', data=df, order=df['state'].value_counts().index)
plt.title('State-wise Killings')
plt.xlabel('State')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(12, 8))
sns.histplot(x='age', hue='gender', data=df, bins=20, kde=True)
plt.title('Age and Gender Breakdown')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
# Load the dataset from Excel
excel_file_path = 'C:/Users/Hp/Downloads/fatal-police-shootings-data (4).xls'  # Replace with the actual path to your Excel file
df= pd.read_excel(excel_file_path)


df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Plot the number of fatal shootings over time
plt.figure(figsize=(12, 6))
df.resample('M').size().plot()
plt.title('Fatal Police Shootings Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Incidents')
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns
excel_file_path = 'C:/Users/Hp/Downloads/fatal-police-shootings-data (4).xls'  # Replace with the actual path to your Excel file
df= pd.read_excel(excel_file_path)
# Distribution by race
plt.figure(figsize=(12, 8))
sns.countplot(x='race', data=df, order=df['race'].value_counts().index)
plt.title('Distribution by Race')
plt.xlabel('Race')
plt.ylabel('Count')
plt.show()

# Age and gender brekdown
plt.figure(figsize=(5, 5))
sns.histplot(x='age', hue='gender', data=df, bins=20, kde=True)
plt.title('Age and Gender Breakdown')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()


plt.figure(figsize=(14, 8))
sns.countplot(x='state', data=df, order=df['state'].value_counts().index)
plt.title('State-wise Fatal Police Shootings')
plt.xlabel('State')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(5, 5))
sns.countplot(x='threat_level', data=df)
plt.title('Circumstances of Fatal Police Shootings')
plt.xlabel('Threat Level')
plt.ylabel('Count')
plt.show()


import seaborn as sns

# Assuming df is your DataFrame with longitude and latitude columns
plt.figure(figsize=(5, 5))
sns.kdeplot(data=df, x='longitude', y='latitude', cmap='viridis', fill=True)
plt.title('Kernel Density Estimation of Fatal Police Shootings')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
excel_file_path = 'C:/Users/Hp/Downloads/fatal-police-shootings-data (4).xls'  # Replace with the actual path to your Excel file
df= pd.read_excel(excel_file_path)

# Assuming df is your DataFrame
# Drop unnecessary columns for this example
df = df.drop(['id', 'name', 'date', 'city', 'state'], axis=1)

# Handle missing values by filling them with the mean value for numerical columns
df = df.fillna(df.mean())

# Convert categorical variables to numerical using one-hot encoding
df = pd.get_dummies(df, columns=['manner_of_death', 'armed', 'gender', 'race', 'signs_of_mental_illness', 'threat_level', 'flee', 'body_camera'])

# Split the data into features (X) and target variable (y)
X = df.drop('is_geocoding_exact', axis=1)
y = df['is_geocoding_exact']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_rf = rf_model.predict(X_test)

# Evaluate the Random Forest model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)

print(f"Random Forest Accuracy: {accuracy_rf}")
print(f"Random Forest Confusion Matrix:\n{conf_matrix_rf}")

# Plot the Random Forest confusion matrix
plt.figure(figsize=(3,3))
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Exact', 'Exact'], yticklabels=['Not Exact', 'Exact'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Random Forest Confusion Matrix')
plt.show()

# Number of top features to display
top_n = 10

# Get the indices of the top N features
top_feature_indices = feature_importances.argsort()[-top_n:][::-1]
top_feature_importances = feature_importances[top_feature_indices]
top_features = X.columns[top_feature_indices]

# Plot the top N features
plt.figure(figsize=(5, 5))
sns.barplot(x=top_feature_importances, y=top_features, orient='h', palette='viridis')
plt.xlabel('Feature Importance')
plt.title(f'Top {top_n} Random Forest Feature Importance')
plt.show()





import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta

# Load the dataset from Excel
excel_file_path = 'C:/Users/Hp/Downloads/fatal-police-shootings-data (4).xls'  # Replace with the actual path to your Excel file
df = pd.read_excel(excel_file_path)

# Preprocess the data
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Resample the data to get the count of shootings per month
monthly_shootings = df.resample('M').size()

# Fit ARIMA model
model = ARIMA(monthly_shootings, order=(5, 1, 0))  # You can adjust the order based on model diagnostics
model_fit = model.fit()

# Forecast future shootings
future_steps = 12  # You can adjust the number of future steps
forecast = model_fit.get_forecast(steps=future_steps)
forecast_index = pd.date_range(start=monthly_shootings.index[-1] + timedelta(days=30), periods=future_steps, freq='M')

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(monthly_shootings.index, monthly_shootings, label='Actual Shootings')
plt.plot(forecast_index, forecast.predicted_mean, color='red', label='Forecasted Shootings')
plt.title('Fatal Police Shootings Forecast')
plt.xlabel('Date')
plt.ylabel('Number of Incidents')
plt.legend()
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta

# Load the dataset from Excel
excel_file_path = 'C:/Users/Hp/Downloads/fatal-police-shootings-data (4).xls'  # Replace with the actual path to your Excel file
df = pd.read_excel(excel_file_path)

# Preprocess the data
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Resample the data to get the count of shootings per month
monthly_shootings = df.resample('M').size()

# Split the data into training and testing sets
train_size = int(len(monthly_shootings) * 0.8)
train, test = monthly_shootings[0:train_size], monthly_shootings[train_size:]

# Fit ARIMA model on the training set
model = ARIMA(train, order=(5, 1, 0))  # You can adjust the order based on model diagnostics
model_fit = model.fit()

# Forecast future shootings on the testing set
future_steps = len(test)
forecast = model_fit.get_forecast(steps=future_steps)
forecast_index = pd.date_range(start=test.index[0], periods=future_steps, freq='M')

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(train.index, train, label='Training Set')
plt.plot(test.index, test, label='Actual Shootings in Test Set')
plt.plot(forecast_index, forecast.predicted_mean, color='red', label='Forecasted Shootings')
plt.title('Fatal Police Shootings Forecast and Actual Values')
plt.xlabel('Date')
plt.ylabel('Number of Incidents')
plt.legend()
plt.show()



import matplotlib.pyplot as plt
import seaborn as sns
excel_file_path = 'C:/Users/Hp/Downloads/fatal-police-shootings-data (4).xls' 
df= pd.read_excel(excel_file_path)
# Distribution by race
plt.figure(figsize=(5, 5))
sns.countplot(x='race', data=df, order=df['race'].value_counts().index)
plt.title('Distribution by Race')
plt.xlabel('Race')
plt.ylabel('Count')
plt.show()





# In[ ]:
