from utils import db_connect
engine = db_connect()

# your code here
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Loading the dataset directly from the URL
url = "https://raw.githubusercontent.com/4GeeksAcademy/linear-regression-project-tutorial/main/medical_insurance_cost.csv"
df = pd.read_csv(url)

# Quick look at the first few rows to make sure it's loaded correctly
print("First rows of the dataset:")
print(df.head())

# Basic info about the dataset (types, non-null counts, etc.)
print("\nGeneral info:")
print(df.info())

# Descriptive stats to understand the numeric variables
print("\nDescriptive statistics:")
print(df.describe())

# Checking for any missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Plotting the distribution of the target variable 'charges'
sns.histplot(df['charges'], kde=True)
plt.title("Distribution of 'charges'")
plt.show()

# Boxplots to visually detect outliers in numeric columns
num_vars = ['age', 'bmi', 'children', 'charges']
for var in num_vars:
    sns.boxplot(x=df[var])
    plt.title(f'Boxplot of {var}')
    plt.show()

# Function to remove outliers using the IQR method
def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

# Removing outliers from 'bmi' and 'charges' (those looked a bit extreme)
df = remove_outliers_iqr(df, 'bmi')
df = remove_outliers_iqr(df, 'charges')

# Converting categorical variables into numeric ones using one-hot encoding
df_encoded = pd.get_dummies(df, drop_first=True)
print("\nEncoded data (first rows):")
print(df_encoded.head())

# Separating features (X) and target (y)
X = df_encoded.drop('charges', axis=1)
y = df_encoded['charges']

# Splitting the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and training the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions with the test data
y_pred = model.predict(X_test)

# Evaluating the model with Mean Squared Error and R-squared
print("\nModel evaluation:")
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Plotting the predicted values against the actual values
plt.scatter(y_test, y_pred)
plt.xlabel("Actual values")
plt.ylabel("Predicted values")
plt.title("Predicted vs Actual")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')  # ideal fit line
plt.show()