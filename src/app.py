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

# Quick look at the first few rows
print("First rows of the dataset:")
print(df.head())

# General info about the dataset
print("\nGeneral info:")
print(df.info())

# Observation:
# - No missing values.
# - 'sex', 'smoker', and 'region' are categorical.
# - Target variable is 'charges' (numeric).

# Descriptive statistics
print("\nDescriptive statistics:")
print(df.describe())

# Observation:
# - BMI ranges from 15.96 to 53.13 — quite a spread.
# - Charges range from $1,122 to $63,770 — definitely has high variance.
# - Children max out at 5; looks reasonable.

# Checking for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Observation:
# - No null values, so no imputation is necessary.

# Plotting the distribution of charges
sns.histplot(df['charges'], kde=True)
plt.title("Distribution of 'charges'")
plt.show()

# Observation:
# - 'charges' is right-skewed (not normally distributed).
# - We might consider log-transforming it for modeling (optional).

# Boxplots to detect outliers in numeric variables
num_vars = ['age', 'bmi', 'children', 'charges']
for var in num_vars:
    sns.boxplot(x=df[var])
    plt.title(f'Boxplot of {var}')
    plt.show()

# Observation:
# - Outliers detected in 'bmi' and 'charges' especially.
# - These may affect the regression model, so we’ll remove them.

# Function to remove outliers using IQR
def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

# Removing outliers from 'bmi' and 'charges'
df = remove_outliers_iqr(df, 'bmi')
df = remove_outliers_iqr(df, 'charges')

# Observation:
# - The dataset is now cleaner and should be less influenced by extreme values.

# Converting categorical variables to numeric using one-hot encoding
df_encoded = pd.get_dummies(df, drop_first=True)
print("\nEncoded data (first rows):")
print(df_encoded.head())

# Observation:
# - Categorical columns like 'sex', 'smoker', and 'region' have been encoded properly.

# Splitting data into features and target
X = df_encoded.drop('charges', axis=1)
y = df_encoded['charges']

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and training a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Model evaluation
print("\nModel evaluation:")
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Observation:
# - R² will tell us how well our model explains the variance.
# - If it’s below 0.7, we might consider feature engineering or using more complex models.

# Visualizing predictions vs actual values
plt.scatter(y_test, y_pred)
plt.xlabel("Actual values")
plt.ylabel("Predicted values")
plt.title("Predicted vs Actual")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')  # ideal fit line
plt.show()

# Observation:
# - If the points cluster close to the diagonal, the model is performing well.
# - Large spread means our model struggles with certain predictions (likely smokers vs non-smokers).
