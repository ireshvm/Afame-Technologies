import pandas as pd

# Load the HR dataset
file_path = 'path/to/HR-Dataset.csv'  # Specify the path to your HR dataset
df = pd.read_csv('HR Data.csv')

# Display initial dataset information
print("Initial Dataset:")
print(df.head())
print(df.info())

# Data Cleansing Steps
# Remove unnecessary columns
columns_to_drop = ['EmployeeNumber', 'Over18', 'StandardHours', 'EmployeeCount']  # Specify columns to drop
df_cleaned = df.drop(columns_to_drop, axis=1)  # Drop specified columns

# Rename columns for clarity
new_column_names = {
    'Attrition': 'Attrition',
    'BusinessTravel': 'TravelFrequency',
    'Department': 'Department',
    'EducationField': 'EducationField',
    'JobRole': 'JobRole',
    'MaritalStatus': 'MaritalStatus',
    'Gender': 'Gender',
    'MonthlyIncome': 'MonthlyIncome',
    'TotalWorkingYears': 'TotalWorkingYears',
    'YearsAtCompany': 'YearsAtCompany',
    'YearsInCurrentRole': 'YearsInCurrentRole',
    'YearsSinceLastPromotion': 'YearsSinceLastPromotion',
    'YearsWithCurrManager': 'YearsWithCurrentManager'
}
df_cleaned = df_cleaned.rename(columns=new_column_names)

# Drop duplicate rows (if any)
df_cleaned = df_cleaned.drop_duplicates()

# Sanitize specific columns
df_cleaned['Gender'] = df_cleaned['Gender'].map({'Female': 'Female', 'Male': 'Male'})  # Sanitize 'Gender' column

# Eliminate NaN values
df_cleaned = df_cleaned.dropna()

# Display cleaned dataset information
print("\nCleaned Dataset:")
print(df_cleaned.head())
print(df_cleaned.info())
