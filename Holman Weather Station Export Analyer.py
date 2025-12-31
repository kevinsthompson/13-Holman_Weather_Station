# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
# ---

# %%
# Weather Data Analysis
# This notebook analyzes a dataset of weather observations using pandas and visualization libraries.

# %%
# 1. Notebook Setup
# Import libraries and configure display/plotting options.
import time
start_time = time.time()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Display library versions
print(f"pandas: {pd.__version__}")
print(f"numpy: {np.__version__}")
print(f"matplotlib: {plt.matplotlib.__version__}")
print(f"seaborn: {sns.__version__}")

# Set plotting style
sns.set(style="whitegrid")
# %matplotlib inline  # Uncomment if running in a notebook

end_time = time.time()
print(f"Cell executed in {time.time() - start_time:.2f} seconds.\n\n")
# %%


# 2. Data Loading
# Load the weather dataset into a pandas DataFrame.

import time
start = time.time()

file_path = 'Helios Weather Station20251230083828.xlsx'
excel_file = pd.ExcelFile(file_path)
dfs = []

for sheet_name in excel_file.sheet_names:
    df = pd.read_excel(excel_file, sheet_name=sheet_name)
    # Ensure the date/time column is parsed; replace 'date' with your actual column name if different
    print(f"Loading sheet: {sheet_name} with shape {df.shape}")
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    dfs.append(df)
    print(f"Loading sheet: {sheet_name} with shape {df.shape}")
    
# Merge all sheets on the 'date' column using outer join to preserve all timestamps
from functools import reduce
merged_df = reduce(lambda left, right: pd.merge(left, right, on='DateTime', how='outer'), dfs)

# Set the date column as the index
merged_df = merged_df.set_index('DateTime').sort_index()


end = time.time()
print(f"Loaded data in {end - start:.2f} seconds.")
print(f"DataFrame shape: {df.shape}")


# %%
# 3. Initial Data Inspection
# Preview the data and basic statistics.

display(df.head())
print("\nColumn names:", df.columns.tolist())
print("\nData types:")
print(df.dtypes)
print("\nInfo:")
df.info()
print("\nSummary statistics:")
display(df.describe())

# %%
# 4. Data Cleaning
# Handle missing values, fix data types, and remove duplicates.

print(f"Missing values before cleaning:\n{df.isnull().sum()}")
print(f"Duplicates before cleaning: {df.duplicated().sum()}")

# Example cleaning steps (customize as needed)
df = df.drop_duplicates()
df = df.dropna()

print(f"Missing values after cleaning:\n{df.isnull().sum()}")
print(f"Duplicates after cleaning: {df.duplicated().sum()}")
print(f"DataFrame shape after cleaning: {df.shape}")

# %%
# 5. Feature Engineering
# Create new columns or transform existing ones.

# Example: Convert temperature from Celsius to Fahrenheit if needed
if 'temperature_C' in df.columns:
    df['temperature_F'] = df['temperature_C'] * 9/5 + 32
    print("Added 'temperature_F' column.")

# Example: Parse date column
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
    print("Parsed 'date' column to datetime.")

display(df.head())

# %%
# 6. Exploratory Data Analysis (EDA)
# Visualize distributions, correlations, and trends.

# Example: Histogram of temperature
if 'temperature_C' in df.columns:
    df['temperature_C'].hist(bins=30)
    plt.title('Temperature Distribution (C)')
    plt.xlabel('Temperature (C)')
    plt.ylabel('Frequency')
    plt.show()

# Example: Correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# %%
# 7. Time Series Analysis
# Aggregate or resample data by time.

# Example: Daily average temperature
if 'date' in df.columns and 'temperature_C' in df.columns:
    daily_avg = df.set_index('date').resample('D')['temperature_C'].mean()
    daily_avg.plot(figsize=(12,4))
    plt.title('Daily Average Temperature (C)')
    plt.ylabel('Temperature (C)')
    plt.show()
    print(daily_avg.describe())

# %%
# 8. Outlier Detection
# Identify and visualize outliers.

# Example: Z-score method for temperature
from scipy.stats import zscore
if 'temperature_C' in df.columns:
    df['temp_zscore'] = zscore(df['temperature_C'])
    outliers = df[np.abs(df['temp_zscore']) > 3]
    print(f"Number of outliers: {outliers.shape[0]}")
    display(outliers[['date', 'temperature_C', 'temp_zscore']])

# %%
# 9. Graphing and Visualization
# Create final, publication-quality graphs.

# Example: Boxplot of temperature by month
if 'date' in df.columns and 'temperature_C' in df.columns:
    df['month'] = df['date'].dt.month
    plt.figure(figsize=(10,6))
    sns.boxplot(x='month', y='temperature_C', data=df)
    plt.title('Monthly Temperature Distribution')
    plt.xlabel('Month')
    plt.ylabel('Temperature (C)')
    plt.show()

# %%
# 10. Conclusions and Next Steps
# Summarize findings and suggest further analysis.

# Summary of key insights
from IPython.display import Markdown, display
def print_insights():
    display(Markdown('''
- Data cleaned and prepared for analysis.
- Key trends and outliers identified in temperature data.
- Visualizations created for further reporting.
- Next steps: deeper analysis, predictive modeling, or integration with other datasets.
'''))

print_insights()





