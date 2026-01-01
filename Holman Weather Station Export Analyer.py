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
import glob
import os
import matplotlib.dates as mdates

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

# Display library versions
print(f"pandas: {pd.__version__}")
print(f"numpy: {np.__version__}")
print(f"matplotlib: {plt.matplotlib.__version__}")
print(f"seaborn: {sns.__version__}")

# Set plotting style
sns.set(style="whitegrid")
%matplotlib inline  

print(f"\n\n-=- -=- -=-\nCell executed in {time.time() - start_time:.3f} seconds.\n-=- -=- -=-\n\n")


# %%
# 2. Data Loading
# Load the weather dataset into a pandas DataFrame.
start_time = time.time()

# Find all files in the current directory that start with "Helios Weather Station" and end with .xlsx
file_pattern = os.path.join(os.getcwd(), "Helios Weather Station*.xlsx")
file_list = glob.glob(file_pattern)

all_dfs = []

print(f"\n")
for file_path in file_list:
    # Read all sheets at once into a dict of DataFrames
    all_sheets = pd.read_excel(file_path, sheet_name=None)
    dfs = []
    for sheet_name, df in all_sheets.items():
        if not df.empty and 'DateTime' in df.columns:
            df = df.copy()
            # Custom parsing to handle both 'YYYY/MM/DD HH:MM:SS' and 'YYYYMMDD'
            df['DateTime'] = df['DateTime'].astype(str).str.strip()
            df['DateTime'] = df['DateTime'].apply(
                lambda x: pd.to_datetime(x, errors='coerce') if '/' in x or '-' in x else pd.to_datetime(x, format='%Y%m%d', errors='coerce')
            )
            df = df.set_index('DateTime')
            dfs.append(df)
    if dfs:
        from functools import reduce
        combined_df = reduce(lambda left, right: left.join(right, how='outer'), dfs)
        combined_df = combined_df.sort_index()
        all_dfs.append(combined_df)
        print(f"Added {combined_df.shape[0]} rows from {os.path.basename(file_path)}")
    else:
        print(f"No valid data found in {file_path}")

if all_dfs:
    # Concatenate all combined DataFrames from all files
    final_df = pd.concat(all_dfs, axis=0)
    final_df = final_df.sort_index()
    print(f"DataFrame shape: {final_df.shape}")
else:
    print("No valid data found in any worksheet of any file.")

df = final_df if all_dfs else pd.DataFrame()
print("DataFrame indexes:", df.index)

end = time.time()
print(f"\n\n-=- -=- -=-\nCell executed in {time.time() - start_time:.3f} seconds.\n-=- -=- -=-\n\n")


# %%
# 3. Initial Data Inspection
# Preview the data and basic statistics.

print("\n\n\n")
# display(df.head())
# print("\nColumn names:", df.columns.tolist())
# print("\nData types:")
# print(df.dtypes)
# print("\nInfo:")
df.info()
# print("\nSummary statistics:")
# display(df.describe())

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
# if 'temperature_C' in df.columns:
    # df['temperature_F'] = df['temperature_C'] * 9/5 + 32
    # print("Added 'temperature_F' column.")

# Example: Parse date column
# if 'date' in df.columns:
    # df['date'] = pd.to_datetime(df['date'])
    # print("Parsed 'date' column to datetime.")

# display(df.head())


# %%
# 6. Exploratory Data Analysis (EDA)
# Visualize distributions, correlations, and trends.


# A. Area bands for Min–Max with distinct colors for Indoor and Outdoor Temperatures
# Single full-figure area plot for Temperatures with averages, labels, title, legend
fig, ax = plt.subplots(figsize=(18, 12))
ax.fill_between(plot_df.index, plot_df['Indoor Temperature Min'], plot_df['Indoor Temperature Max'],
                alpha=0.35, color='tab:blue', label='Indoor Min–Max')
ax.fill_between(plot_df.index, plot_df['Outdoor Temperature Min'], plot_df['Outdoor Temperature Max'],
                alpha=0.35, color='tab:orange', label='Outdoor Min–Max')

# Average lines
ax.plot(plot_df.index, plot_df['Indoor Temperature Avg'], color='blue', linewidth=2, label='Indoor Avg')
ax.plot(plot_df.index, plot_df['Outdoor Temperature Avg'], color='orange', linewidth=2, label='Outdoor Avg')

# Labels, title, legend
ax.set_xlabel('Date')
ax.set_ylabel('Temperature (°C)')
ax.set_title('Plot A. Indoor vs Outdoor Temperature: Min–Max Areas and Daily Averages')
ax.legend()

# X-axis labels for each data point
ax.set_xticks(plot_df.index)
ax.set_xticklabels(plot_df.index.strftime('%Y-%m-%d'), rotation=45, ha='right')
ax.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()


# B. Area bands for Min–Max with distinct colors for Indoor and Outdoor Humidity
# Single full-figure area plot for Humidity with averages, labels, title, legend
fig, ax = plt.subplots(figsize=(18, 12))
ax.fill_between(humidity_df.index, humidity_df['Indoor Humidity Min'], humidity_df['Indoor Humidity Max'],
                alpha=0.35, color='tab:blue', label='Indoor Humidity Min–Max')
ax.fill_between(humidity_df.index, humidity_df['Outdoor Humidity Min'], humidity_df['Outdoor Humidity Max'],
                alpha=0.35, color='tab:orange', label='Outdoor Humidity Min–Max')

# Average lines
ax.plot(humidity_df.index, humidity_df['Indoor Humidity Avg'], color='blue', linewidth=2, label='Indoor Humidity Avg')
ax.plot(humidity_df.index, humidity_df['Outdoor Humidity Avg'], color='orange', linewidth=2, label='Outdoor Humidity Avg')

# Labels, title, legend
ax.set_xlabel('Date')
ax.set_ylabel('Relative Humidity (%)')
ax.set_title('Plot B. Daily Indoor vs Outdoor Humidity: Min–Max Areas and Averages')
ax.legend()

# X-axis labels for each data point
ax.set_xticks(humidity_df.index)
ax.set_xticklabels(humidity_df.index.strftime('%Y-%m-%d'), rotation=45, ha='right')

ax.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()


# C. Area bands for Min–Max with distinct colors for Indoor Temperatures and Humidity
# Single full-figure area plot for Temperatures with averages, labels, title, legend

fig = plt.figure(figsize=(18, 12))
ax_temp = fig.add_subplot(2, 1, 1)
ax_hum = fig.add_subplot(2, 1, 2, sharex=ax_temp)

# Temperature (top)
ax_temp.fill_between(plot_df.index, plot_df['Indoor Temperature Min'], plot_df['Indoor Temperature Max'],
                     alpha=0.35, color='tab:blue', label='Indoor Min–Max')
ax_temp.plot(plot_df.index, plot_df['Indoor Temperature Avg'],
             color='blue', linewidth=2, label='Indoor Avg')
ax_temp.set_ylabel('Temperature (°C)')
ax_temp.grid(True, linestyle='--', alpha=0.3)

# Humidity (bottom)
ax_hum.fill_between(humidity_df.index, humidity_df['Indoor Humidity Min'], humidity_df['Indoor Humidity Max'],
                    alpha=0.35, color='tab:orange', label='Indoor Humidity Min–Max')
ax_hum.plot(humidity_df.index, humidity_df['Indoor Humidity Avg'],
            color='orange', linewidth=2, label='Indoor Humidity Avg')
ax_hum.set_xlabel('Date')
ax_hum.set_ylabel('Relative Humidity (%)')
ax_hum.grid(True, linestyle='--', alpha=0.3)

# Title and x-ticks (match original formatting on bottom axis)
fig.suptitle('Plot C. Indoor Temperature and Humidity: Min–Max Areas and Daily Averages')
ax_temp.tick_params(labelbottom=False)
ax_hum.set_xticks(plot_df.index)
ax_hum.set_xticklabels(plot_df.index.strftime('%Y-%m-%d'), rotation=45, ha='right')

# Combined legend (same labels/colors)
h1, l1 = ax_temp.get_legend_handles_labels()
h2, l2 = ax_hum.get_legend_handles_labels()
fig.legend(h1 + h2 , l1 + l2, loc='upper right', ncol=1)
plt.tight_layout(rect=[0, 0, 1, 0.96])


# D. Area bands for Min–Max with distinct colors for Outdoor Temperature and Humidity
fig = plt.figure(figsize=(18, 12))
ax_temp = fig.add_subplot(2, 1, 1)
ax_hum = fig.add_subplot(2, 1, 2, sharex=ax_temp)

# Outdoor Temperature (top)
ax_temp.fill_between(plot_df.index, plot_df['Outdoor Temperature Min'], plot_df['Outdoor Temperature Max'],
                     alpha=0.35, color='tab:blue', label='Outdoor Min–Max')
ax_temp.plot(plot_df.index, plot_df['Outdoor Temperature Avg'],
             color='blue', linewidth=2, label='Outdoor Avg')
ax_temp.set_ylabel('Temperature (°C)')
ax_temp.grid(True, linestyle='--', alpha=0.3)

# Outdoor Humidity (bottom)
ax_hum.fill_between(humidity_df.index, humidity_df['Outdoor Humidity Min'], humidity_df['Outdoor Humidity Max'],
                    alpha=0.35, color='tab:orange', label='Outdoor Humidity Min–Max')
ax_hum.plot(humidity_df.index, humidity_df['Outdoor Humidity Avg'],
            color='orange', linewidth=2, label='Outdoor Humidity Avg')
ax_hum.set_xlabel('Date')
ax_hum.set_ylabel('Relative Humidity (%)')
ax_hum.grid(True, linestyle='--', alpha=0.3)

# Title and x-ticks (match formatting on bottom axis)
fig.suptitle('Plot D. Outdoor Temperature and Humidity: Min–Max Areas and Daily Averages')
ax_temp.tick_params(labelbottom=False)
ax_hum.set_xticks(plot_df.index)
ax_hum.set_xticklabels(plot_df.index.strftime('%Y-%m-%d'), rotation=45, ha='right')

# Combined legend
h1, l1 = ax_temp.get_legend_handles_labels()
h2, l2 = ax_hum.get_legend_handles_labels()
fig.legend(h1 + h2 , l1 + l2, loc='upper right', ncol=1)
plt.tight_layout(rect=[0, 0, 1, 0.96])

# %%






# %%
# B . Time Series Line Plot: Humidity Trends
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['Indoor Humidity Avg'], label='Indoor Humidity Avg')
plt.plot(df.index, df['Outdoor Humidity Avg'], label='Outdoor Humidity Avg')
plt.xlabel('Date')
plt.ylabel('Humidity (%)')
plt.title('Indoor vs. Outdoor Average Humidity Over Time')
plt.legend()
plt.tight_layout()
plt.show()

# %%
# 3. Bar Plot: Rainfall Total per Period
plt.figure(figsize=(12, 5))
plt.bar(df.index, df['Rainfall Total'])
plt.xlabel('Date')
plt.ylabel('Rainfall (mm)')
plt.title('Rainfall Total per Period')
plt.tight_layout()
plt.show()

# 4. Box Plot: Temperature Distribution
plt.figure(figsize=(8, 6))
sns.boxplot(data=df[['Indoor Temperature Avg', 'Outdoor Temperature Avg']])
plt.ylabel('Temperature (°C)')
plt.title('Distribution of Indoor and Outdoor Average Temperatures')
plt.tight_layout()
plt.show()

# 5. Scatter Plot: Outdoor Temperature vs. Outdoor Humidity
plt.figure(figsize=(8, 6))
plt.scatter(df['Outdoor Temperature Avg'], df['Outdoor Humidity Avg'])
plt.xlabel('Outdoor Temperature Avg (°C)')
plt.ylabel('Outdoor Humidity Avg (%)')
plt.title('Outdoor Temperature vs. Outdoor Humidity')
plt.tight_layout()
plt.show()

# 6. Line Plot: Barometric Pressure Over Time
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['Barometric Pressure Avg'])
plt.xlabel('Date')
plt.ylabel('Barometric Pressure (hPa)')
plt.title('Barometric Pressure Over Time')
plt.tight_layout()
plt.show()

# 7. Line Plot: Wind Speed and Gusts
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['Wind Speed Avg'], label='Wind Speed Avg')
plt.plot(df.index, df['Wind Gust Avg'], label='Wind Gust Avg')
plt.xlabel('Date')
plt.ylabel('Wind Speed (km/h)')
plt.title('Wind Speed and Gusts Over Time')
plt.legend()
plt.tight_layout()
plt.show()

# 8. Heatmap: Correlation Matrix
plt.figure(figsize=(14, 10))
corr = df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', annot_kws={"size": 10}, linewidths=0.5)
plt.title('Correlation Heatmap of Weather Variables')
plt.tight_layout()
plt.show()

# 9. Grouped Bar Plot: Max vs. Min Temperatures
temp_df = df[['Indoor Temperature Max', 'Indoor Temperature Min', 'Outdoor Temperature Max', 'Outdoor Temperature Min']]
temp_df_plot = temp_df.reset_index(drop=True)
temp_df_plot.plot(kind='bar', figsize=(14, 6))
plt.xlabel('Period')
plt.ylabel('Temperature (°C)')
plt.title('Max vs. Min Temperatures (Indoor & Outdoor)')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

# 10. Box Plot: Wind Speed and Gusts
plt.figure(figsize=(8, 6))
sns.boxplot(data=df[['Wind Speed Avg', 'Wind Gust Avg']])
plt.ylabel('Wind Speed (km/h)')
plt.title('Distribution of Wind Speed and Gusts')
plt.tight_layout()
plt.show()























# # Correlation heatmap
# import seaborn as sns
# plt.figure(figsize=(18, 14))
# sns.heatmap(
#     combined_df.corr(),
#     annot=True,
#     fmt=".2f",
#     cmap='coolwarm',
#     annot_kws={"size": 12},
#     linewidths=0.5,
#     cbar_kws={"shrink": 0.8, "aspect": 30}
# )
# plt.title('Correlation Heatmap', fontsize=18)
# plt.xticks(fontsize=12, rotation=45, ha='right')
# plt.yticks(fontsize=12)
# plt.tight_layout()
# plt.show()




# # 7. Time Series Analysis
# # Aggregate or resample data by time.

# # Example: Daily average temperature
# if 'date' in df.columns and 'temperature_C' in df.columns:
#     daily_avg = df.set_index('date').resample('D')['temperature_C'].mean()
#     daily_avg.plot(figsize=(12,4))
#     plt.title('Daily Average Temperature (C)')
#     plt.ylabel('Temperature (C)')
#     plt.show()
#     print(daily_avg.describe())

# # %%
# # 8. Outlier Detection
# # Identify and visualize outliers.

# # Example: Z-score method for temperature
# from scipy.stats import zscore
# if 'temperature_C' in df.columns:
#     df['temp_zscore'] = zscore(df['temperature_C'])
#     outliers = df[np.abs(df['temp_zscore']) > 3]
#     print(f"Number of outliers: {outliers.shape[0]}")
#     display(outliers[['date', 'temperature_C', 'temp_zscore']])

# # %%
# # 9. Graphing and Visualization
# # Create final, publication-quality graphs.

# # Example: Boxplot of temperature by month
# if 'date' in df.columns and 'temperature_C' in df.columns:
#     df['month'] = df['date'].dt.month
#     plt.figure(figsize=(10,6))
#     sns.boxplot(x='month', y='temperature_C', data=df)
#     plt.title('Monthly Temperature Distribution')
#     plt.xlabel('Month')
#     plt.ylabel('Temperature (C)')
#     plt.show()

# # %%
# # 10. Conclusions and Next Steps
# # Summarize findings and suggest further analysis.

# # Summary of key insights
# from IPython.display import Markdown, display
# def print_insights():
#     display(Markdown('''
# - Data cleaned and prepared for analysis.
# - Key trends and outliers identified in temperature data.
# - Visualizations created for further reporting.
# - Next steps: deeper analysis, predictive modeling, or integration with other datasets.
# '''))

# print_insights()





