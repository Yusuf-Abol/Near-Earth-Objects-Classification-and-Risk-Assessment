"""
  Near-Earth Objects Classification and Risk Assessment: An Exploratory Study
  
  by Yusuf Abolarinwa
  
"""
# Import important librabies

import pandas as pd 
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


import shap
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import mannwhitneyu



# Load & inspect data
df_raw = pd.read_csv("C:\\Users\\DELL\\Desktop\\Hazardous NEOs\\data\\raw\\nearest-earth-objects(1910-2024).csv")
df_raw

# Counts the number of missing values in individual features
df_raw.isnull().sum()




"""
 Data Preprocessing
  
  - write a data wrangling function

"""



# Write a data wrangling function
def wrangle(filepath):
  
  # Read in the Neos data
  df_raw = pd.read_csv(filepath)
  
  # Compute medians of numeric columns only
  medians = df_raw.select_dtypes(include=["number"]).median()

  # Fill missing values with computed medians
  df_raw.fillna(medians, inplace=True)
  
  # Drop duplicates 
  df_raw.drop_duplicates(inplace=True)
   
  # Feature engineering: creating new ratio feature
  df_raw["velocity_to_diameter_ratio"]= (df_raw["relative_velocity"] / df_raw["estimated_diameter_min"])/1e7
  
  # Rename binary target, PHOs  with "0" and "1"
  df_raw["phos"] = (df_raw["is_hazardous"]).astype(int)
  
  # Drop the redundant columns 
  df_raw.drop(
    columns=[
    "name",
    "is_hazardous",
    "neo_id", 
    "orbiting_body",
    "estimated_diameter_min"
    ], 
    inplace=True)

  # Formatting the scientific notation in "miss_distance" column
  df_raw["miss_distance"] = (df_raw["miss_distance"])/1e7
  
  return df_raw

# apply the function
df_cleaned = wrangle("C:\\Users\\DELL\\Desktop\\Hazardous NEOs\\data\\raw\\nearest-earth-objects(1910-2024).csv")
df_cleaned.head()

# Post-Cleaning: Double-check on the data
df_cleaned.isnull().sum()




"""
 Exploratory Data Analysis (EDA)
 
  - Detect outliers
 
  - Data distribution
  
  - Scatter 
  - Correlation matrix
  
  - Features importance with SHAP
 
"""


# -- Detect outliers

# Ensure only 5 features are selected
features = df_cleaned.iloc[:, :-1].columns[:5]

# Define subplot layout dynamically
num_features = len(features)
fig, axes = plt.subplots(1, num_features, figsize=(5 * num_features, 5), squeeze=False)

# Loop through each feature and create a boxplot
for i, feature in enumerate(features):
    sns.boxplot(x="phos", y=feature, data=df_cleaned, ax=axes[0, i], 
                boxprops={"facecolor": "mediumslateblue", "edgecolor": "black"}, 
                medianprops={"color": "black"}, 
                whiskerprops={"color": "black"}, 
                capprops={"color": "black"})
    
    axes[0, i].set_title(f"Boxplot of {feature}")

plt.tight_layout()
plt.show()


# -- Resolving Outliers issue


# Feature Summary Statistics
df_cleaned.describe().iloc[1:]


print(np.mean(df_cleaned['absolute_magnitude'])+ 2*np.std(df_cleaned['absolute_magnitude']))
print(np.mean(df_cleaned['relative_velocity'])+ 2*np.std(df_cleaned['relative_velocity']))
print(np.mean(df_cleaned['velocity_to_diameter_ratio'])+ 2*np.std(df_cleaned['velocity_to_diameter_ratio']))
print(np.mean(df_cleaned['estimated_diameter_max'])+ 2*np.std(df_cleaned['estimated_diameter_max']))


# Create a copy of the cleaned dataframe
df0 = df_cleaned.copy()

# Use `df0` instead of `df_cleaned` in the cutoffs condition
cutoffs = (
    (df0["absolute_magnitude"] < 28.7) &
    (df0["relative_velocity"] < 100_000) &
    (df0["velocity_to_diameter_ratio"] < 0.7) &
    (df0["estimated_diameter_max"] < 1.7)
)

df0 = df0[cutoffs]

# -- Outliers issue resolved 

# Ensure only 5 features are selected
features0 = df0.iloc[:, :-1].columns[:5]

# Define subplot layout dynamically
num_features = len(features0)
fig, axes = plt.subplots(1, num_features, figsize=(5 * num_features, 5), squeeze=False)

# Loop through each feature and create a boxplot
for i, feature in enumerate(features0):
    sns.boxplot(x="phos", y=feature, data=df0, ax=axes[0, i], 
                boxprops={"facecolor": "mediumslateblue", "edgecolor": "black"}, 
                medianprops={"color": "black"}, 
                whiskerprops={"color": "black"}, 
                capprops={"color": "black"})
    
    axes[0, i].set_title(f"Boxplot of {feature}")

plt.tight_layout()
plt.show()


# -- Data distribution

df = df0.copy()
df.shape

# (a) Distribution of Potentially Hazardous Objects (PHOs)

df["phos"].value_counts(normalize=True).plot(
    kind="pie", autopct='%1.1f%%', startangle=90, 
    title="Phos & non-Phos Asteroids", colors=["mediumslateblue", "lightcoral"]);

# (b) Estimated diameter & Velocity ratio

# estimated diameter distribution
sns.histplot(df["estimated_diameter_max"], bins=30, kde=True, color="mediumslateblue")
plt.title("Distribution of Asteriods' Estimated Diameter")
plt.show();

# velocity-diameter distribution
sns.histplot(df["velocity_to_diameter_ratio"], bins=30, kde=True, color="mediumslateblue")
plt.title("Distribution of Asteriods' velocity_to_diameter_ratio")
plt.show();


# - Scatter plots

# Scatter plot of Estimated Diameter vs Absolute Magnitude
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x='velocity_to_diameter_ratio', 
    y='estimated_diameter_max', 
    hue='phos', 
    data=df, 
    palette=['mediumslateblue', 'lightcoral']  # Custom colors
)
plt.title('Estimated Diameter vs Absolute Magnitude')
plt.show()

# Scatter plot of Estimated Diameter vs Absolute Magnitude
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x='absolute_magnitude', 
    y='estimated_diameter_max', 
    hue='phos', 
    data=df, 
    palette=['mediumslateblue', 'lightcoral']  
)
plt.title('Estimated Diameter vs Absolute Magnitude')
plt.show()


# - Correlation matrix

from matplotlib.colors import LinearSegmentedColormap

# Custom colormap
cmap = LinearSegmentedColormap.from_list("custom", ["mediumslateblue", "white", "lightcoral"])

# Plot heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.drop(columns="phos").corr(), annot=True, cmap=cmap, fmt=".2f", linewidths=0.5, square=True)

plt.title("Correlation Matrix of Features", fontsize=12)
plt.xticks(rotation=45, ha="right")
plt.show()


# - Features importance

# Define features and target
X = df.drop(columns=["phos"])  # Features
y = df["phos"]  # Binary target for Potentially Hazardous Objects or Asteroids

# Reduce dataset size (randomly select 500 samples)
X_sample = X.sample(500, random_state=42)
y_sample = y.loc[X_sample.index]  # Ensure matching labels

# Train Random Forest Model on the smaller dataset
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_sample, y_sample)

# Compute SHAP values using TreeExplainer
explainer = shap.TreeExplainer(rf)
shap_values = explainer(X_sample)  

# Extract correct SHAP values for binary classification
shap_values_class_1 = shap_values[:, :, 1]  # Extract last dimension (class 1)

# Plot Summary
shap.summary_plot(shap_values_class_1, X_sample)



"""
Hypothesis test & test of significance

  - On asteroids diameter abd velocity
  
  - On astroids brightness and nearness to Earth
  
  - Cliff's Delta effect size for NEOs features
 
"""

# - On asteroids diameter abd velocity

# List of features to test
diameter_velocity = ['estimated_diameter_max', 'relative_velocity']

# Initialize lists to store results
feature_names = []
u_statistics = []
p_values = []
effect_sizes = []

# Loop through each feature and perform Mann-Whitney U test
for i in diameter_velocity:
    hazardous = df[df['phos'] == 1][i]
    non_hazardous = df[df['phos'] == 0][i]
    
    # Mann-Whitney U Test (alternative='greater' checks if hazardous NEOs have larger values)
    u_stat, p_val = mannwhitneyu(hazardous, non_hazardous, alternative='greater')
    
    # Calculate Cliff's Delta (effect size)
    n_hazardous = len(hazardous)
    n_non_hazardous = len(non_hazardous)
    cliff_delta = (2 * u_stat - n_hazardous * n_non_hazardous) / (n_hazardous * n_non_hazardous)
    
    # Store results in lists
    feature_names.append(i)
    u_statistics.append(u_stat)
    p_values.append(p_val)
    effect_sizes.append(cliff_delta)

# Create DataFrame
mann_whit_results = pd.DataFrame({
    'Feature': feature_names,
    'U Statistic': u_statistics,
    'P-value': p_values,
    'Effect Size (Cliff\'s Delta)': effect_sizes
})

# Display the DataFrame
print(mann_whit_results)


# - On astroids brightness and nearness to Earth

# List of features to test
abs_mag_miss_dist = ['absolute_magnitude','miss_distance']

# Initialize lists to store results
feature_names = []
u_statistics = []
p_values = []
effect_sizes = []

# Loop through each feature and perform Mann-Whitney U test
for i in abs_mag_miss_dist:
    hazardous = df[df['phos'] == 1][i]
    non_hazardous = df[df['phos'] == 0][i]
    
    # Mann-Whitney U Test (alternative='less' checks if hazardous NEOs have smaller values)
    u_stat, p_val = mannwhitneyu(hazardous, non_hazardous, alternative='less')
    
    # Calculate Cliff's Delta (effect size)
    n_hazardous = len(hazardous)
    n_non_hazardous = len(non_hazardous)
    cliff_delta = (2 * u_stat - n_hazardous * n_non_hazardous) / (n_hazardous * n_non_hazardous)
    
    # Store results in lists
    feature_names.append(i)
    u_statistics.append(u_stat)
    p_values.append(p_val)
    effect_sizes.append(cliff_delta)

# Create DataFrame with results
mann_whit_results = pd.DataFrame({
    'Feature': feature_names,
    'U Statistic': u_statistics,
    'P-value': p_values,
    'Effect Size (Cliff\'s Delta)': effect_sizes
})

# Display the results
print(mann_whit_results)


# - Cliff's Delta effect size for NEOs features

# Data dictionary for effect sizes and features
effect_dict = {'Feature': ['estimated_diameter_max', 'relative_velocity', 'absolute_magnitude', 'miss_distance'],
        'Effect Size': [0.656539, 0.297492, -0.656539, -0.021729]}

# Create DataFrame and plot
fig = px.bar(pd.DataFrame(effect_dict), x='Feature', y='Effect Size', 
             title="Effect Sizes (Cliff's Delta)", color_discrete_sequence=['mediumslateblue'])

# Show the plot
fig.show()

