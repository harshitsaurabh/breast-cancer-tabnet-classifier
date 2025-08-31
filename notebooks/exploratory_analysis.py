# Exploratory Analysis for Breast Cancer Wisconsin Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =======================
# Load Data
# =======================
data = pd.read_csv('../data/breast-cancer-wisconsin-data_data.csv')
data.drop(columns={'id', 'Unnamed: 32'}, inplace=True)

# =======================
# Quick Look
# =======================
print("Shape of dataset:", data.shape)
print("Columns:", data.columns.tolist())
print("\nData Info:")
print(data.info())
print("\nMissing values per column:")
print(data.isnull().sum())
print("\nClass distribution:")
print(data['diagnosis'].value_counts())

# =======================
# Basic Statistics
# =======================
display(data.describe().T)

# =======================
# Class Balance Plot
# =======================
plt.figure(figsize=(6,4))
sns.countplot(x='diagnosis', data=data, palette='viridis')
plt.title("Class Distribution (M = Malignant, B = Benign)")
plt.xlabel("Diagnosis")
plt.ylabel("Count")
plt.show()

# =======================
# Correlation Heatmap
# =======================
plt.figure(figsize=(12,10))
corr = data.drop(columns='diagnosis').corr()
sns.heatmap(corr, cmap='coolwarm', center=0, cbar=True)
plt.title("Feature Correlation Heatmap")
plt.show()

# =======================
# Pairplot (sample of features for clarity)
# =======================
sampled_features = ['radius_mean', 'texture_mean', 'smoothness_mean', 'area_mean', 'diagnosis']
sns.pairplot(data[sampled_features], hue='diagnosis', diag_kind='kde', palette='Set2')
plt.suptitle("Pairplot of Selected Features", y=1.02)
plt.show()

# =======================
# Distribution of Key Features
# =======================
key_features = ['radius_mean', 'texture_mean', 'smoothness_mean', 'area_mean', 'compactness_mean']

plt.figure(figsize=(14,8))
for i, feature in enumerate(key_features):
    plt.subplot(2, 3, i+1)
    sns.histplot(data=data, x=feature, hue='diagnosis', bins=30, kde=True, palette='muted')
    plt.title(f"{feature} Distribution")
plt.tight_layout()
plt.show()

# =======================
# Boxplots for Feature Comparison
# =======================
plt.figure(figsize=(14,8))
for i, feature in enumerate(key_features):
    plt.subplot(2, 3, i+1)
    sns.boxplot(x='diagnosis', y=feature, data=data, palette='Set3')
    plt.title(f"{feature} by Diagnosis")
plt.tight_layout()
plt.show()

# =======================
# Correlation with Target
# =======================
# Convert diagnosis to numeric for correlation
data_numeric = data.copy()
data_numeric['diagnosis'] = data_numeric['diagnosis'].map({'M':1, 'B':0})
corr_with_target = data_numeric.corr()['diagnosis'].sort_values(ascending=False)
print("\nCorrelation of features with 'diagnosis':\n", corr_with_target)

plt.figure(figsize=(8,6))
corr_with_target.drop('diagnosis').plot(kind='bar')
plt.title("Feature Correlation with Diagnosis")
plt.ylabel("Correlation coefficient")
plt.show()
