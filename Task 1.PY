import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Dataset Load
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)

# 2. Dataset Overview
print("--- FIRST 5 ROWS ---")
print(df.head())

print("\n--- DATASET INFO ---")
print(df.info())

print("\n--- STATISTICAL SUMMARY ---")
print(df.describe())

# 3. Visualization
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.histplot(df['sepal_length'], kde=True, color='teal')
plt.title('Sepal Length Distribution')

plt.subplot(1, 2, 2)
sns.countplot(x='species', data=df, palette='viridis')
plt.title('Species Count')

plt.tight_layout()
plt.show()