import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load Titanic dataset for cleaning
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# 1. Missing Values Handling
print("Missing values before cleaning:\n", df.isnull().sum())

# Age me median fill karenge aur Cabin drop karenge
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['Cabin'], inplace=True)

print("\nMissing values after cleaning:\n", df.isnull().sum())

# 2. Visualizations & Outlier Check
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Correlation Heatmap
sns.heatmap(df[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].corr(), annot=True, cmap='coolwarm', ax=axes[0])
axes[0].set_title('Correlation Heatmap')

# Outliers check in Fare
sns.boxplot(x=df['Fare'], ax=axes[1], color='coral')
axes[1].set_title('Fare Outliers')

plt.tight_layout()
plt.show()