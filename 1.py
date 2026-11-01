
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv("titanic.csv")

print("Original Dataset:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

le = LabelEncoder()

df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])

scaler = StandardScaler()

num_cols = ['Age', 'Fare', 'SibSp', 'Parch']

df[num_cols] = scaler.fit_transform(df[num_cols])

print("\nCleaned Dataset:")
print(df.head())

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

df.to_csv("cleaned_titanic.csv", index=False)

print("\nDataset prepared successfully!")
