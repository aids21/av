
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("iris.csv")

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

start = time.time()
model1 = LogisticRegression(max_iter=200)
model1.fit(X_train_scaled, y_train)
pred1 = model1.predict(X_test_scaled)
time1 = time.time() - start

accuracy1 = accuracy_score(y_test, pred1)

start = time.time()
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

model2 = LogisticRegression(max_iter=200)
model2.fit(X_train_pca, y_train)
pred2 = model2.predict(X_test_pca)
time2 = time.time() - start

accuracy2 = accuracy_score(y_test, pred2)

print("Original Features:", X.shape[1])
print("Reduced Features:", X_train_pca.shape[1])

print("\nAccuracy Before PCA:", accuracy1 * 100, "%")
print("Accuracy After PCA:", accuracy2 * 100, "%")

print("\nTime Before PCA:", time1, "seconds")
print("Time After PCA:", time2, "seconds")

print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)

print("\nTotal Variance Retained:",
      sum(pca.explained_variance_ratio_) * 100, "%")
