import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

df = pd.read_csv("Mall_Customers.csv")
X = df.drop(columns=["CustomerID"], errors="ignore")
X = X.select_dtypes(include=np.number)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
inertias = []
K_range = range(1, 11)
for k in K_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
plt.figure()
plt.plot(list(K_range), inertias, marker="o")
plt.xticks(list(K_range))
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia (within-cluster SSE)")
plt.title("Elbow Method")
plt.tight_layout()
plt.savefig("elbow_method.png")
plt.close()
p1 = np.array([1, inertias[0]])
p2 = np.array([len(inertias), inertias[-1]])
distances = []
for i, sse in enumerate(inertias, start=1):
    p = np.array([i, sse])
    distance = np.abs(np.cross(p2 - p1, p1 - p)) / np.linalg.norm(p2 - p1)
    distances.append(distance)
best_k = int(np.argmax(distances) + 1 if len(distances) > 0 else 2)
print(f"Best K (elbow): {best_k}")

kmeans = KMeans(n_clusters=max(best_k, 2), n_init=10, random_state=42)
labels = kmeans.fit_predict(X_scaled)
X_with_labels = X.copy()
X_with_labels["cluster"] = labels
X_with_labels.to_csv("clustered_customers.csv", index=False)
feat_x, feat_y = None, None
preferred_pairs = [
    ("Annual Income (k$)", "Spending Score (1-100)"),
    ("Annual Income (k$)", "Age"),
    ("Age", "Spending Score (1-100)")
]
for a, b in preferred_pairs:
    if a in X.columns and b in X.columns:
        feat_x, feat_y = a, b
        break
if feat_x is None:
    feat_x, feat_y = X.columns[:2]
plt.figure()
for c in range(kmeans.n_clusters):
    idx = labels == c
    plt.scatter(X.loc[idx, feat_x], X.loc[idx, feat_y], s=22, label=f"Cluster {c}")
plt.title("K-Means clusters")
plt.xlabel(feat_x)
plt.ylabel(feat_y)
plt.legend()
plt.tight_layout()
plt.savefig("clusters.png")
plt.close()

sil = silhouette_score(X_scaled, labels)
print(f"Silhouette Score: {sil:.3f}")