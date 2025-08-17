## Mall Customer Segmentation using K-Means

📌 Overview

This project performs customer segmentation on the Mall Customers dataset using the K-Means clustering algorithm. The clustering process helps in identifying distinct customer groups based on their spending patterns and income, useful for targeted marketing strategies.

----------------------------------

⚙️ Libraries Used

- **pandas → For dataset handling and preprocessing.**

- **numpy → For numerical computations.**

- **matplotlib → For plotting elbow curve and cluster visualization.**

- **sklearn (scikit-learn) → For K-Means clustering and silhouette score evaluation.**

------------------------

## ⚙️ Installation

- **pip install pandas numpy matplotlib scikit-learn**

-------------------------

## 🛠 Changes Made

- **Dataset Loading – Updated the code to load the Mall_Customers dataset using pd.read_csv("Mall_Customers.csv").**

- **Feature Selection – Chose Annual Income and Spending Score columns as features for clustering.**

- **Elbow Method – Implemented to automatically find the best value of K.**

- **Cluster Visualization – Added plots to clearly display customer groups with different colors.***

- **Evaluation – Included Silhouette Score to check the quality of the clusters.**

------------------------------------

## 📌 Conclusion  

- **The Elbow Method suggested the optimal number of clusters as K = 4.**

- **The K-Means algorithm grouped customers into 4 distinct clusters based on Annual Income and Spending Score.**

- **The Silhouette Score = 0.404, which indicates that the clusters are reasonably well-formed but have some overlap.**

- **The visualization clearly shows customer groups separated by spending habits and income levels.**
