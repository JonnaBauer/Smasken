
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import pandas as pd
df = pd.read_csv('training_data_ht2025.csv')


df["increase_stock"] = df["increase_stock"].map({
    "high_bike_demand": 1,
    "low_bike_demand": 0
})


X = df.drop(columns=["increase_stock"])
Y = df["increase_stock"]

rs_amt = 5
k_amt = 101
k_range = range(1,k_amt+1, 2)

helper_range = range(0, len(k_range))


rs_range = range(1,rs_amt)

accuracy = np.zeros(len(helper_range))

for r in rs_range:
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=r
    )
    for help, k in zip(helper_range, k_range):
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)

        accuracy[help] += model.score(X_test, y_test)/len(helper_range)





plt.figure(figsize=(10, 6))
plt.plot(list(k_range), accuracy, marker='o')
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Average Accuracy")
plt.title("KNN Accuracy vs k (averaged over random states)")
plt.grid(True)

plt.tight_layout()
plt.show()