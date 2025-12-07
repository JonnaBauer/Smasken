
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from sklearn.metrics import accuracy_score


import pandas as pd
df = pd.read_csv('training_data_ht2025.csv')


df["increase_stock"] = df["increase_stock"].map({
    "high_bike_demand": 1,
    "low_bike_demand": 0
})

df["weather_severity"] = (
      df["precip"] * 2
    + df["snow"] * 3
    + (df["temp"] < 5).astype(int) * 1
    + (df["windspeed"] > 15).astype(int) * 2
    + (df["visibility"] < 5).astype(int) * 2
)

X = df[["weather_severity", "hour_of_day"]]

Y = df["increase_stock"]


X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

knn = KNeighborsClassifier()

k_amt = 101
k_range = range(1,k_amt+1, 2)


param_grid = {'n_neighbors': k_range}


grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get full CV results
results = pd.DataFrame(grid_search.cv_results_)

# Keep useful columns and sort by accuracy
ranking = results[["params", "mean_test_score", "std_test_score"]]
ranking = ranking.sort_values(by="mean_test_score", ascending=False).reset_index(drop=True)

# Display ranking
print("\nRanking of all k values (best → worst):\n")
print(ranking)