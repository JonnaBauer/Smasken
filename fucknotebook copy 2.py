from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from sklearn.metrics import accuracy_score

import sklearn.pipeline as skl_pl
import sklearn.preprocessing as skl_pp


import pandas as pd
df = pd.read_csv('training_data_ht2025.csv')




df["increase_stock"] = df["increase_stock"].map({
    "high_bike_demand": 1,
    "low_bike_demand": 0
})

#######################
# new metrics:

df["weather_severity"] = (
      df["precip"] * 2
    + df["snow"] * 3
    + (df["temp"] < 5).astype(int) * 1
    + (df["windspeed"] > 15).astype(int) * 2
    + (df["visibility"] < 5).astype(int) * 2
)
df["rush_hour_severity"] = ((df["hour_of_day"].between(7, 9)) | 
                   (df["hour_of_day"].between(16, 18))).astype(int)
df["time_off"] = ((1 - df["weekday"]) | 
                   (df["holiday"]))

conditions = [ #based on the "volume of high bike demand by hour" graph
    df["hour_of_day"].between(8, 14),
    df["hour_of_day"].between(15, 19),
]
values = [1, 2]
df["rush_hour_severity"] = np.select(conditions, values, default=0)


def season(month):
    if month in [12, 1, 2]:  #winter
        return 0
    elif month in [3, 4, 5]: #spring
        return 1
    elif month in [6, 7, 8]: #summer
        return 2
    else:                    #autumn
        return 3

df["season"] = df["month"].apply(season)

df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)

df["temp_extreme"] = ((df["temp"] < 0) | (df["temp"] > 30)).astype(int)  
df["precip_intensity"] = df["precip"] ** 2

df["day_type"] = np.where(df["holiday"] == 1, 2, np.where(df["weekday"] == 0, 1, 0))  # 0=weekday, 1=weekend, 2=holiday

######################


X = df[["weather_severity", "rush_hour_severity", "hour_sin", "hour_cos", "time_off", "month_sin", "month_cos"]]

Y = df["increase_stock"]

random_states = range(0,1)  

rs = 2
all_results = pd.DataFrame() 

print(f"Running GridSearch for random_state = {rs}")

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.5, random_state=rs
)
# Train - Validation - Test Split
import sklearn.model_selection as skl_ms

X_train, X_val_test, y_train, y_val_test = skl_ms.train_test_split(
    X,
    Y,
    test_size=0.4,
    random_state=rs,
    stratify=df['increase_stock']
)
X_val, X_test, y_val, y_test = skl_ms.train_test_split(
    X_val_test,
    y_val_test,
    test_size=0.5,
    random_state=rs,
    stratify=y_val_test
)


k_amt = 71
k_range = range(1,k_amt+1, 2)


pipe = skl_pl.Pipeline([
    ('scaler', skl_pp.StandardScaler()),
    ('knn', KNeighborsClassifier())
])
param_grid = {'knn__n_neighbors': k_range,
                'knn__weights': ['uniform', 'distance'],
                'knn__metric': ['euclidean','manhattan','minkowski']}
grid_search = GridSearchCV(
    pipe, param_grid, cv=5, scoring='f1', n_jobs=-1
)
grid_search.fit(X_train, y_train)


results = pd.DataFrame(grid_search.cv_results_)
results["random_state"] = rs

all_results = pd.concat([all_results, results], ignore_index=True)
from sklearn.metrics import classification_report, confusion_matrix, f1_score

best_knn = grid_search.best_estimator_
y_pred = best_knn.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print("F1 Score (for positive class):", f1_score(y_test, y_pred))
print("Macro F1 Score:", f1_score(y_test, y_pred, average='macro'))


# ------------------------------------------------------------------------------------
# Average the scores across all random_states
# ------------------------------------------------------------------------------------
avg_scores = (
    all_results.groupby(["param_knn__n_neighbors", "param_knn__weights", "param_knn__metric"])
    .agg(
        mean_accuracy=("mean_test_score", "mean"),
    )
    .reset_index()
)

final_ranking = avg_scores.sort_values(
    by="mean_accuracy",
    ascending=False
).reset_index(drop=True)



print("\n===== FINAL AVERAGED RANKING ACROSS RANDOM STATES =====\n")
print(final_ranking)