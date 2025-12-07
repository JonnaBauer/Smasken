
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

######################


X = df[["weather_severity", "rush_hour_severity", "time_off", "season"]]

Y = df["increase_stock"]

random_states = range(0,10)  


all_results = pd.DataFrame() 
for rs in random_states:
    print(f"Running GridSearch for random_state = {rs}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=rs
    )
    
    k_amt = 71
    k_range = range(1,k_amt+1, 2)

    knn = KNeighborsClassifier()
    param_grid = {'n_neighbors': k_range}

    grid_search = GridSearchCV(
        knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)


    results = pd.DataFrame(grid_search.cv_results_)
    results["random_state"] = rs

    all_results = pd.concat([all_results, results], ignore_index=True)


# ------------------------------------------------------------------------------------
# Average the scores across all random_states
# ------------------------------------------------------------------------------------
avg_scores = (
    all_results.groupby("param_n_neighbors")
    .agg(
        mean_accuracy=("mean_test_score", "mean"),
        std_accuracy=("mean_test_score", "std"),
    )
    .reset_index()
)


final_ranking = avg_scores.sort_values(
    by="mean_accuracy",
    ascending=False
).reset_index(drop=True)



print("\n===== FINAL AVERAGED RANKING ACROSS RANDOM STATES =====\n")
print(final_ranking)