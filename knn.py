from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

import sklearn.metrics as skl_m

import sklearn.pipeline as skl_pl
import sklearn.preprocessing as skl_pp


import pandas as pd
from features import load_data, add_features, get_X_y


df = load_data("training_data_ht2025.csv")
df = add_features(df)

X, y = get_X_y(df)

beta_score = 2
rs = 2

X_train, X_val_test, y_train, y_val_test = train_test_split(
    X, y, test_size=0.4, random_state=rs, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_val_test, y_val_test, test_size=0.5, random_state=rs, stratify=y_val_test
)

all_results = pd.DataFrame() 


k_amt = 101
k_range = range(1,k_amt+1, 2)


pipe = skl_pl.Pipeline([
    ('scaler', skl_pp.StandardScaler()),
    ('knn', KNeighborsClassifier())
])
f2_scorer = skl_m.make_scorer(skl_m.fbeta_score, beta=beta_score)
param_grid = {'knn__n_neighbors': k_range,
                'knn__weights': ['uniform', 'distance'],
                'knn__metric': ['euclidean','manhattan','chebyshev']}
grid_search = GridSearchCV(
    pipe, param_grid, cv=5, scoring=f2_scorer, n_jobs=-1
)
grid_search.fit(X_train, y_train)


results = pd.DataFrame(grid_search.cv_results_)
results["random_state"] = rs

all_results = pd.concat([all_results, results], ignore_index=True)
from sklearn.metrics import classification_report, confusion_matrix, f1_score

best_knn = grid_search.best_estimator_
y_pred = best_knn.predict(X_test)



# Results -------------------------------------

print("\n===== Confusion Matrix =====\n")
print(confusion_matrix(y_test, y_pred))


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



print("\n===== Ranking =====\n")
print(final_ranking)

print("\n===== F2 beta =====\n")
print(skl_m.fbeta_score(y_test, y_pred, beta=beta_score))