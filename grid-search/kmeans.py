from matplotlib import pyplot as plt

from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.metrics import *
from sklearn.model_selection import cross_val_predict

from in_out import *
from preprocessing import *
from models import *

X_train, y_train = load_shrunk_data()

pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95)),
        ("kmeans", KMeans(n_clusters=200)),
        (
            "svc",
            LogisticRegression(
                C=0.2, l1_ratio=0.3, penalty="elasticnet", solver="saga", max_iter=5000
            ),
        ),
    ]
)

y_pred = cross_val_predict(pipeline, X_train, y_train, cv=3, method="decision_function")
print("Cross Validation")
print("AUC :")
print(roc_auc_score(y_train, y_pred))
print("Accuracy :")
print(accuracy_score(y_train, np.where(y_pred > 0.5, 1, 0)))
print("Precision :")
print(precision_score(y_train, np.where(y_pred > 0.5, 1, 0)))
print("Balanced accuracy :")
print(balanced_accuracy_score(y_train, np.where(y_pred > 0.5, 1, 0)))

# pipeline = Pipeline(
#     [
#         ("scaler", StandardScaler()),
#         ("pca", PCA(n_components=0.95)),
#     ]
# )
#
# X_train = pipeline.fit_transform(X_train)
#
# X_axis = list(range(1, 20))
#
# kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(X_train) for k in X_axis]
# inertias = [model.inertia_ for model in kmeans_per_k]
#
#
# plt.figure(figsize=(8, 3.5))
# plt.plot(X_axis, inertias, "bo-")
# plt.xlabel("$k$", fontsize=14)
# plt.ylabel("Inertia", fontsize=14)
# plt.show()
