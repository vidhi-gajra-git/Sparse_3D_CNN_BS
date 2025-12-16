import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score

def average_accuracy(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    class_acc = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
    return class_acc.mean()

def evaluate_classifiers(
    X, Y,
    ranked_bands,
    band_sizes=(20, 25, 30),
    classifiers=("rf", "svc", "knn")
):
    results = []

    for k in band_sizes:
        selected_bands = np.argsort(ranked_bands.tolist())
        print(f"↗️ selected_bands = {selected_bands[:k]}")
        Xk = X[:, selected_bands[:k]]

        for clf_name in classifiers:
            start = time.time()

            if clf_name == "rf":
                clf = RandomForestClassifier(
                    n_estimators=200,
                    random_state=42
                )
            elif clf_name == "svc":
                clf = SVC(
                    C=1000,
                    kernel="rbf",
                    gamma="scaled"
                )
            elif clf_name == "knn":
                clf = KNeighborsClassifier(
                    n_neighbors=Y.nunique()
                )
            else:
                continue

            clf.fit(Xk, Y)
            y_pred = clf.predict(Xk)

            oa = accuracy_score(Y, y_pred)
            aa = average_accuracy(Y, y_pred)
            kappa = cohen_kappa_score(Y, y_pred)

            results.append({
                "classifier": clf_name,
                "num_bands": k,
                "OA": oa,
                "AA": aa,
                "Kappa": kappa,
                "classifier_time_sec": time.time() - start,
                "selected_bands": selected_bands
            })

    return results
