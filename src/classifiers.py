import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score
from sklearn.model_selection import train_test_split


def average_accuracy(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    class_acc = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
    return class_acc.mean()


def evaluate_classifiers(
    X, Y,
    band_imp,
    band_sizes=(20, 25, 30),
    classifiers=("rf", "svc", "knn"),
    test_size=0.2,
    random_state=42
):
    results = []

    # 🔑 ranked_bands MUST already be band indices sorted by importance
    for k in band_sizes:
        ranked_bands=np.argsort(band_imp)
        selected_bands = ranked_bands[-k:][::-1]
        selected_bands = sorted(selected_bands.tolist())

        print(
    f"↗️ selected_bands (top-{k}) = {selected_bands}\n"
    f"ranked_band_indices (low→high) = {ranked_bands}"
)

        Xk = X[:, selected_bands]

        # ---- train / test split ----
        X_train, X_test, Y_train, Y_test = train_test_split(
            Xk, Y,
            test_size=test_size,
            stratify=Y,
            random_state=random_state
        )

        for clf_name in classifiers:
            start = time.time()

            if clf_name == "rf":
                clf = RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                   
                )

            elif clf_name == "svc":
                clf = SVC(
                    C=1000,
                    kernel="rbf",
                    gamma="scale",
                    class_weight="balanced"
                )

            elif clf_name == "knn":
                n_classes = len(np.unique(Y_train))
                clf = KNeighborsClassifier(
                    n_neighbors=min(5, n_classes)
                )

            else:
                continue

            # ---- train ----
            clf.fit(X_train, Y_train)

            # ---- test ----
            y_pred = clf.predict(X_test)

            oa = accuracy_score(Y_test, y_pred)
            aa = average_accuracy(Y_test, y_pred)
            kappa = cohen_kappa_score(Y_test, y_pred)

            elapsed = time.time() - start

            print(
                f"Classifier={clf_name}, "
                f"bands={k}, "
                f"OA={oa:.4f}, "
                f"AA={aa:.4f}, "
                f"Kappa={kappa:.4f}, "
                f"time={elapsed:.2f}s"
            )

            results.append({
                "classifier": clf_name,
                "num_bands": k,
                "OA": oa,
                "AA": aa,
                "Kappa": kappa,
                "classifier_time_sec": elapsed,
                "selected_bands": selected_bands
            })

    return results
