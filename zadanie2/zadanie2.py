import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#------------------------------ Uprava dat --------------------------------------------------- #

def prepdata(save_path="data/z2_data_1y_clean.csv"):
    data = pd.read_csv("data/z2_data_1y.csv", sep=";")

    print("Pocet pred odstranenim :", len(data))

    data = data.drop(columns=["instant"])
    data = data.drop(columns=["date"])
    data = data[data["humidity"] >= 0]
    data = data.dropna()

    print("Pocet po odstraneni :", len(data))

    data = data.drop_duplicates()

    print("Pocet po odstraneni duplikatov :", len(data))

    data.to_csv(save_path, index=False, sep=";")
    return data

#------------------------------ Zakodovanie, rozdelenie dat a normalizacia dat ------------------ #

def encode_hot(data):

    data_hot = pd.get_dummies(data, columns=["weather"], prefix="weather")

    data_hot.to_csv("data/z2_data_1y_encode_hot.csv", index=False, sep=";")

    split_data("data/z2_data_1y_encode_hot.csv", "data/z2_hot.csv")
    normalize_data("data/z2_hot.csv", "data/z2_hot_norm.csv")

def encode_label(data):

    le = LabelEncoder()
    data_label = data.copy()

    data_label["weather"] = le.fit_transform(data["weather"])

    data_label.to_csv("data/z2_data_1y_encode_label.csv", index=False, sep=";")

    split_data("data/z2_data_1y_encode_label.csv", "data/z2_label.csv")
    normalize_data("data/z2_label.csv", "data/z2_label_norm.csv")



def split_data(input_path, output_path="data/z2_data_split.csv", test_size=0.2, random_state=5):
    data = pd.read_csv(input_path, sep=";")

    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)


    train_data["set"] = "train"
    test_data["set"] = "test"


    combined = pd.concat([train_data, test_data], ignore_index=True)
    combined.to_csv(output_path, index=False, sep=";")


def normalize_data(input_path, output_path):

    data = pd.read_csv(input_path, sep=";")

    train_data = data[data["set"] == "train"].copy()
    test_data = data[data["set"] == "test"].copy()


    y_train = train_data["count"]
    y_test = test_data["count"]

    X_train = train_data.drop(columns=["count", "set"])
    X_test = test_data.drop(columns=["count", "set"])


    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)


    train_scaled = X_train.copy()
    train_scaled["count"] = y_train.values
    train_scaled["set"] = "train"

    test_scaled = X_test.copy()
    test_scaled["count"] = y_test.values
    test_scaled["set"] = "test"

    combined_scaled = pd.concat([train_scaled, test_scaled], ignore_index=True)
    combined_scaled.to_csv(output_path, index=False, sep=";")


#------------------------------ Rozhodovaci strom --------------------------------------- #

def train_decision_tree(input_path="data/z2_label.csv", random_state=5, max_depth=None, visualize=False):


    data = pd.read_csv(input_path, sep=";")

    train = data[data["set"] == "train"].copy()
    test = data[data["set"] == "test"].copy()

    X_train = train.drop(columns=["count", "set"])
    y_train = train["count"]
    X_test = test.drop(columns=["count", "set"])
    y_test = test["count"]


    model = DecisionTreeRegressor(random_state=random_state, max_depth=max_depth)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("=== Decision Tree Regressor ===")
    print(f"Max depth: {max_depth}")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"R²:  {r2:.4f}")
    print("===============================")


    if visualize:
        plt.figure(figsize=(12, 6))
        plot_tree(model, feature_names=X_train.columns, filled=True, max_depth=max_depth)
        plt.title(f"Vizualizácia Rozhodovacieho stromu (max_depth={max_depth})")
        plt.show()

    return model

#------------------------------ Stromovy subor AdaBoost --------------------------------------- #

def train_adaboost(input_path="data/z2_label.csv", random_state=5, n_estimators=100, depth=3, lr = 0.1):


    data = pd.read_csv(input_path, sep=";")

    train = data[data["set"] == "train"].copy()
    test = data[data["set"] == "test"].copy()

    X_train = train.drop(columns=["count", "set"])
    y_train = train["count"]
    X_test = test.drop(columns=["count", "set"])
    y_test = test["count"]

    base_tree = DecisionTreeRegressor(max_depth=depth, random_state=random_state)

    model = AdaBoostRegressor(
        estimator=base_tree,
        n_estimators=n_estimators,
        random_state=random_state,
        learning_rate=lr
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("=== AdaBoost Regressor ===")
    print(f"Estimators: {n_estimators}")
    print(f"Base tree depth: {depth}")
    print(f"Learning rate: {lr}")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"R²:  {r2:.4f}")
    print("===============================")

    return model

#------------------------------ SVM --------------------------------------- #

def train_svm_regressor(input_path="data/z2_label_norm.csv", kernel="rbf", C=100, epsilon=0.1, gamma = 0.01):

    data = pd.read_csv(input_path, sep=";")

    train = data[data["set"] == "train"].copy()
    test = data[data["set"] == "test"].copy()

    X_train = train.drop(columns=["count", "set"])
    y_train = train["count"]
    X_test = test.drop(columns=["count", "set"])
    y_test = test["count"]

    model = SVR(kernel=kernel, C=C, epsilon=epsilon, gamma = gamma)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("=== Support Vector Regressor (SVM) ===")
    print(f"Kernel: {kernel}, C={C}, epsilon={epsilon}, gamma={gamma}")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"R²:  {r2:.4f}")
    print("===============================")

    return model





def plot_feature_importance(model, input_path="data/z2_label.csv", top_n=10):
    data = pd.read_csv(input_path, sep=";")
    features = [col for col in data.columns if col not in ["count", "set"]]

    importances = model.feature_importances_

    fi_df = pd.DataFrame({
        "Feature": features,
        "Importance": importances
    }).sort_values("Importance", ascending=False)

    fi_top = fi_df.head(top_n)


    plt.figure(figsize=(10, 6))
    plt.barh(fi_top["Feature"], fi_top["Importance"], color="steelblue")
    plt.gca().invert_yaxis()
    plt.title(f"Top {top_n} najdôležitejších vstupných parametrov (AdaBoost)")
    plt.xlabel("Význam parametra")
    plt.tight_layout()
    plt.show()

    print("\nTop najdôležitejšie atribúty:")
    print(fi_top.to_string(index=False))



if __name__ == "__main__":

    data_clean = prepdata()
    encode_hot(data_clean)
    encode_label(data_clean)

    # for d in [3, 4, 5, 8,9, 10, 12, 15, 50]:
    #     train_decision_tree("data/z2_label.csv", max_depth=d, visualize=True)


    # for d in [3, 4, 5, 8,9, 10, 12, 15, 50]:
    #     train_adaboost("data/z2_label.csv", n_estimators=100, depth=d)

    # for n in [50, 100, 200, 400, 800, 1600, 2000]:
    #     for d in [3, 4, 5, 8, 9, 10, 12, 15, 50]:
    #         for l in [0.05, 0.1, 0.2]:
    #             train_adaboost("data/z2_label.csv", n_estimators=n, depth=d, lr = l)

    # best_model = train_adaboost(
    #     input_path="data/z2_label.csv",
    #     n_estimators=400,
    #     depth=10,
    #     random_state=5,
    #     lr=0.2
    # )
    #
    # plot_feature_importance(best_model, "data/z2_label.csv", top_n=10)

    for C in [1, 10, 100, 200, 300]:
        for eps in [0.05, 0.1, 0.2]:
            for g in ["scale", "auto", 0.01, 0.05, 0.1, 0.5, 1]:
                train_svm_regressor(
                    "data/z2_label_norm.csv",
                    kernel="rbf",
                    C=C,
                    epsilon=eps,
                    gamma = g
        )

