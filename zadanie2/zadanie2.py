import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
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

    residuals = y_test - y_pred

    if visualize:
        plt.figure(figsize=(12, 6))
        plot_tree(model, feature_names=X_train.columns, filled=True, max_depth=max_depth)
        plt.title(f"Vizualizácia Rozhodovacieho stromu (max_depth={max_depth})")
        plt.show()

        plt.figure(figsize=(8, 5))
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(0, color="red", linestyle="--")
        plt.title(f"Reziduály - Decision Tree (max_depth={max_depth})")
        plt.xlabel("Predikované hodnoty")
        plt.ylabel("Reziduály")
        plt.show()

    return model


#------------------------------ Stromovy subor AdaBoost --------------------------------------- #

def train_adaboost(input_path="data/z2_label.csv", random_state=5, n_estimators=100, depth=3, lr = 0.1, visualize=False):


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

    residuals = y_test - y_pred

    if visualize:

        plt.figure(figsize=(8, 5))
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(0, color="red", linestyle="--")
        plt.title(f"Reziduály - AdaBoost (max_depth={depth})")
        plt.xlabel("Predikované hodnoty")
        plt.ylabel("Reziduály")
        plt.show()

    return model

def evaluate_adaboost(input_path="data/z2_label.csv", random_state=5, lr=0.1):
    data = pd.read_csv(input_path, sep=";")

    train = data[data["set"] == "train"].copy()
    test = data[data["set"] == "test"].copy()

    X_train = train.drop(columns=["count", "set"])
    y_train = train["count"]
    X_test = test.drop(columns=["count", "set"])
    y_test = test["count"]

    results = []

    depths = [3, 5, 8, 10, 12, 15, 50]
    estimators = [50, 100, 200, 400, 800, 1600]

    for d in depths:
        for n in estimators:
            base_tree = DecisionTreeRegressor(max_depth=d, random_state=random_state)
            model = AdaBoostRegressor(
                estimator=base_tree,
                n_estimators=n,
                random_state=random_state,
                learning_rate=lr
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            results.append((d, n, r2))

    df = pd.DataFrame(results, columns=["Depth", "Estimators", "R2"])
    pivot = df.pivot(index="Estimators", columns="Depth", values="R2")



    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu")
    plt.title(f"Heatmap R² pre AdaBoost (learning_rate={lr})")
    plt.xlabel("Max depth")
    plt.ylabel("Estimators")
    plt.tight_layout()
    plt.show()

    return df

#------------------------------ SVM --------------------------------------- #

def train_svm_regressor(input_path="data/z2_label_norm.csv", kernel="rbf", C=100, epsilon=0.1, gamma = 0.01, visualize=False):

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

    residuals = y_test - y_pred

    if visualize:
        plt.figure(figsize=(8, 5))
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(0, color="red", linestyle="--")
        plt.title(f"Reziduály - Model SVM")
        plt.xlabel("Predikované hodnoty")
        plt.ylabel("Reziduály")
        plt.show()



    return model


#------------------------------ Korelacna matica --------------------------------------- #

def debug_top_correlations(df, top=10, method="pearson"):
    corr = df.corr(method=method, numeric_only=True)

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    pairs = (
        corr.where(~mask)
            .stack()
            .abs()
            .sort_values(ascending=False)
            .head(top)
    )
    print(f"Top {top} |{method}| korelácie:")
    for (a,b), v in pairs.items():
        print(f"{a:>18}  ~  {b:<18}  |r|={v:.3f}")
    print("Max |r|:", pairs.max() if len(pairs) else "N/A")

def train_adaboost_corr_reduced(
    input_path="data/z2_label_norm.csv",
    output_path="data/z2_label_corr_reduced.csv",
    corr_threshold=0.10,
    random_state=5,
    n_estimators=400,
    depth=10,
    lr=0.2,
    visualize=False
):

    data = pd.read_csv(input_path, sep=";")
    train = data[data["set"] == "train"].copy()
    test  = data[data["set"] == "test"].copy()

    X_train = train.drop(columns=["count", "set"])
    y_train = train["count"]
    X_test  = test.drop(columns=["count", "set"])
    y_test  = test["count"]



    corr = X_train.corr(numeric_only=True)
    to_drop = set()
    cols = corr.columns.tolist()
    for i in range(1, len(cols)):
        for j in range(i):
            if abs(corr.iloc[i, j]) > corr_threshold:
                to_drop.add(cols[i])

    kept = [c for c in X_train.columns if c not in to_drop]


    X_train_red = X_train[kept]
    X_test_red  = X_test[kept]

    train_red = X_train_red.copy()
    train_red["count"] = y_train.values
    train_red["set"] = "train"
    test_red = X_test_red.copy()
    test_red["count"] = y_test.values
    test_red["set"] = "test"
    pd.concat([train_red, test_red], ignore_index=True).to_csv(output_path, index=False, sep=";")

    base_tree = DecisionTreeRegressor(max_depth=depth, random_state=random_state)
    model = AdaBoostRegressor(
        estimator=base_tree,
        n_estimators=n_estimators,
        learning_rate=lr,
        random_state=random_state
    )
    model.fit(X_train_red, y_train)
    y_pred = model.predict(X_test_red)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)

    print("\n=== AdaBoost po redukcii korelácií ===")
    print(f"Ponechané príznaky: {len(kept)}  |  Zahodené: {len(to_drop)}")
    print(f"MAE: {mae:.4f} | MSE: {mse:.4f} | R²: {r2:.4f}")
    print("=======================================")

    if visualize:
        residuals = y_test - y_pred
        plt.figure(figsize=(8, 5))
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(0, color="red", linestyle="--")
        plt.title("Reziduály - AdaBoost (po korelačnej redukcii)")
        plt.xlabel("Predikované hodnoty")
        plt.ylabel("Reziduály")
        plt.tight_layout()
        plt.show()


    return model, {"MAE": mae, "MSE": mse, "R2": r2}, sorted(list(to_drop)), kept

#------------------------------ Feature importance --------------------------------------- #

def train_adaboost_top_importance(
    input_path="data/z2_label.csv",
    output_path="data/z2_label_topk_imp.csv",
    top_k=8,
    random_state=5, n_estimators=400, depth=10, lr=0.2, visualize=False
):

    data = pd.read_csv(input_path, sep=";")

    train = data[data["set"] == "train"].copy()
    test  = data[data["set"] == "test"].copy()

    X_train = train.drop(columns=["count", "set"])
    y_train = train["count"]
    X_test  = test.drop(columns=["count", "set"])
    y_test  = test["count"]

    base_tree = DecisionTreeRegressor(max_depth=depth, random_state=random_state)
    probe_model = AdaBoostRegressor(
        estimator=base_tree,
        n_estimators=n_estimators,
        learning_rate=lr,
        random_state=random_state
    )
    probe_model.fit(X_train, y_train)

    importances = pd.Series(probe_model.feature_importances_, index=X_train.columns)
    importances = importances.sort_values(ascending=False)
    # print("\nTop atribúty podľa dôležitosti:")
    # print(importances.head(top_k))

    top_features = importances.head(top_k).index.tolist()

    X_train_red = X_train[top_features]
    X_test_red  = X_test[top_features]

    model = AdaBoostRegressor(
        estimator=DecisionTreeRegressor(max_depth=depth, random_state=random_state),
        n_estimators=n_estimators,
        learning_rate=lr,
        random_state=random_state
    )
    model.fit(X_train_red, y_train)
    y_pred = model.predict(X_test_red)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)

    print("\n=== AdaBoost po redukcii podľa dôležitosti ===")
    print(f"Top {top_k} atribútov: {top_features}")
    print(f"MAE: {mae:.4f} | MSE: {mse:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}")
    print("==============================================")

    if visualize:
        residuals = y_test - y_pred
        plt.figure(figsize=(8, 5))
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(0, color="red", linestyle="--")
        plt.title(f"Reziduály - AdaBoost (Top {top_k} importance)")
        plt.xlabel("Predikované hodnoty")
        plt.ylabel("Reziduály")
        plt.show()

    train_red = X_train_red.copy()
    train_red["count"] = y_train.values
    train_red["set"] = "train"

    test_red = X_test_red.copy()
    test_red["count"] = y_test.values
    test_red["set"] = "test"

    reduced_data = pd.concat([train_red, test_red], ignore_index=True)
    reduced_data.to_csv(output_path, index=False, sep=";")

    return model, {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}, top_features


#------------------------------ PCA redukcia --------------------------------------- #

def train_adaboost_pca(
    input_path="data/z2_label_norm.csv",
    output_path="data/z2_label_pca_var.csv",
    variance=0.90,
    random_state=5, n_estimators=400, depth=10, lr=0.2, visualize=False
):

    data = pd.read_csv(input_path, sep=";")

    train = data[data["set"] == "train"].copy()
    test  = data[data["set"] == "test"].copy()

    X_train = train.drop(columns=["count", "set"])
    y_train = train["count"]
    X_test  = test.drop(columns=["count", "set"])
    y_test  = test["count"]

    pca = PCA(n_components=variance)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca  = pca.transform(X_test)

    print(f"[PCA] Počet komponentov: {pca.n_components_}")
    print(f"[PCA] Zachovaná variancia: {pca.explained_variance_ratio_.sum():.3f}")

    base_tree = DecisionTreeRegressor(max_depth=depth, random_state=random_state)
    model = AdaBoostRegressor(
        estimator=base_tree,
        n_estimators=n_estimators,
        learning_rate=lr,
        random_state=random_state
    )
    model.fit(X_train_pca, y_train)
    y_pred = model.predict(X_test_pca)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)

    print("\n=== AdaBoost po redukcii pomocou PCA ===")
    print(f"Komponenty: {pca.n_components_}")
    print(f"MAE: {mae:.4f} | MSE: {mse:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}")
    print("========================================")

    if visualize:
        residuals = y_test - y_pred
        plt.figure(figsize=(8, 5))
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(0, color="red", linestyle="--")
        plt.title("Reziduály - AdaBoost po PCA redukcii")
        plt.xlabel("Predikované hodnoty")
        plt.ylabel("Reziduály")
        plt.show()

    pca_cols = [f"PC{i+1}" for i in range(pca.n_components_)]
    train_red = pd.DataFrame(X_train_pca, columns=pca_cols)
    train_red["count"] = y_train.values
    train_red["set"] = "train"

    test_red = pd.DataFrame(X_test_pca, columns=pca_cols)
    test_red["count"] = y_test.values
    test_red["set"] = "test"

    reduced_data = pd.concat([train_red, test_red], ignore_index=True)
    reduced_data.to_csv(output_path, index=False, sep=";")

    return model, {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}, pca


#------------------------------ Vizualizacia --------------------------------------- #

def plot_three(input_path="data/z2_label.csv"):
    data = pd.read_csv(input_path, sep=";")

    fig = px.scatter_3d(
        data,
        x="hour",
        y="temperature",
        z="humidity",
        color="count",
        color_continuous_scale="plasma",
        title="Interaktívna 3D vizualizácia dát podľa počtu bicyklov",
        opacity=0.6
    )


    fig.update_traces(marker=dict(size=3, line=dict(width=0)))
    fig.update_layout(
        scene=dict(
            xaxis_title="Hodina",
            yaxis_title="Teplota [°C]",
            zaxis_title="Vlhkosť [%]",
            aspectmode="cube"
        ),
        coloraxis_colorbar=dict(title="Count"),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig.show()


def plot_pca_3d(input_path="data/z2_label_norm.csv"):
    data = pd.read_csv(input_path, sep=";")

    X = data.drop(columns=["count", "set"])
    y = data["count"]

    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)

    df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2", "PC3"])
    df_pca["count"] = y

    fig = px.scatter_3d(
        df_pca,
        x="PC1",
        y="PC2",
        z="PC3",
        color="count",
        color_continuous_scale="plasma",
        title="Redukcia dimenzie pomocou PCA (3 komponenty)",
        opacity=0.6
    )

    fig.update_traces(marker=dict(size=2))
    fig.update_layout(
        scene=dict(
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="PC3",
            aspectmode="cube"
        ),
        coloraxis_colorbar=dict(title="Count"),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig.show()

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
    plt.title(f"Graf dôležitosti parametrov")
    plt.xlabel("Význam parametra")
    plt.ylabel("Parameter")
    plt.tight_layout()
    plt.show()

    print("\nTop najdôležitejšie atribúty:")
    print(fi_top.to_string(index=False))



if __name__ == "__main__":

    data_clean = prepdata()
    encode_hot(data_clean)
    encode_label(data_clean)

    #plot_three("data/z2_label.csv")
    # plot_pca_3d(input_path="data/z2_label_norm.csv")

    # model_corr, metrics_corr, dropped, kept = train_adaboost_corr_reduced(
    #     input_path="data/z2_label_norm.csv",
    #     output_path="data/z2_label_corr_reduced.csv",
    #     corr_threshold=0.6,
    #     n_estimators=400, depth=10, lr=0.2,
    #     visualize=True
    # )

    # train_adaboost_top_importance(
    #     input_path="data/z2_label.csv",
    #     output_path="data/z2_label_topk_imp.csv",
    #     top_k=8,
    #     n_estimators=400,
    #     depth=10,
    #     lr=0.2,
    #     visualize=True
    # )

    train_adaboost_pca(
        input_path="data/z2_label_norm.csv",
        output_path="data/z2_label_pca_var.csv",
        variance=0.85,
        visualize=True
    )



    # for d in [3, 12]:
    #     train_decision_tree("data/z2_label.csv", max_depth=d, visualize=True)


    # for d in [3, 4, 5, 8,9, 10, 12, 15, 50]:
    #     train_adaboost("data/z2_label.csv", n_estimators=100, depth=d)

    # for n in [50, 100, 200, 400, 800, 1600]:
    #     for d in [3, 5, 8, 10, 12, 15, 50]:
    #         train_adaboost("data/z2_label.csv", n_estimators=n, depth=d, lr = 0.2, visualize= False)

    #best_model = train_adaboost("data/z2_label.csv", n_estimators=400, depth=10, lr=0.2, visualize= True)

    # evaluate_adaboost("data/z2_label.csv", lr=0.2)

    # best_model = train_adaboost(
    #     input_path="data/z2_label.csv",
    #     n_estimators=400,
    #     depth=10,
    #     random_state=5,
    #     lr=0.2
    # )
    #
    # plot_feature_importance(best_model, "data/z2_label.csv", top_n=10)

    # for C in [1, 10, 100, 200, 300]:
    #     for eps in [0.05, 0.1, 0.2]:
    #         for g in ["scale", "auto", 0.01, 0.05, 0.1, 0.5, 1]:
    #             train_svm_regressor(
    #                 "data/z2_label_norm.csv",
    #                 kernel="rbf",
    #                 C=C,
    #                 epsilon=eps,
    #                 gamma = g,
    #                 visualize=False
    #     )

    #train_svm_regressor("data/z2_label_norm.csv",kernel="rbf",C=300,epsilon=0.05,gamma=1,visualize=True)