import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from torch.optim.lr_scheduler import LambdaLR




#---------------------------- Data Cleaning ---------------------------------#
def prep_data():
    data = pd.read_csv("data/zadanie1-data.csv", sep=";")

    #print("Pôvodný počet riadkov:", len(data))

    data = data[(data["age"] >= 18) & (data["age"] <= 100)]

    data = data.drop(columns=["default"]) # vacsina zaznamov ma "no", iba jedno "yes"
    data = data.drop(columns=["euribor3m"]) # polovica null
    data = data.drop(columns=["pdays"]) # vacsina 999
    data = data.drop(columns=["duration"]) # ma byt iba na benchmark
    data = data.drop(columns=["previous"]) # vacsina 0.0

    data = data.dropna()
    #print("Počet riadkov po odstránení NaN:", len(data))

    Q1 = data["campaign"].quantile(0.25)
    Q3 = data["campaign"].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    data = data[(data["campaign"] >= lower_bound) & (data["campaign"] <= upper_bound)]
    #print(f"Po odstránení outlierov (IQR): {len(data)} riadkov zostáva.")

    data.to_csv("data/zadanie1_data_clean.csv", index=False, sep=";")

    #---------------------------- One-Hot Encoding ---------------------------------#

    y = data["subscribed"]
    X = data.drop(columns=["subscribed"])

    X_encoded = pd.get_dummies(X, drop_first=True)
    X_encoded = X_encoded.astype(float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)

    X_scaled = pd.DataFrame(X_scaled, columns=X_encoded.columns)

    encoded_data = pd.concat([X_scaled, y.reset_index(drop=True)], axis=1)

    encoded_data.to_csv("data/zadanie1_data_encoded.csv", index=False, sep=";")

    #---------------------------- Data split ---------------------------------#

    data = pd.read_csv("data/zadanie1_data_encoded.csv", sep=";")

    y = data["subscribed"]
    X = data.drop(columns=["subscribed"])


    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.10, random_state=5, stratify=y
    )


    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1111, random_state=42, stratify=y_temp
    )

    y_train = (y_train == "yes").astype(int)
    y_val = (y_val == "yes").astype(int)
    y_test = (y_test == "yes").astype(int)

    return X_train, X_val, X_test, y_train, y_val, y_test


#---------------------------- Sklearn ---------------------------------#

def sklearn_model(X_train, X_val, X_test, y_train, y_val, y_test):

    model = LogisticRegression(max_iter=1000, solver='liblinear')
    model.fit(X_train, y_train)


    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    print("Presnosť (tréning):", accuracy_score(y_train, y_train_pred))
    print("Presnosť (validačná):", accuracy_score(y_val, y_val_pred))
    print("Presnosť (testovacia):", accuracy_score(y_test, y_test_pred))


    cm = confusion_matrix(y_test, y_test_pred)
    classes = ["No", "Yes"]

    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Konfúzna matica - Testovacia množina")
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)


    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], "d"),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("Skutočná hodnota")
    plt.xlabel("Predikovaná hodnota")
    plt.tight_layout()
    plt.show()

    cm = confusion_matrix(y_train, y_train_pred)
    classes = ["No", "Yes"]

    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Konfúzna matica - Trénovacia množina")
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], "d"),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("Skutočná hodnota")
    plt.xlabel("Predikovaná hodnota")
    plt.tight_layout()
    plt.show()

# ---------------------------- EDA ---------------------------------#

def eda_1():

    data = pd.read_csv("data/zadanie1-data.csv", sep=";")


    bins = [15, 25, 35, 45, 55, 65, 75, 85, 95, data["age"].max() + 1]
    labels = ["15–25", "26–35", "36–45", "46–55", "56–65", "66–75", "76–85", "86–95", "95+"]
    data["age_group"] = pd.cut(data["age"], bins=bins, labels=labels, right=False)


    pivot = data.pivot_table(values="campaign", index="job", columns="age_group", aggfunc="mean", observed=False)


    plt.figure(figsize=(10, 7))
    ax = sns.heatmap(pivot, cmap="YlGnBu", annot=True, fmt=".2f", linewidths=0.5)

    plt.title("Priemerný počet kontaktov podľa práce a vekovej skupiny", fontsize=13, fontweight="bold")
    plt.xlabel("Veková skupina",  fontweight="bold")
    plt.ylabel("Typ práce",  fontweight="bold")


    colorbar = ax.collections[0].colorbar
    colorbar.set_label("Priemerný počet kontaktov", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.show()

def eda_2():
    data = pd.read_csv("data/zadanie1_data_clean.csv", sep=";")


    data["debt_level"] = (data["housing"].map({"yes": 1, "no": 0}) +
                          data["loan"].map({"yes": 1, "no": 0}))


    pivot = data.groupby("job")["debt_level"].mean().sort_values(ascending=False)


    plt.figure(figsize=(10, 6))
    sns.barplot(x=pivot.values, y=pivot.index, palette="YlGnBu")

    plt.title("Priemerná počet pôžičiek podľa zamestnania", fontsize=13, fontweight="bold")
    plt.xlabel("Priemerná úroveň zadĺženia (od 0 po 2 pôžičky)", fontweight="bold")
    plt.ylabel("Typ práce", fontweight="bold")
    plt.tight_layout()
    plt.show()


def eda_3():
    data = pd.read_csv("data/zadanie1-data.csv", sep=";")
    data = data[(data["age"] >= 18) & (data["age"] <= 100)]
    data = data[(data["duration"] >= 0)]
    Q1 = data["duration"].quantile(0.25)
    Q3 = data["duration"].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    data = data[(data["duration"] >= lower_bound) & (data["duration"] <= upper_bound)]

    plt.figure(figsize=(10, 6))
    plt.hexbin(
        data["age"],
        data["duration"],
        gridsize=40,
        cmap="viridis",
        mincnt=1
    )
    plt.colorbar(label="Počet klientov")
    plt.title("Pomer dĺžky hovoru k veku klienta", fontsize=13, fontweight="bold")
    plt.xlabel("Vek klienta", fontweight="bold")
    plt.ylabel("Dĺžka hovoru (v sekundách)", fontweight="bold")
    plt.tight_layout()
    plt.show()


def eda_4():
    data = pd.read_csv("data/zadanie1-data.csv", sep=";")
    data = data.dropna(subset=["month", "day_of_week", "marital", "subscribed"])


    data["subscribed"] = (data["subscribed"] == "yes").astype(int)


    month_order = ["jan", "feb", "mar", "apr", "may", "jun",
                   "jul", "aug", "sep", "oct", "nov", "dec"]
    day_order = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]

    data["day_of_week"] = pd.Categorical(data["day_of_week"], categories=day_order, ordered=True)


    pivot = data.pivot_table(
        values="subscribed",
        index="month",
        columns="day_of_week",
        aggfunc="mean"
    ).reindex(index=month_order, columns=day_order)


    for marital_status in data["marital"].unique():
        subset = data[data["marital"] == marital_status]
        pivot = subset.pivot_table(
            values="subscribed",
            index="month",
            columns="day_of_week",
            aggfunc="mean"
        ).reindex(month_order)


        plt.figure(figsize=(8, 6))
        sns.heatmap(pivot, cmap="YlGnBu", annot=True, fmt=".2f", linewidths=0.5, cbar_kws={'label': 'Pravdepodobnosť úspechu kampane'})
        plt.title(f"Úspešnosť kampane podľa mesiaca a dňa pre stav: {marital_status}", fontsize=12, fontweight="bold")
        plt.xlabel("Deň kontaktu", fontweight="bold")
        plt.ylabel("Mesiac", fontweight="bold")
        plt.tight_layout()
        plt.show()

def eda_5():
    data = pd.read_csv("data/zadanie1-data.csv", sep=";")
    data = data[(data["age"] >= 18) & (data["age"] <= 100)]


    plt.figure(figsize=(10, 6))
    sns.violinplot(
        data=data,
        x="education",
        y="age",
        hue="subscribed",
        split=True,
        inner="quart",
        palette="coolwarm"
    )

    plt.title("Úspešnosť kampane vzhľadom na vek a vzdelanie klienta", fontsize=13, fontweight="bold")
    plt.xlabel("Vzdelanie", fontweight="bold")
    plt.ylabel("Vek klienta", fontweight="bold")
    plt.legend(title="Subscribed")
    plt.tight_layout()
    plt.show()




#---------------------------- Neuronka ---------------------------------#

def neuronka(
    X_train, X_val, X_test, y_train, y_val, y_test,
    hidden_layers=[128, 64, 32],
    epochs=200,
    lr=0.001,
    batch_size=32,
    dropout_rate=0.0,
    wd = 0.0,
    best_val_loss = float("inf"),
    patience = 15,
    wait = 0,
    min_delta = 0.001
):

# ---------------------------- Tensor ---------------------------------#
    X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_val_t = torch.tensor(X_val.values, dtype=torch.float32)
    y_val_t = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_t = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)



    class NN_pokus(nn.Module):
        def __init__(self, input_dim, hidden_layers, dropout_rate):
            super().__init__()
            layers = []
            prev_dim = input_dim
            for hidden_dim in hidden_layers:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
                if dropout_rate > 0:
                    layers.append(nn.Dropout(dropout_rate))
                prev_dim = hidden_dim
            layers.append(nn.Linear(prev_dim, 1))
            layers.append(nn.Sigmoid())
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    model =NN_pokus(X_train.shape[1], hidden_layers, dropout_rate)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)

    train_losses, val_losses, test_losses = [], [], []
    train_accs, val_accs, test_accs = [], [], []

    #---------------------------- Trening ---------------------------------#
    for epoch in range(epochs):
        model.train()
        running_loss, correct_train = 0, 0
        for xb, yb in train_dl:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            correct_train += ((preds > 0.5).float() == yb).sum().item()

        train_loss = running_loss / len(train_dl.dataset)
        train_acc = correct_train / len(train_dl.dataset)

        #---------------------------- Validacia ---------------------------------#
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t)
            val_loss = criterion(val_preds, y_val_t).item()
            val_acc = ((val_preds > 0.5).float() == y_val_t).float().mean().item()

            test_preds = model(X_test_t)
            test_loss = criterion(test_preds, y_test_t).item()
            test_acc = ((test_preds > 0.5).float() == y_test_t).float().mean().item()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        test_accs.append(test_acc)

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"⏹ Early stopping at epoch {epoch + 1} (best val_loss={best_val_loss:.4f})")
                model.load_state_dict(best_model_state)
                break

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:03d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

    #---------------------------- Graf Loss ---------------------------------#
    plt.figure(figsize=(7, 5))
    plt.plot(train_losses, label="Train Loss", linewidth=2)
    plt.plot(val_losses, label="Validation Loss", linewidth=2)
    plt.title(f"Graf loss pri trénovaní", fontsize=12, fontweight='bold')
    plt.xlabel("Epocha")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    #---------------------------- Graf acc ---------------------------------#
    plt.figure(figsize=(7, 5))
    plt.plot(train_accs, label="Train Accuracy", linewidth=2)
    plt.plot(val_accs, label="Validation Accuracy", linewidth=2)
    plt.title(f"Graf accuracy pri trénovaní", fontsize=12, fontweight='bold')
    plt.xlabel("Epocha")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    #---------------------------- Vysledky ---------------------------------#
    model.eval()
    with torch.no_grad():
        train_preds_binary = (model(X_train_t) > 0.5).float()
        test_preds_binary = (model(X_test_t) > 0.5).float()

        train_final_acc = accuracy_score(y_train_t, train_preds_binary)
        test_final_acc = accuracy_score(y_test_t, test_preds_binary)

    print(f"\n{'=' * 70}")
    print(f"VÝSLEDKY TRÉNOVANIA")
    print(f"{'=' * 70}")
    print(f"Finálna Train Accuracy: {train_final_acc:.4f}")
    print(f"Finálna Test Accuracy:  {test_final_acc:.4f}")

    #---------------------------- Konfuzna matica train ---------------------------------#
    cm_train = confusion_matrix(y_train_t, train_preds_binary)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm_train, cmap="Blues")
    plt.title(f"Konfúzna matica - Trénovacia množina", fontweight='bold')
    plt.xlabel("Predikovaná")
    plt.ylabel("Skutočná")
    plt.xticks([0, 1], ["No", "Yes"])
    plt.yticks([0, 1], ["No", "Yes"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm_train[i, j],
                     ha="center", va="center",
                     color="white" if cm_train[i, j] > cm_train.max() / 2 else "black",
                     fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()

    #---------------------------- Konfuzna matica test ---------------------------------#
    cm_test = confusion_matrix(y_test_t, test_preds_binary)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm_test, cmap="Blues")
    plt.title(f"Konfúzna matica - Testovacia množina", fontweight='bold')
    plt.xlabel("Predikovaná")
    plt.ylabel("Skutočná")
    plt.xticks([0, 1], ["No", "Yes"])
    plt.yticks([0, 1], ["No", "Yes"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm_test[i, j],
                     ha="center", va="center",
                     color="white" if cm_test[i, j] > cm_test.max() / 2 else "black",
                     fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = prep_data()

    #eda_1()
    #eda_2()
    #eda_3()
    #eda_4()
    #eda_5()

    #sklearn_model(X_train, X_val, X_test, y_train, y_val, y_test)
    neuronka(
        X_train, X_val, X_test, y_train, y_val, y_test,
        hidden_layers=[256, 128, 64, 32],
        epochs=500,
        lr=0.00001 ,
        batch_size=128,
        dropout_rate=0.3,
        patience = 30,
        wd = 1e-5
    )


