import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv("data/zadanie1-data.csv", sep=";")

#---------------------------- Data Cleaning ---------------------------------#

print("Pôvodný počet riadkov:", len(data))

data = data[(data["age"] >= 18) & (data["age"] <= 100)]
print("Po odstránení nereálnych vekov:", len(data))

data = data.drop(columns=["default"]) # vacsina zaznamov ma "no", iba jedno "yes"
data = data.drop(columns=["euribor3m"]) # polovica null
data = data.drop(columns=["pdays"]) # vacsina 999
data = data.drop(columns=["duration"]) # ma byt iba na benchmark
data = data.drop(columns=["previous"]) # vacsina 0.0

data = data.dropna()
print("Počet riadkov po odstránení NaN:", len(data))

Q1 = data["campaign"].quantile(0.25)
Q3 = data["campaign"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

data = data[(data["campaign"] >= lower_bound) & (data["campaign"] <= upper_bound)]
print(f"Po odstránení outlierov (IQR): {len(data)} riadkov zostáva.")

data.to_csv("data/zadanie1_data_clean.csv", index=False, sep=";")

#---------------------------- One-Hot Encoding ---------------------------------#

y = data["subscribed"]
X = data.drop(columns=["subscribed"])


X_encoded = pd.get_dummies(X, drop_first=True)

# spojíme späť cieľovú premennú
encoded_data = pd.concat([X_encoded, y], axis=1)

# uloženie zakódovaných dát
encoded_data.to_csv("data/zadanie1_data_encoded.csv", index=False, sep=";")

#---------------------------- Data split ---------------------------------#

data = pd.read_csv("data/zadanie1_data_encoded.csv", sep=";")

y = data["subscribed"]
X = data.drop(columns=["subscribed"])


X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)


X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp
)


#---------------------------- Sklearn ---------------------------------#

y_train = (y_train == "yes").astype(int)
y_val = (y_val == "yes").astype(int)
y_test = (y_test == "yes").astype(int)


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





