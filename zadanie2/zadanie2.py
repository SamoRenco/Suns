import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

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


def encode_hot(data):

    data_hot = pd.get_dummies(data, columns=["weather"], prefix="weather")

    data_hot.to_csv("data/z2_data_1y_encode_hot.csv", index=False, sep=";")

def encode_label(data):

    le = LabelEncoder()
    data_label = data.copy()

    data_label["weather"] = le.fit_transform(data["weather"])

    data_label.to_csv("data/z2_data_1y_encode_label.csv", index=False, sep=";")


def split_data(input_path, output_path="data/z2_data_split.csv", test_size=0.2, random_state=5):
    data = pd.read_csv(input_path, sep=";")

    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)


    train_data["set"] = "train"
    test_data["set"] = "test"


    combined = pd.concat([train_data, test_data], ignore_index=True)
    combined.to_csv(output_path, index=False, sep=";")




if __name__ == "__main__":

    data_clean = prepdata()
    encode_hot(data_clean)
    encode_label(data_clean)

    split_data("data/z2_data_1y_encode_label.csv", "data/z2_label.csv")
    split_data("data/z2_data_1y_encode_hot.csv", "data/z2_hot.csv")