import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image


# -------------------- KONŠTANTY -------------------------------------------#

DATA_DIR = "data"
SUBFOLDERS = ["01", "04", "07", "10"]
IMG_SIZE = (224, 224)
BATCH_SIZE = 64
NUM_EPOCHS = 120
LR = 1e-4
DROPOUT_RATE = 0.4
EARLY_STOPPING_PATIENCE = 8


# -------------------- CUDA / GPU DETEKCIA --------------------------------#


def print_device_info():
    print("Start timestamp:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU detected:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("GPU NOT detected, using CPU.")
    return device


# -------------------- TRANSFER LEARNING – FEATURE EXTRACTION -------------------- #

mobilenet_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_mobilenet(device):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier = nn.Identity()
    model.to(device)
    model.eval()
    return model


def extract_feature_vector(model, img_path, device):
    img = Image.open(img_path).convert("RGB")
    img = mobilenet_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = model(img).cpu().numpy().flatten()

    return feat


# -------------------- PyTorch DATASET + DATALOADER ------------------------#

# Augmentácia len pre trénovaciu množinu
train_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])

# Validačná a testovacia množina – bez augmentácie
val_test_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
])


class LinearRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 128),
            nn.ReLU(),

            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)


class FeatureDataset(Dataset):
    def __init__(self, df):
        feature_cols = [c for c in df.columns if c.startswith("f")]
        self.X = df[feature_cols].values
        self.y = df["Irradiance"].values

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return X, y


class IrradianceDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        y = torch.tensor(row["Irradiance"], dtype=torch.float32)
        return img, y


# -------------------- EVALUÁCIA MODELU -----------------------------------#


def evaluate_loader(model, loader, device):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device).unsqueeze(1)

            preds = model(X)
            y_true.extend(y.cpu().numpy().flatten())
            y_pred.extend(preds.cpu().numpy().flatten())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return mse, mae, rmse, r2, y_true, y_pred


# -------------------- CNN MODEL (PyTorch) ---------------------------------#



class CNNRegressor(nn.Module):
    def __init__(self, dropout_rate=DROPOUT_RATE):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # nn.Conv2d(128, 256, kernel_size=3, padding=1),
            # nn.BatchNorm2d(256),
            # nn.ReLU(),
            # nn.MaxPool2d(2),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 512),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = self.regressor(x)
        return x


# -------------------- LOAD CSV -------------------------------------------#


def load_all_csv(data_dir=DATA_DIR, subfolders=SUBFOLDERS):
    dfs = []

    for folder in subfolders:
        csv_path = os.path.join(data_dir, folder, "out_data.csv")
        if not os.path.exists(csv_path):
            print(f"CSV nenájdené: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        df["day_folder"] = folder


        df["DateTime"] = df["DateTime"].str.split("#").str[0]

        # PARSOVANIE
        df["DateTime"] = pd.to_datetime(
            df["DateTime"],
            format="%m/%d/%Y %H:%M:%S.%f",
            errors="coerce"
        )

        dfs.append(df)

    full_df = pd.concat(dfs, ignore_index=True)

    #print("CSV rows BEFORE dropna:", len(full_df))
    full_df = full_df.dropna(subset=["DateTime"])
    #print("CSV rows AFTER dropna:", len(full_df))

    return full_df


# -------------------- CSV KEY ---------------------------------------------#


# def add_csv_match_keys(df):
#     df["DateTime"] = df["DateTime"].str.split("#").str[0]
#     df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
#     return df


# -------------------- EXTRACT TIMESTAMP -----------------------------------#


def extract_timestamp_from_filename(fn):
    match = re.search(r"(\d{8}_\d{2}-\d{2}-\d{2})", fn)
    return match.group(1) if match else None


def match_images_to_csv(df):
    matched = []

    # set pre rýchle vyhľadávanie
    csv_names = set(df["PictureName"].values)

    for folder in SUBFOLDERS:
        img_dir = os.path.join(DATA_DIR, folder, "original")
        if not os.path.exists(img_dir):
            continue

        for filename in os.listdir(img_dir):
            if not filename.lower().endswith(".png"):
                continue

            if filename in csv_names:
                row = df[df["PictureName"] == filename].iloc[0]
                matched.append({
                    "image_path": os.path.join(img_dir, filename),
                    "Irradiance": row["Irradiance"]
                })
            else:
                print(f"❗ Not found in CSV: {filename}")

    #print(f"\nMatched {len(matched)} images using PictureName exact matching.\n")
    return pd.DataFrame(matched)




def extract_features_dataframe(matched_df, device):
    print("Loading MobileNetV2 for feature extraction...")
    model = load_mobilenet(device)

    feature_list = []
    feature_vectors = []

    for idx, row in matched_df.iterrows():
        img_path = row["image_path"]
        feat = extract_feature_vector(model, img_path, device)

        feature_list.append({
            "image_path": img_path,
            "Irradiance": row["Irradiance"]
        })
        feature_vectors.append(feat)

        if idx % 200 == 0:
            print(f"Processed {idx}/{len(matched_df)} images...")

    df_features = pd.DataFrame(feature_list)
    df_vecs = pd.DataFrame(feature_vectors, columns=[f"f{i}" for i in range(len(feature_vectors[0]))])

    final_df = pd.concat([df_features, df_vecs], axis=1)
    final_df.to_parquet("mobilenet_features.parquet")

    print("Saved mobilenet_features.parquet")
    return final_df


# -------------------- SPLIT DATA ------------------------------------------#


def split_dataframe(df, train_ratio=0.8, val_ratio=0.1):
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    print(f"Train: {len(train_df)} rows")
    print(f"Val:   {len(val_df)} rows")
    print(f"Test:  {len(test_df)} rows")

    return train_df, val_df, test_df


def plot_training_curves(train_losses, val_losses):
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="Train MSE", linewidth=2)
    plt.plot(val_losses, label="Val MSE", linewidth=2)
    plt.title("Priebeh tréningu (MSE)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_curve.png")
    plt.show()


# ===========
# 2) Residual plots
# ===========
def plot_residuals(true_vals, pred_vals, title, filename):
    residuals = true_vals - pred_vals

    plt.figure(figsize=(8,5))
    plt.scatter(true_vals, residuals, s=8, alpha=0.6)
    plt.axhline(0, color="red", linestyle="--")
    plt.title(f"Reziduály – {title}")
    plt.xlabel("Skutočná hodnota")
    plt.ylabel("Reziduál (y_true - y_pred)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


# ===========
# 3) Pred vs True (scatter)
# ===========
def plot_pred_vs_true(true_vals, pred_vals, title, filename):
    plt.figure(figsize=(6,6))
    plt.scatter(true_vals, pred_vals, s=8, alpha=0.6)
    plt.plot([true_vals.min(), true_vals.max()],
             [true_vals.min(), true_vals.max()],
             color="red", linestyle="--")
    plt.title(f"Predikované vs Skutočné – {title}")
    plt.xlabel("Skutočná hodnota")
    plt.ylabel("Predikovaná hodnota")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


# ================================
#   VISUALIZATION OF CLUSTERS
# ================================

def show_cluster_examples(features_df, save_dir="cluster_examples", max_per_cluster=16):
    os.makedirs(save_dir, exist_ok=True)

    clusters = sorted(features_df["cluster"].unique())

    for cluster_id in clusters:
        subset = features_df[features_df["cluster"] == cluster_id].head(max_per_cluster)

        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        axes = axes.flatten()

        for ax, (_, row) in zip(axes, subset.iterrows()):
            img = Image.open(row["image_path"])
            ax.imshow(img)
            ax.set_title(f"Irr={row['Irradiance']:.1f}")
            ax.axis("off")

        plt.tight_layout()
        out_path = os.path.join(save_dir, f"cluster_{cluster_id}_examples.png")
        plt.savefig(out_path)
        plt.close()

        print(f"Saved image grid for cluster {cluster_id} → {out_path}")


def save_cluster_average_images(features_df, save_dir="cluster_averages"):
    os.makedirs(save_dir, exist_ok=True)

    clusters = sorted(features_df["cluster"].unique())

    for cluster_id in clusters:
        subset = features_df[features_df["cluster"] == cluster_id]

        imgs = []
        for img_path in subset["image_path"].values:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((224, 224))
            imgs.append(np.array(img, dtype=np.float32))

        if len(imgs) == 0:
            continue

        avg_img = np.mean(imgs, axis=0).astype(np.uint8)

        out_path = os.path.join(save_dir, f"cluster_{cluster_id}_average.png")
        Image.fromarray(avg_img).save(out_path)

        print(f"Saved average image for cluster {cluster_id} → {out_path}")



# -------------------- MAIN ------------------------------------------------#


def main():
    device = print_device_info()

    # 1) CSV
    df = load_all_csv()
    print("CSV rows:", len(df))
    #df = add_csv_match_keys(df)

    total_imgs = 0

    # 2) MATCH OBRÁZKOV
    matched_df = match_images_to_csv(df)

    print("\n===== EXTRACTING FEATURES WITH MOBILENET =====")
    features_df = extract_features_dataframe(matched_df, device)
    print(features_df.head())

    # ============================
    #  ZHLUKOVANIE FEATURES (K-MEANS)
    # ============================

    print("\n===== RUNNING K-MEANS CLUSTERING =====")

    # Vyberieme len feature stĺpce (f0...f1279)
    feature_cols = [c for c in features_df.columns if c.startswith("f")]
    X = features_df[feature_cols].values

    # Normalizácia (veľmi dôležité pre KMeans)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # KMeans
    kmeans = KMeans(n_clusters=6, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)


    features_df["cluster"] = clusters

    print("Clustering done! Cluster distribution:")
    print(features_df["cluster"].value_counts())

    # =====================================
    #   CLUSTER VISUALIZATION REQUIRED BY TASK
    # =====================================
    print("\n===== GENERATING CLUSTER IMAGE GRIDS =====")
    show_cluster_examples(features_df)

    print("\n===== GENERATING AVERAGE IMAGES PER CLUSTER =====")
    save_cluster_average_images(features_df)


    # 3) SPLIT
    train_df, val_df, test_df = split_dataframe(features_df)

    train_dataset = FeatureDataset(train_df)
    val_dataset = FeatureDataset(val_df)
    test_dataset = FeatureDataset(test_df)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 4) MODEL + OPTIMIZER
    model = LinearRegressor().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    epochs_no_improve = 0

    # 5) TRAINING LOOP
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device).unsqueeze(1)

            optimizer.zero_grad()
            preds = model(X)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * X.size(0)

        epoch_loss /= len(train_loader.dataset)

        # VALIDATION
        val_mse, _, _, _, _, _ = evaluate_loader(model, val_loader, device)

        train_losses.append(epoch_loss)
        val_losses.append(val_mse)

        print(f"Epoch {epoch}/{NUM_EPOCHS} | Train MSE: {epoch_loss:.4f} | Val MSE: {val_mse:.4f}")

        # EARLY STOPPING
        if val_mse < best_val_loss:
            best_val_loss = val_mse
            torch.save(model.state_dict(), "best_mlp.pt")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print("Early stopping activated!")
                break

    # Load best model
    model.load_state_dict(torch.load("best_mlp.pt"))


    # 6) EVALUATION
    print("\nEvaluating model on TRAINING SET...")
    train_mse, train_mae, train_rmse, train_r2, train_true, train_pred = \
        evaluate_loader(model, train_loader, device)

    print("\nEvaluating model on TEST SET...")
    test_mse, test_mae, test_rmse, test_r2, test_true, test_pred = \
        evaluate_loader(model, test_loader, device)

    # === TRAINING CURVES ===
    plot_training_curves(train_losses, val_losses)

    # === RESIDUALS ===
    plot_residuals(train_true, train_pred, "Training – MLP", "mlp_residuals_train.png")
    plot_residuals(test_true, test_pred, "Test – MLP", "mlp_residuals_test.png")

    # === PRED VS TRUE ===
    plot_pred_vs_true(train_true, train_pred, "Training – MLP", "mlp_pred_vs_true_train.png")
    plot_pred_vs_true(test_true, test_pred, "Test – MLP", "mlp_pred_vs_true_test.png")

    print("\n======= FINAL RESULTS (MLP + MobileNet features) =======")
    print("TRAIN:")
    print(f"MSE: {train_mse:.4f}")
    print(f"MAE: {train_mae:.4f}")
    print(f"RMSE: {train_rmse:.4f}")
    print(f"R²: {train_r2:.4f}")

    print("\nTEST:")
    print(f"MSE: {test_mse:.4f}")
    print(f"MAE: {test_mae:.4f}")
    print(f"RMSE: {test_rmse:.4f}")
    print(f"R²: {test_r2:.4f}")

    # ======== CURVES + PLOTS =========
    plot_training_curves(train_losses, val_losses)
    plot_residuals(train_true, train_pred, "Training", "residuals_train.png")
    plot_residuals(test_true, test_pred, "Test", "residuals_test.png")
    plot_pred_vs_true(train_true, train_pred, "Training", "pred_vs_true_train.png")
    plot_pred_vs_true(test_true, test_pred, "Test", "pred_vs_true_test.png")


    # # 4) TRANSFORMÁCIE + DATASETY + DATALOADER
    # train_dataset = IrradianceDataset(train_df, transform=train_transform)
    # val_dataset = IrradianceDataset(val_df, transform=val_test_transform)
    # test_dataset = IrradianceDataset(test_df, transform=val_test_transform)
    #
    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=BATCH_SIZE,
    #     shuffle=True,
    #     num_workers=4,
    #     pin_memory=True
    # )
    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=BATCH_SIZE,
    #     shuffle=False,
    #     num_workers=4,
    #     pin_memory=True
    # )
    # test_loader = DataLoader(
    #     test_dataset,
    #     batch_size=BATCH_SIZE,
    #     shuffle=False,
    #     num_workers=4,
    #     pin_memory=True
    # )

    # # 5) CNN MODEL
    # model = CNNRegressor(dropout_rate=DROPOUT_RATE).to(device)
    # criterion = nn.MSELoss()
    # optimizer = torch.optim.Adam(
    #     model.parameters(),
    #     lr=LR,
    #     weight_decay= 1e-4
    #
    # )
    #
    # # scheduler – znižuje LR, keď sa nezlepšuje validačný MSE
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     mode='min',
    #     factor=0.5,
    #     patience=5,
    #     min_lr=1e-6,
    #     verbose=True
    # )
    #
    # best_val_loss = float("inf")
    # best_state_dict = None
    # train_losses = []
    # val_losses = []
    # best_val_loss = float("inf")
    # epochs_no_improve = 0
    #
    # # 6) TRAIN LOOP + "checkpoint"
    # for epoch in range(1, NUM_EPOCHS + 1):
    #     model.train()
    #     epoch_loss = 0.0
    #
    #     for X, y in train_loader:
    #         X = X.to(device)
    #         y = y.to(device).unsqueeze(1)
    #
    #         optimizer.zero_grad()
    #         preds = model(X)
    #         loss = criterion(preds, y)
    #         loss.backward()
    #         optimizer.step()
    #
    #         epoch_loss += loss.item() * X.size(0)
    #
    #     epoch_loss /= len(train_loader.dataset)
    #
    #     # VALIDÁCIA
    #     val_mse, _, _, _, _, _ = evaluate_loader(model, val_loader, device)
    #     scheduler.step(val_mse)
    #
    #
    #     train_losses.append(epoch_loss)
    #     val_losses.append(val_mse)
    #
    #
    #     print(f"Epoch {epoch}/{NUM_EPOCHS} - "
    #           f"train MSE: {epoch_loss:.4f} | val MSE: {val_mse:.4f}")
    #
    #     # uložíme najlepší model
    #     if val_mse < best_val_loss:
    #
    #         best_val_loss = val_mse
    #         best_state_dict = model.state_dict()
    #         torch.save(best_state_dict, "best_model.pt")
    #         epochs_no_improve = 0
    #     else:
    #         epochs_no_improve += 1
    #         if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
    #             print(f"\nEarly stopping triggered (no improvement for {EARLY_STOPPING_PATIENCE} epochs).")
    #             break
    #
    # print("\nLoading BEST model from checkpoint...")
    # if best_state_dict is None:
    #     best_state_dict = torch.load("best_model.pt", map_location=device)
    # model.load_state_dict(best_state_dict)
    #
    # # 7) EVALUÁCIA NA TRAIN + TEST
    # print("\nEvaluating model on TRAINING SET...")
    # train_mse, train_mae, train_rmse, train_r2, train_y_true, train_y_pred = \
    #     evaluate_loader(model, train_loader, device)
    #
    # print("\nEvaluating model on TEST SET...")
    # test_mse, test_mae, test_rmse, test_r2, test_y_true, test_y_pred = \
    #     evaluate_loader(model, test_loader, device)
    #
    # print("\n================ MODEL METRICS ================")
    # print("TRAINING:")
    # print(f"MSE:  {train_mse:.4f}")
    # print(f"MAE:  {train_mae:.4f}")
    # print(f"RMSE: {train_rmse:.4f}")
    # print(f"R2:   {train_r2:.4f}")
    #
    # print("\nTESTING:")
    # print(f"MSE:  {test_mse:.4f}")
    # print(f"MAE:  {test_mae:.4f}")
    # print(f"RMSE: {test_rmse:.4f}")
    # print(f"R2:   {test_r2:.4f}")
    #
    #
    # # np.save("train_true.npy", train_y_true)
    # # np.save("train_pred.npy", train_y_pred)
    # # np.save("test_true.npy", test_y_true)
    # # np.save("test_pred.npy", test_y_pred)
    #
    # print("\nReziduály uložené do .npy súborov.")
    # print("================================================\n")
    #
    # # ========  TRAINING CURVES  =========
    # plot_training_curves(train_losses, val_losses)
    #
    # # ======== RESIDUALS – TRAINING ======
    # plot_residuals(train_y_true, train_y_pred,
    #                title="Training set",
    #                filename="residuals_train.png")
    #
    # # ======== RESIDUALS – TESTING =======
    # plot_residuals(test_y_true, test_y_pred,
    #                title="Test set",
    #                filename="residuals_test.png")
    #
    # # ======== PRED VS TRUE ===============
    # plot_pred_vs_true(train_y_true, train_y_pred,
    #                   title="Training",
    #                   filename="pred_vs_true_train.png")
    #
    # plot_pred_vs_true(test_y_true, test_y_pred,
    #                   title="Testing",
    #                   filename="pred_vs_true_test.png")

    print("End timestamp:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

if __name__ == "__main__":
    main()
