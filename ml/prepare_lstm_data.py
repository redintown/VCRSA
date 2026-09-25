import csv
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader


BASE_DIR = Path("sumo/datasets/highway/processed")
ML_DIR = Path("ml")

TRAIN_FILE = BASE_DIR / "train.csv"
VAL_FILE = BASE_DIR / "validation.csv"
TEST_FILE = BASE_DIR / "test.csv"

SCALER_FILE = ML_DIR / "highway_lstm_scaler.pkl"

SEQUENCE_LENGTH = 10

FEATURES = [
    "x",
    "y",
    "speed",
    "acceleration",
    "heading",
    "lane_id",
    "direction",
    "zone_id",
]

BATCH_SIZE = 64


def load_csv(path):

    with path.open(newline="") as f:
        reader = csv.DictReader(f)

        rows = list(reader)

    return rows


def extract_features(rows):

    X = []

    y = []

    for row in rows:

        sequence = []

        for t in range(SEQUENCE_LENGTH):

            values = [
                float(row[f"t{t}_x"]),
                float(row[f"t{t}_y"]),
                float(row[f"t{t}_speed"]),
                float(row[f"t{t}_acceleration"]),
                float(row[f"t{t}_heading"]),
                float(row[f"t{t}_lane_id"]),
                float(row[f"t{t}_direction"]),
                float(row[f"t{t}_zone_id"]),
            ]

            sequence.append(values)

        X.append(sequence)
        y.append(float(row["target"]))

    return np.array(X, dtype=np.float32), np.array(
        y, dtype=np.float32
    )


def main():

    print("Loading datasets...")

    train_rows = load_csv(TRAIN_FILE)
    val_rows = load_csv(VAL_FILE)
    test_rows = load_csv(TEST_FILE)

    print(f"Train rows:      {len(train_rows)}")
    print(f"Validation rows: {len(val_rows)}")
    print(f"Test rows:       {len(test_rows)}")

    X_train, y_train = extract_features(train_rows)
    X_val, y_val = extract_features(val_rows)
    X_test, y_test = extract_features(test_rows)

    print()
    print("Original shapes:")
    print("  X_train:", X_train.shape)
    print("  X_val:  ", X_val.shape)
    print("  X_test: ", X_test.shape)

    # ---------------------------------------------------------
    # Normalize features using TRAINING DATA ONLY
    # ---------------------------------------------------------

    scaler = StandardScaler()

    # Flatten:
    # (samples, timesteps, features)
    # ->
    # (samples * timesteps, features)
    train_2d = X_train.reshape(-1, len(FEATURES))

    scaler.fit(train_2d)

    def transform(X):

        original_shape = X.shape

        X_2d = X.reshape(
            -1,
            len(FEATURES)
        )

        X_scaled = scaler.transform(X_2d)

        return X_scaled.reshape(original_shape).astype(
            np.float32
        )

    X_train = transform(X_train)
    X_val = transform(X_val)
    X_test = transform(X_test)

    # Save scaler for later inference
    ML_DIR.mkdir(parents=True, exist_ok=True)

    with SCALER_FILE.open("wb") as f:
        pickle.dump(scaler, f)

    # ---------------------------------------------------------
    # Convert to PyTorch tensors
    # ---------------------------------------------------------

    X_train_tensor = torch.tensor(X_train)
    y_train_tensor = torch.tensor(y_train)

    X_val_tensor = torch.tensor(X_val)
    y_val_tensor = torch.tensor(y_val)

    X_test_tensor = torch.tensor(X_test)
    y_test_tensor = torch.tensor(y_test)

    # ---------------------------------------------------------
    # PyTorch datasets
    # ---------------------------------------------------------

    train_dataset = TensorDataset(
        X_train_tensor,
        y_train_tensor
    )

    val_dataset = TensorDataset(
        X_val_tensor,
        y_val_tensor
    )

    test_dataset = TensorDataset(
        X_test_tensor,
        y_test_tensor
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print()
    print("Normalized shapes:")
    print("  X_train:", X_train.shape)
    print("  X_val:  ", X_val.shape)
    print("  X_test: ", X_test.shape)

    print()
    print("Target shapes:")
    print("  y_train:", y_train.shape)
    print("  y_val:  ", y_val.shape)
    print("  y_test: ", y_test.shape)

    print()
    print("DataLoader:")
    print("  Train batches:", len(train_loader))
    print("  Val batches:  ", len(val_loader))
    print("  Test batches: ", len(test_loader))

    print()
    print("Scaler saved:")
    print(f"  {SCALER_FILE}")

    # ---------------------------------------------------------
    # Verify one batch
    # ---------------------------------------------------------

    batch_X, batch_y = next(iter(train_loader))

    print()
    print("Sample training batch:")
    print("  X:", batch_X.shape)
    print("  y:", batch_y.shape)

    print()
    print("Preparation complete.")


if __name__ == "__main__":
    main()
