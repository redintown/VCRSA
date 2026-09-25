import csv
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from lstm_model import MobilityLSTM


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = Path("sumo/datasets/highway/processed")
CHECKPOINT_DIR = Path("ml/checkpoints")

TRAIN_FILE = BASE_DIR / "train.csv"
VAL_FILE = BASE_DIR / "validation.csv"
TEST_FILE = BASE_DIR / "test.csv"

SCALER_FILE = Path("ml/highway_lstm_scaler.pkl")

BEST_MODEL_FILE = CHECKPOINT_DIR / "best_highway_lstm.pt"


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

SEQUENCE_LENGTH = 10
NUM_FEATURES = 8

HIDDEN_SIZE = 64
NUM_LAYERS = 2

BATCH_SIZE = 64
EPOCHS = 50

LEARNING_RATE = 0.001

PATIENCE = 7

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ---------------------------------------------------------
# Data loading
# ---------------------------------------------------------

def load_csv(path):

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


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

    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.float32),
    )


# ---------------------------------------------------------
# Normalization
# ---------------------------------------------------------

def normalize(X, scaler):

    original_shape = X.shape

    X_2d = X.reshape(
        -1,
        NUM_FEATURES
    )

    X_scaled = scaler.transform(X_2d)

    return X_scaled.reshape(
        original_shape
    ).astype(np.float32)


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

def calculate_metrics(predictions, targets):

    predictions = np.array(predictions)
    targets = np.array(targets)

    errors = predictions - targets

    mae = np.mean(
        np.abs(errors)
    )

    rmse = np.sqrt(
        np.mean(errors ** 2)
    )

    return mae, rmse


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

def evaluate(model, loader, criterion):

    model.eval()

    total_loss = 0.0
    all_predictions = []
    all_targets = []

    with torch.no_grad():

        for X, y in loader:

            X = X.to(DEVICE)
            y = y.to(DEVICE)

            predictions = model(X)

            loss = criterion(
                predictions,
                y
            )

            total_loss += (
                loss.item() * X.size(0)
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_targets.extend(
                y.cpu().numpy()
            )

    average_loss = (
        total_loss / len(loader.dataset)
    )

    mae, rmse = calculate_metrics(
        all_predictions,
        all_targets
    )

    return average_loss, mae, rmse


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("=" * 60)
    print("Highway Mobility LSTM Training")
    print("=" * 60)

    print()
    print("Device:", DEVICE)

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # -----------------------------------------------------
    # Load data
    # -----------------------------------------------------

    print()
    print("Loading datasets...")

    train_rows = load_csv(TRAIN_FILE)
    val_rows = load_csv(VAL_FILE)
    test_rows = load_csv(TEST_FILE)

    X_train, y_train = extract_features(
        train_rows
    )

    X_val, y_val = extract_features(
        val_rows
    )

    X_test, y_test = extract_features(
        test_rows
    )

    # -----------------------------------------------------
    # Load training-fitted scaler
    # -----------------------------------------------------

    print("Loading scaler...")

    with SCALER_FILE.open("rb") as f:
        scaler = pickle.load(f)

    X_train = normalize(
        X_train,
        scaler
    )

    X_val = normalize(
        X_val,
        scaler
    )

    X_test = normalize(
        X_test,
        scaler
    )

    # -----------------------------------------------------
    # PyTorch datasets
    # -----------------------------------------------------

    train_dataset = TensorDataset(
        torch.tensor(X_train),
        torch.tensor(y_train)
    )

    val_dataset = TensorDataset(
        torch.tensor(X_val),
        torch.tensor(y_val)
    )

    test_dataset = TensorDataset(
        torch.tensor(X_test),
        torch.tensor(y_test)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=torch.cuda.is_available()
    )

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    model = MobilityLSTM(
        input_size=NUM_FEATURES,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=1,
    ).to(DEVICE)

    print()
    print("Model:")
    print(model)

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print()
    print(
        "Trainable parameters:",
        total_parameters
    )

    # -----------------------------------------------------
    # Loss and optimizer
    # -----------------------------------------------------

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    print()
    print("=" * 60)
    print("Training")
    print("=" * 60)

    for epoch in range(1, EPOCHS + 1):

        model.train()

        running_loss = 0.0

        for X, y in train_loader:

            X = X.to(
                DEVICE,
                non_blocking=True
            )

            y = y.to(
                DEVICE,
                non_blocking=True
            )

            optimizer.zero_grad()

            predictions = model(X)

            loss = criterion(
                predictions,
                y
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item() * X.size(0)
            )

        train_loss = (
            running_loss
            / len(train_loader.dataset)
        )

        val_loss, val_mae, val_rmse = evaluate(
            model,
            val_loader,
            criterion
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val MAE: {val_mae:.4f}s | "
            f"Val RMSE: {val_rmse:.4f}s"
        )

        # -------------------------------------------------
        # Save best model
        # -------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "input_size":
                        NUM_FEATURES,

                    "hidden_size":
                        HIDDEN_SIZE,

                    "num_layers":
                        NUM_LAYERS,

                    "sequence_length":
                        SEQUENCE_LENGTH,

                    "best_val_loss":
                        best_val_loss,
                },
                BEST_MODEL_FILE
            )

            print(
                "  -> Best model saved."
            )

        else:

            epochs_without_improvement += 1

        # -------------------------------------------------
        # Early stopping
        # -------------------------------------------------

        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print()
            print(
                "Early stopping triggered."
            )

            break

    # -----------------------------------------------------
    # Load best model
    # -----------------------------------------------------

    print()
    print("=" * 60)
    print("Loading best model")
    print("=" * 60)

    checkpoint = torch.load(
        BEST_MODEL_FILE,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # -----------------------------------------------------
    # Final test evaluation
    # -----------------------------------------------------

    test_loss, test_mae, test_rmse = evaluate(
        model,
        test_loader,
        criterion
    )

    print()
    print("=" * 60)
    print("Final Test Results")
    print("=" * 60)

    print(
        f"Test MSE:  {test_loss:.4f}"
    )

    print(
        f"Test MAE:  {test_mae:.4f} seconds"
    )

    print(
        f"Test RMSE: {test_rmse:.4f} seconds"
    )

    print()
    print(
        "Best model:",
        BEST_MODEL_FILE
    )


if __name__ == "__main__":
    main()
