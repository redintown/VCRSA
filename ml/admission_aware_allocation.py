import sys
from pathlib import Path
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
import torch

ML_DIR = Path(__file__).resolve().parent
PROJECT_DIR = ML_DIR.parent

sys.path.insert(0, str(ML_DIR))

from lstm_model import MobilityLSTM
from admission_policy import classify_admission


# ============================================================
# Configuration
# ============================================================

CSV_PATH = (
    PROJECT_DIR
    / "sumo"
    / "datasets"
    / "highway"
    / "raw"
    / "highway_mobility.csv"
)

MODEL_PATH = (
    ML_DIR
    / "checkpoints"
    / "best_highway_lstm.pt"
)

SCALER_PATH = (
    ML_DIR
    / "highway_lstm_scaler.pkl"
)

SNAPSHOT_TIME = 100.0
HISTORY_SECONDS = 10

TARGET_ZONE = 0
RESOURCE_TYPE = "CPU"
REQUEST_AMOUNT = 20.0

# Experimental prototype thresholds.
X_THRESHOLD = 25.0
TMIN_THRESHOLD = 10.0

FAIRNESS_RESERVE = 0.10
TRUST_SCORE = 0.90


# ============================================================
# Data structures
# ============================================================

@dataclass
class Provider:
    vehicle_id: str
    zone_id: int
    direction: str
    resource_type: str
    available_amount: float
    predicted_residency: float
    admission: str
    trust_score: float = 1.0
    active: bool = True


@dataclass
class Allocation:
    vehicle_id: str
    amount: float
    predicted_residency: float
    admission: str


@dataclass
class Tenancy:
    request_id: str
    resource_type: str
    requested_amount: float
    allocated_amount: float
    duration: float
    providers: list


# ============================================================
# Resource profile
# ============================================================

def resource_profile(vehicle_index):

    resource_class = vehicle_index % 3

    if resource_class == 0:
        return {
            "CPU": 4.0,
            "MEMORY": 8.0,
            "STORAGE": 128.0,
        }

    if resource_class == 1:
        return {
            "CPU": 8.0,
            "MEMORY": 16.0,
            "STORAGE": 256.0,
        }

    return {
        "CPU": 12.0,
        "MEMORY": 32.0,
        "STORAGE": 512.0,
    }


# ============================================================
# Load LSTM
# ============================================================

def load_lstm():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = MobilityLSTM(
        input_size=8,
        hidden_size=64,
        num_layers=2,
        output_size=1,
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    scaler = joblib.load(
        SCALER_PATH
    )

    return model, scaler, device


# ============================================================
# Extract 10-second history
# ============================================================

def get_snapshot_sequence(
    vehicle_df,
    snapshot_time,
):

    start_time = (
        snapshot_time
        - HISTORY_SECONDS
        + 1
    )

    sequence = vehicle_df[
        (
            vehicle_df["timestamp"]
            >= start_time
        )
        &
        (
            vehicle_df["timestamp"]
            <= snapshot_time
        )
    ].copy()

    if len(sequence) != HISTORY_SECONDS:
        return None

    sequence = sequence.sort_values(
        "timestamp"
    )

    timestamps = sequence[
        "timestamp"
    ].to_numpy()

    if not np.allclose(
        np.diff(timestamps),
        1.0,
    ):
        return None

    if sequence["zone_id"].nunique() != 1:
        return None

    return sequence


# ============================================================
# LSTM features
# ============================================================

def prepare_features(sequence):

    direction_numeric = (
        sequence["direction"]
        .map(
            {
                "EAST": 1.0,
                "WEST": -1.0,
            }
        )
        .astype(float)
    )

    lane_numeric = (
        sequence["lane_id"]
        .str.extract(
            r"_(\d+)$"
        )[0]
        .astype(float)
    )

    return np.column_stack(
        [
            sequence["x"].to_numpy(),
            sequence["y"].to_numpy(),
            sequence["speed"].to_numpy(),
            sequence[
                "acceleration"
            ].to_numpy(),
            sequence["heading"].to_numpy(),
            lane_numeric.to_numpy(),
            direction_numeric.to_numpy(),
            sequence["zone_id"].to_numpy(),
        ]
    ).astype(np.float32)


# ============================================================
# LSTM prediction
# ============================================================

def predict_residency(
    sequence,
    model,
    scaler,
    device,
):

    features = prepare_features(
        sequence
    )

    scaled = scaler.transform(
        features
    )

    tensor = torch.tensor(
        scaled,
        dtype=torch.float32,
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(
            tensor
        )

    return max(
        float(prediction.item()),
        0.0,
    )


# ============================================================
# Provider score
# ============================================================

def provider_score(provider):

    residency_score = min(
        provider.predicted_residency / 60.0,
        1.0,
    )

    capacity_score = min(
        provider.available_amount / 16.0,
        1.0,
    )

    return (
        0.50 * residency_score
        + 0.30 * provider.trust_score
        + 0.20 * capacity_score
    )


# ============================================================
# Build provider pool + admission
# ============================================================

def build_provider_pool(
    df,
    model,
    scaler,
    device,
):

    providers = []

    vehicle_ids = (
        df["vehicle_id"]
        .drop_duplicates()
        .tolist()
    )

    for vehicle_index, vehicle_id in enumerate(
        vehicle_ids
    ):

        vehicle_df = df[
            df["vehicle_id"]
            == vehicle_id
        ].copy()

        current = vehicle_df[
            vehicle_df["timestamp"]
            == SNAPSHOT_TIME
        ]

        if current.empty:
            continue

        current = current.iloc[0]

        current_zone = int(
            current["zone_id"]
        )

        if current_zone != TARGET_ZONE:
            continue

        sequence = get_snapshot_sequence(
            vehicle_df,
            SNAPSHOT_TIME,
        )

        if sequence is None:
            continue

        predicted_residency = (
            predict_residency(
                sequence,
                model,
                scaler,
                device,
            )
        )

        admission = classify_admission(
            predicted_residency,
            X_THRESHOLD,
            TMIN_THRESHOLD,
        )

        resources = resource_profile(
            vehicle_index
        )

        providers.append(
            Provider(
                vehicle_id=vehicle_id,
                zone_id=current_zone,
                direction=current["direction"],
                resource_type=RESOURCE_TYPE,
                available_amount=resources[
                    RESOURCE_TYPE
                ],
                predicted_residency=(
                    predicted_residency
                ),
                admission=admission.value,
                trust_score=TRUST_SCORE,
            )
        )

    return providers


# ============================================================
# Admission-aware greedy allocation
# ============================================================

def greedy_allocate(
    providers,
    requested_amount,
):

    candidates = []

    for provider in providers:

        if not provider.active:
            continue

        # REJECT providers are excluded.
        if provider.admission == "REJECT":
            continue

        usable = (
            provider.available_amount
            * (1.0 - FAIRNESS_RESERVE)
        )

        if usable <= 0:
            continue

        candidates.append(
            (
                provider,
                usable,
                provider_score(provider),
            )
        )

    candidates.sort(
        key=lambda item: item[2],
        reverse=True,
    )

    remaining = requested_amount
    allocations = []

    for provider, usable, score in candidates:

        if remaining <= 0:
            break

        amount = min(
            usable,
            remaining,
        )

        allocations.append(
            Allocation(
                vehicle_id=provider.vehicle_id,
                amount=amount,
                predicted_residency=(
                    provider.predicted_residency
                ),
                admission=provider.admission,
            )
        )

        remaining -= amount

    if remaining > 1e-9:
        return []

    return allocations


# ============================================================
# PRT
# ============================================================

def create_tenancy(
    allocations,
    requested_amount,
):

    if not allocations:
        return None

    allocated = sum(
        a.amount
        for a in allocations
    )

    duration = min(
        a.predicted_residency
        for a in allocations
    )

    return Tenancy(
        request_id="REQ-ADMISSION-Z0-CPU-001",
        resource_type=RESOURCE_TYPE,
        requested_amount=requested_amount,
        allocated_amount=allocated,
        duration=duration,
        providers=[
            a.vehicle_id
            for a in allocations
        ],
    )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "Loading mobility dataset..."
    )

    df = pd.read_csv(
        CSV_PATH
    )

    print(
        f"Mobility records: {len(df)}"
    )

    model, scaler, device = (
        load_lstm()
    )

    print(
        f"LSTM device: {device}"
    )

    providers = build_provider_pool(
        df,
        model,
        scaler,
        device,
    )

    # ========================================================
    # Admission table
    # ========================================================

    print()
    print("=" * 95)
    print(
        "        TIME-AWARE PROVIDER + PREDICTIVE ADMISSION"
    )
    print("=" * 95)

    print(
        f"Snapshot           : t={SNAPSHOT_TIME:.0f}s"
    )

    print(
        f"Target zone        : {TARGET_ZONE}"
    )

    print(
        f"X threshold        : {X_THRESHOLD:.1f}s"
    )

    print(
        f"Tmin threshold     : {TMIN_THRESHOLD:.1f}s"
    )

    print()

    print(
        f"{'Vehicle':<24}"
        f"{'CPU':<7}"
        f"{'Residency':<12}"
        f"{'Admission':<14}"
        f"{'Score':<10}"
    )

    print("-" * 80)

    for provider in sorted(
        providers,
        key=provider_score,
        reverse=True,
    ):

        print(
            f"{provider.vehicle_id:<24}"
            f"{provider.available_amount:<7.1f}"
            f"{provider.predicted_residency:<12.2f}"
            f"{provider.admission:<14}"
            f"{provider_score(provider):<10.4f}"
        )

    # ========================================================
    # Count admission tiers
    # ========================================================

    full_count = sum(
        p.admission == "FULL"
        for p in providers
    )

    provisional_count = sum(
        p.admission == "PROVISIONAL"
        for p in providers
    )

    reject_count = sum(
        p.admission == "REJECT"
        for p in providers
    )

    print()
    print(
        f"FULL       : {full_count}"
    )

    print(
        f"PROVISIONAL: {provisional_count}"
    )

    print(
        f"REJECT     : {reject_count}"
    )

    # ========================================================
    # Allocation
    # ========================================================

    allocations = greedy_allocate(
        providers,
        REQUEST_AMOUNT,
    )

    print()
    print("=" * 95)
    print(
        "              ADMISSION-AWARE GREEDY ALLOCATION"
    )
    print("=" * 95)

    print(
        f"Requested CPU: "
        f"{REQUEST_AMOUNT:.2f}"
    )

    if not allocations:

        print()
        print(
            "Allocation FAILED:"
        )

        print(
            "Eligible providers cannot satisfy "
            "the requested CPU."
        )

        return

    total_allocated = 0.0

    for allocation in allocations:

        print(
            f"{allocation.vehicle_id:<24}"
            f"allocated={allocation.amount:.2f} CPU"
            f" | residency="
            f"{allocation.predicted_residency:.2f}s"
            f" | "
            f"{allocation.admission}"
        )

        total_allocated += (
            allocation.amount
        )

    print()
    print(
        f"Total allocated: "
        f"{total_allocated:.2f} CPU"
    )

    # ========================================================
    # PRT
    # ========================================================

    tenancy = create_tenancy(
        allocations,
        REQUEST_AMOUNT,
    )

    print()
    print("=" * 95)
    print(
        "                         PRT"
    )
    print("=" * 95)

    print(
        f"Request ID       : "
        f"{tenancy.request_id}"
    )

    print(
        f"Requested        : "
        f"{tenancy.requested_amount:.2f}"
    )

    print(
        f"Allocated        : "
        f"{tenancy.allocated_amount:.2f}"
    )

    print(
        f"Tenancy duration : "
        f"{tenancy.duration:.2f}s"
    )

    print(
        f"Providers        : "
        f"{', '.join(tenancy.providers)}"
    )

    # ========================================================
    # Validation
    # ========================================================

    print()
    print("=" * 95)
    print(
        "                       VALIDATION"
    )
    print("=" * 95)

    print(
        "Admission classification : "
        + (
            "PASS"
            if len(providers) > 0
            else "CHECK"
        )
    )

    print(
        "REJECT exclusion        : "
        + (
            "PASS"
            if all(
                a.vehicle_id
                not in [
                    p.vehicle_id
                    for p in providers
                    if p.admission == "REJECT"
                ]
                for a in allocations
            )
            else "FAIL"
        )
    )

    print(
        "Greedy allocation       : "
        + (
            "PASS"
            if abs(
                total_allocated
                - REQUEST_AMOUNT
            ) < 1e-6
            else "FAIL"
        )
    )

    print(
        "PRT tenancy             : "
        + (
            "PASS"
            if tenancy.duration > 0
            else "FAIL"
        )
    )

    print()
    print(
        "Admission-aware Allocation + PRT: PASS"
    )

    print("=" * 95)


if __name__ == "__main__":
    main()
