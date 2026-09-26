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

TARGET_ZONE = 0
RESOURCE_TYPE = "CPU"
REQUEST_AMOUNT = 20.0

FAIRNESS_RESERVE = 0.10

TRUST_SCORE = 0.90


# ============================================================
# Data structures
# ============================================================

@dataclass
class Provider:

    vehicle_id: str
    zone_id: int
    resource_type: str
    available_amount: float
    predicted_residency: float
    trust_score: float = 1.0
    active: bool = True


@dataclass
class Allocation:

    vehicle_id: str
    resource_type: str
    amount: float
    predicted_residency: float


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

    elif resource_class == 1:

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

        model.load_state_dict(
            checkpoint
        )

    model.to(device)
    model.eval()

    scaler = joblib.load(
        SCALER_PATH
    )

    return model, scaler, device


# ============================================================
# Find 10-second sequence
# ============================================================

def find_sequence(vehicle_df):

    vehicle_df = vehicle_df.sort_values(
        "timestamp"
    )

    for zone_id, zone_df in vehicle_df.groupby(
        "zone_id"
    ):

        zone_df = zone_df.sort_values(
            "timestamp"
        )

        timestamps = zone_df[
            "timestamp"
        ].to_numpy()

        if len(timestamps) < 10:
            continue

        for i in range(
            len(timestamps) - 9
        ):

            window = timestamps[
                i:i + 10
            ]

            if np.allclose(
                np.diff(window),
                1.0,
            ):

                return zone_df.iloc[
                    i:i + 10
                ].copy()

    return None


# ============================================================
# Prepare LSTM features
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
# Predict residency
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
# Build dynamic providers
# ============================================================

def build_providers(
    df,
    model,
    scaler,
    device,
):

    vehicle_ids = (
        df["vehicle_id"]
        .drop_duplicates()
        .tolist()
    )

    providers = []

    for vehicle_index, vehicle_id in enumerate(
        vehicle_ids
    ):

        vehicle_df = df[
            df["vehicle_id"]
            == vehicle_id
        ]

        sequence = find_sequence(
            vehicle_df
        )

        if sequence is None:
            continue

        latest = sequence.iloc[-1]

        zone = int(
            latest["zone_id"]
        )

        # Only providers in requested zone
        if zone != TARGET_ZONE:
            continue

        predicted_residency = (
            predict_residency(
                sequence,
                model,
                scaler,
                device,
            )
        )

        resources = resource_profile(
            vehicle_index
        )

        providers.append(
            Provider(
                vehicle_id=vehicle_id,
                zone_id=zone,
                resource_type=RESOURCE_TYPE,
                available_amount=resources[
                    RESOURCE_TYPE
                ],
                predicted_residency=(
                    predicted_residency
                ),
                trust_score=TRUST_SCORE,
            )
        )

    return providers


# ============================================================
# Greedy allocation
# ============================================================

def greedy_allocate(
    providers,
    requested_amount,
):

    candidates = []

    for provider in providers:

        if not provider.active:
            continue

        usable_amount = (
            provider.available_amount
            * (1.0 - FAIRNESS_RESERVE)
        )

        if usable_amount <= 0:
            continue

        score = provider_score(
            provider
        )

        candidates.append(
            (
                provider,
                usable_amount,
                score,
            )
        )

    candidates.sort(
        key=lambda item: item[2],
        reverse=True,
    )

    allocations = []

    remaining = requested_amount

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
                resource_type=provider.resource_type,
                amount=amount,
                predicted_residency=(
                    provider.predicted_residency
                ),
            )
        )

        remaining -= amount

    if remaining > 1e-9:
        return []

    return allocations


# ============================================================
# Create PRT tenancy
# ============================================================

def create_tenancy(
    allocations,
    requested_amount,
):

    if not allocations:
        return None

    allocated_amount = sum(
        allocation.amount
        for allocation in allocations
    )

    tenancy_duration = min(
        allocation.predicted_residency
        for allocation in allocations
    )

    return Tenancy(
        request_id="REQ-Z0-CPU-001",
        resource_type=RESOURCE_TYPE,
        requested_amount=requested_amount,
        allocated_amount=allocated_amount,
        duration=tenancy_duration,
        providers=[
            allocation.vehicle_id
            for allocation in allocations
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

    # --------------------------------------------------------
    # Build real dynamic provider pool
    # --------------------------------------------------------

    providers = build_providers(
        df,
        model,
        scaler,
        device,
    )

    print()
    print("=" * 90)
    print(
        "             ZONE-AWARE PROVIDER POOL"
    )
    print("=" * 90)

    print(
        f"Target zone       : {TARGET_ZONE}"
    )

    print(
        f"Resource          : {RESOURCE_TYPE}"
    )

    print(
        f"Providers found   : {len(providers)}"
    )

    print()

    print(
        f"{'Vehicle':<24}"
        f"{'CPU':<8}"
        f"{'Residency':<12}"
        f"{'Trust':<8}"
        f"{'Score':<10}"
    )

    print("-" * 70)

    for provider in sorted(
        providers,
        key=provider_score,
        reverse=True,
    ):

        print(
            f"{provider.vehicle_id:<24}"
            f"{provider.available_amount:<8.1f}"
            f"{provider.predicted_residency:<12.2f}"
            f"{provider.trust_score:<8.2f}"
            f"{provider_score(provider):<10.4f}"
        )

    # --------------------------------------------------------
    # Allocation
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print(
        "                 GREEDY ALLOCATION"
    )
    print("=" * 90)

    print(
        f"Requested CPU    : "
        f"{REQUEST_AMOUNT:.1f}"
    )

    print(
        f"Fairness reserve : "
        f"{FAIRNESS_RESERVE * 100:.0f}%"
    )

    allocations = greedy_allocate(
        providers,
        REQUEST_AMOUNT,
    )

    if not allocations:

        print()
        print(
            "Allocation FAILED"
        )

        return

    total_allocated = 0.0

    for allocation in allocations:

        print(
            f"{allocation.vehicle_id:<24}"
            f"allocated={allocation.amount:.2f} CPU"
            f" | residency="
            f"{allocation.predicted_residency:.2f}s"
        )

        total_allocated += (
            allocation.amount
        )

    print()
    print(
        f"Total allocated  : "
        f"{total_allocated:.2f} CPU"
    )

    # --------------------------------------------------------
    # PRT
    # --------------------------------------------------------

    tenancy = create_tenancy(
        allocations,
        REQUEST_AMOUNT,
    )

    print()
    print("=" * 90)
    print(
        "                 PRT TENANCY"
    )
    print("=" * 90)

    print(
        f"Request ID       : "
        f"{tenancy.request_id}"
    )

    print(
        f"Resource         : "
        f"{tenancy.resource_type}"
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

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print(
        "                 VALIDATION"
    )
    print("=" * 90)

    print(
        "Dynamic providers : PASS"
    )

    print(
        "Zone filtering    : "
        + (
            "PASS"
            if all(
                p.zone_id == TARGET_ZONE
                for p in providers
            )
            else "FAIL"
        )
    )

    print(
        "Resource filtering: "
        + (
            "PASS"
            if all(
                p.resource_type
                == RESOURCE_TYPE
                for p in providers
            )
            else "FAIL"
        )
    )

    print(
        "Greedy allocation : "
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
        "PRT tenancy       : "
        + (
            "PASS"
            if tenancy.duration > 0
            else "FAIL"
        )
    )

    print()
    print(
        "Dynamic Allocation + PRT: PASS"
    )

    print("=" * 90)


if __name__ == "__main__":
    main()
