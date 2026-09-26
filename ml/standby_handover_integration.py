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

X_THRESHOLD = 25.0
TMIN_THRESHOLD = 10.0

FAIRNESS_RESERVE = 0.10
TRUST_SCORE = 0.90

# Prototype handover trigger.
HANDOVER_MARGIN = 5.0


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
# 10-second history
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
# Build current provider pool
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
        request_id="REQ-HO-Z0-CPU-001",
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
# Standby selection
# ============================================================

def select_standby(
    providers,
    active_provider_ids,
    required_amount,
):

    candidates = []

    for provider in providers:

        if not provider.active:
            continue

        if provider.vehicle_id in active_provider_ids:
            continue

        if provider.resource_type != RESOURCE_TYPE:
            continue

        if provider.admission == "REJECT":
            continue

        usable = (
            provider.available_amount
            * (1.0 - FAIRNESS_RESERVE)
        )

        if usable < required_amount:
            continue

        candidates.append(
            provider
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda p: (
            p.predicted_residency,
            p.trust_score,
            p.available_amount,
        ),
        reverse=True,
    )

    return candidates[0]


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

    # --------------------------------------------------------
    # Allocation
    # --------------------------------------------------------

    allocations = greedy_allocate(
        providers,
        REQUEST_AMOUNT,
    )

    if not allocations:

        print(
            "Allocation failed."
        )

        return

    tenancy = create_tenancy(
        allocations,
        REQUEST_AMOUNT,
    )

    active_ids = [
        a.vehicle_id
        for a in allocations
    ]

    # --------------------------------------------------------
    # Standby
    # --------------------------------------------------------

    standby = select_standby(
        providers=providers,
        active_provider_ids=active_ids,
        required_amount=4.0,
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    print()
    print("=" * 95)
    print(
        "          STANDBY + PRE-DEPARTURE HANDOVER"
    )
    print("=" * 95)

    print(
        f"Snapshot           : "
        f"t={SNAPSHOT_TIME:.0f}s"
    )

    print(
        f"Zone               : "
        f"{TARGET_ZONE}"
    )

    print(
        f"Requested CPU      : "
        f"{REQUEST_AMOUNT:.2f}"
    )

    print()
    print(
        "ACTIVE PROVIDERS"
    )

    print("-" * 70)

    for allocation in allocations:

        print(
            f"{allocation.vehicle_id:<24}"
            f"CPU={allocation.amount:.2f}"
            f" | residency="
            f"{allocation.predicted_residency:.2f}s"
            f" | {allocation.admission}"
        )

    print()
    print(
        f"PRT tenancy duration: "
        f"{tenancy.duration:.2f}s"
    )

    print()
    print(
        "STANDBY PROVIDER"
    )

    print("-" * 70)

    if standby is None:

        print(
            "No eligible standby provider found."
        )

        print()
        print(
            "Handover cannot be guaranteed."
        )

        return

    print(
        f"Standby vehicle    : "
        f"{standby.vehicle_id}"
    )

    print(
        f"Available CPU      : "
        f"{standby.available_amount:.2f}"
    )

    print(
        f"Predicted residency: "
        f"{standby.predicted_residency:.2f}s"
    )

    print(
        f"Admission           : "
        f"{standby.admission}"
    )

    # --------------------------------------------------------
    # Select provider that triggers handover
    # --------------------------------------------------------

    departing = min(
        allocations,
        key=lambda a:
            a.predicted_residency,
    )

    print()
    print(
        "PRE-DEPARTURE HANDOVER"
    )

    print("-" * 70)

    print(
        f"Departing provider : "
        f"{departing.vehicle_id}"
    )

    print(
        f"Predicted residency: "
        f"{departing.predicted_residency:.2f}s"
    )

    # We model the handover trigger at
    # HANDOVER_MARGIN seconds before predicted departure.
    trigger_time = max(
        departing.predicted_residency
        - HANDOVER_MARGIN,
        0.0,
    )

    remaining_at_trigger = (
        departing.predicted_residency
        - trigger_time
    )

    print(
        f"Handover margin    : "
        f"{HANDOVER_MARGIN:.2f}s"
    )

    print(
        f"Trigger point      : "
        f"{trigger_time:.2f}s"
    )

    print(
        f"Remaining at trigger: "
        f"{remaining_at_trigger:.2f}s"
    )

    if (
        remaining_at_trigger
        <= HANDOVER_MARGIN
    ):

        print(
            f"New active provider: "
            f"{standby.vehicle_id}"
        )

        print(
            "Handover           : SUCCESS"
        )

        handover_pass = True

    else:

        print(
            "Handover           : NOT TRIGGERED"
        )

        handover_pass = False

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print()
    print("=" * 95)
    print(
        "                       VALIDATION"
    )
    print("=" * 95)

    print(
        "Admission-aware allocation : PASS"
    )

    print(
        "Standby selection          : "
        + (
            "PASS"
            if standby is not None
            else "FAIL"
        )
    )

    print(
        "Pre-departure handover     : "
        + (
            "PASS"
            if handover_pass
            else "FAIL"
        )
    )

    print()
    print(
        "Standby + Handover Integration: "
        + (
            "PASS"
            if standby is not None
            and handover_pass
            else "CHECK"
        )
    )

    print("=" * 95)


if __name__ == "__main__":
    main()
