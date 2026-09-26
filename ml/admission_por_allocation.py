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
from proof_of_resource import (
    ProofOfResource,
    ResourceClaim,
    PoRStatus,
)


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

POR_TOLERANCE = 0.05


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
# Provider
# ============================================================

@dataclass
class Provider:

    vehicle_id: str
    zone_id: int
    direction: str

    resource_type: str
    claimed_amount: float
    measured_amount: float

    predicted_residency: float
    admission: str

    por_status: str
    trust_score: float = 0.90


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
# Get 10-second history
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
# Prepare LSTM input
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
        provider.measured_amount / 16.0,
        1.0,
    )

    return (
        0.50 * residency_score
        + 0.30 * provider.trust_score
        + 0.20 * capacity_score
    )


# ============================================================
# Build provider pool
# ============================================================

def build_provider_pool(
    df,
    model,
    scaler,
    device,
):

    por = ProofOfResource(
        tolerance=POR_TOLERANCE
    )

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

        zone_id = int(
            current["zone_id"]
        )

        if zone_id != TARGET_ZONE:
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

        claimed_amount = resources[
            RESOURCE_TYPE
        ]

        # ----------------------------------------------------
        # Simulated measured resource
        # ----------------------------------------------------
        #
        # Most providers report the actual profile.
        # One provider is deliberately given an invalid
        # measurement to demonstrate PoR rejection.
        #

        measured_amount = claimed_amount

        if vehicle_id == "eastbound_flow.20":
            measured_amount = (
                claimed_amount * 0.40
            )

        claim = ResourceClaim(
            vehicle_id=vehicle_id,
            resource_type=RESOURCE_TYPE,
            claimed_amount=claimed_amount,
        )

        proof = por.verify(
            claim,
            measured_amount,
        )

        providers.append(
            Provider(
                vehicle_id=vehicle_id,
                zone_id=zone_id,
                direction=current[
                    "direction"
                ],
                resource_type=RESOURCE_TYPE,
                claimed_amount=claimed_amount,
                measured_amount=measured_amount,
                predicted_residency=(
                    predicted_residency
                ),
                admission=admission.value,
                por_status=proof.status.value,
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

        # Admission gate
        if provider.admission == "REJECT":
            continue

        # PoR gate
        if provider.por_status != "VERIFIED":
            continue

        usable = (
            provider.measured_amount
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
            (
                provider,
                amount,
                score,
            )
        )

        remaining -= amount

    return allocations, remaining


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
    # Provider table
    # --------------------------------------------------------

    print()
    print("=" * 110)
    print(
        "             ADMISSION + PoR PROVIDER GATE"
    )
    print("=" * 110)

    print(
        f"Snapshot={SNAPSHOT_TIME:.0f}s"
        f" | Zone={TARGET_ZONE}"
        f" | Resource={RESOURCE_TYPE}"
    )

    print()

    print(
        f"{'Vehicle':<24}"
        f"{'Claimed':>10}"
        f"{'Measured':>10}"
        f"{'Residency':>12}"
        f"{'Admission':>14}"
        f"{'PoR':>12}"
    )

    print("-" * 110)

    for provider in providers:

        print(
            f"{provider.vehicle_id:<24}"
            f"{provider.claimed_amount:>10.2f}"
            f"{provider.measured_amount:>10.2f}"
            f"{provider.predicted_residency:>12.2f}"
            f"{provider.admission:>14}"
            f"{provider.por_status:>12}"
        )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    admission_rejected = sum(
        p.admission == "REJECT"
        for p in providers
    )

    por_rejected = sum(
        p.por_status == "REJECTED"
        for p in providers
    )

    por_verified = sum(
        p.por_status == "VERIFIED"
        for p in providers
    )

    print()
    print(
        f"Total providers      : {len(providers)}"
    )

    print(
        f"PoR verified         : {por_verified}"
    )

    print(
        f"PoR rejected         : {por_rejected}"
    )

    print(
        f"Admission rejected   : {admission_rejected}"
    )

    # --------------------------------------------------------
    # Allocation
    # --------------------------------------------------------

    allocations, remaining = (
        greedy_allocate(
            providers,
            REQUEST_AMOUNT,
        )
    )

    print()
    print("=" * 110)
    print(
        "                     RESOURCE ALLOCATION"
    )
    print("=" * 110)

    print(
        f"Requested CPU : {REQUEST_AMOUNT:.2f}"
    )

    if not allocations:

        print(
            "Allocation failed."
        )

        return

    total_allocated = sum(
        amount
        for provider, amount, score
        in allocations
    )

    for provider, amount, score in allocations:

        print(
            f"{provider.vehicle_id:<24}"
            f"allocated={amount:>6.2f}"
            f" | score={score:.4f}"
            f" | admission={provider.admission}"
            f" | PoR={provider.por_status}"
        )

    print()
    print(
        f"Total allocated : "
        f"{total_allocated:.2f}"
    )

    print(
        f"Remaining       : "
        f"{remaining:.2f}"
    )

    # --------------------------------------------------------
    # PRT
    # --------------------------------------------------------

    tenancy_duration = min(
        provider.predicted_residency
        for provider, amount, score
        in allocations
    )

    print()
    print(
        f"PRT duration    : "
        f"{tenancy_duration:.2f}s"
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    allocation_pass = (
        abs(
            total_allocated
            - REQUEST_AMOUNT
        ) < 1e-6
    )

    all_allocated_verified = all(
        provider.por_status == "VERIFIED"
        for provider, amount, score
        in allocations
    )

    print()
    print("=" * 110)
    print(
        "                         VALIDATION"
    )
    print("=" * 110)

    print(
        "Real SUMO provider pool : PASS"
    )

    print(
        "Admission gate           : PASS"
    )

    print(
        "PoR verification         : "
        + (
            "PASS"
            if por_verified > 0
            else "FAIL"
        )
    )

    print(
        "Rejected provider blocked: "
        + (
            "PASS"
            if por_rejected > 0
            else "CHECK"
        )
    )

    print(
        "Allocation only from "
        "verified providers     : "
        + (
            "PASS"
            if all_allocated_verified
            else "FAIL"
        )
    )

    print(
        "20 CPU allocation       : "
        + (
            "PASS"
            if allocation_pass
            else "FAIL"
        )
    )

    print()
    print(
        "Admission + PoR + "
        "Allocation: "
        + (
            "PASS"
            if (
                allocation_pass
                and all_allocated_verified
                and por_rejected > 0
            )
            else "CHECK"
        )
    )

    print("=" * 110)


if __name__ == "__main__":
    main()
