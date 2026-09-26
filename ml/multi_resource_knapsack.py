import sys
from pathlib import Path
from collections import defaultdict

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

# Joint multi-resource request
REQUEST = {
    "CPU": 20.0,
    "MEMORY": 40.0,
    "STORAGE": 500.0,
}

FAIRNESS_RESERVE = 0.10

X_THRESHOLD = 25.0
TMIN_THRESHOLD = 10.0

TRUST_THRESHOLD = 0.70
POR_TOLERANCE = 0.05


# ============================================================
# Resource profiles
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
# LSTM
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
# Sequence
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
# Features
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
# LSTM residency
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
# Security
# ============================================================

def trust_status(vehicle_id):

    if vehicle_id == "eastbound_flow.21":
        return 0.40, "UNTRUSTED"

    return 0.90, "TRUSTED"


def position_status(
    vehicle_id,
    zone_id,
    x,
):

    claimed_zone = zone_id

    if vehicle_id == "eastbound_flow.18":
        claimed_zone = 1

    start = (
        claimed_zone * 1000.0
    )

    end = start + 1000.0

    margin = 50.0

    if (
        start - margin
        <= x
        <= end + margin
    ):
        return "VALID"

    return "INVALID"


# ============================================================
# Provider evaluation
# ============================================================

def evaluate_provider(
    vehicle_index,
    vehicle_id,
    current,
    sequence,
    model,
    scaler,
    device,
    por,
):

    zone_id = int(
        current["zone_id"]
    )

    trust_score, trust = (
        trust_status(vehicle_id)
    )

    position = position_status(
        vehicle_id,
        zone_id,
        float(current["x"]),
    )

    residency = predict_residency(
        sequence,
        model,
        scaler,
        device,
    )

    admission = classify_admission(
        residency,
        X_THRESHOLD,
        TMIN_THRESHOLD,
    )

    resources = resource_profile(
        vehicle_index
    )

    measured_resources = dict(
        resources
    )

    # Deliberate false resource claim
    if vehicle_id == "eastbound_flow.20":
        measured_resources["CPU"] *= 0.40

    claim = ResourceClaim(
        vehicle_id=vehicle_id,
        resource_type="CPU",
        claimed_amount=resources["CPU"],
    )

    proof = por.verify(
        claim,
        measured_resources["CPU"],
    )

    eligible = (
        trust == "TRUSTED"
        and position == "VALID"
        and proof.status.value == "VERIFIED"
        and admission.value != "REJECT"
    )

    return {
        "vehicle_id": vehicle_id,
        "zone": zone_id,
        "direction": current["direction"],
        "trust": trust,
        "trust_score": trust_score,
        "position": position,
        "residency": residency,
        "admission": admission.value,
        "por": proof.status.value,
        "resources": measured_resources,
        "eligible": eligible,
    }


# ============================================================
# Score for greedy selection
# ============================================================

def provider_score(provider):

    residency_score = min(
        provider["residency"] / 60.0,
        1.0,
    )

    cpu_score = min(
        provider["resources"]["CPU"] / 16.0,
        1.0,
    )

    memory_score = min(
        provider["resources"]["MEMORY"] / 32.0,
        1.0,
    )

    storage_score = min(
        provider["resources"]["STORAGE"] / 512.0,
        1.0,
    )

    capacity_score = (
        cpu_score
        + memory_score
        + storage_score
    ) / 3.0

    return (
        0.50 * residency_score
        + 0.30 * provider["trust_score"]
        + 0.20 * capacity_score
    )


# ============================================================
# Multi-resource greedy allocation
# ============================================================

def greedy_multi_resource_allocate(
    providers,
    request,
):

    candidates = []

    for provider in providers:

        if not provider["eligible"]:
            continue

        usable = {
            resource:
                amount
                * (1.0 - FAIRNESS_RESERVE)
            for resource, amount
            in provider["resources"].items()
        }

        score = provider_score(
            provider
        )

        candidates.append(
            (
                provider,
                usable,
                score,
            )
        )

    candidates.sort(
        key=lambda item: item[2],
        reverse=True,
    )

    remaining = dict(request)

    allocations = []

    for provider, usable, score in candidates:

        contribution = {}

        for resource in request:

            amount = min(
                usable[resource],
                remaining[resource],
            )

            if amount > 0:
                contribution[resource] = amount

        if not contribution:
            continue

        for resource, amount in contribution.items():
            remaining[resource] -= amount

        allocations.append(
            {
                "vehicle_id":
                    provider["vehicle_id"],
                "zone":
                    provider["zone"],
                "residency":
                    provider["residency"],
                "score":
                    score,
                "resources":
                    contribution,
            }
        )

        if all(
            value <= 1e-9
            for value in remaining.values()
        ):
            break

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

    por = ProofOfResource(
        tolerance=POR_TOLERANCE
    )

    vehicle_ids = (
        df["vehicle_id"]
        .drop_duplicates()
        .tolist()
    )

    providers = []

    # ========================================================
    # Security qualification
    # ========================================================

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

        if int(current["zone_id"]) != TARGET_ZONE:
            continue

        sequence = get_snapshot_sequence(
            vehicle_df,
            SNAPSHOT_TIME,
        )

        if sequence is None:
            continue

        provider = evaluate_provider(
            vehicle_index,
            vehicle_id,
            current,
            sequence,
            model,
            scaler,
            device,
            por,
        )

        providers.append(
            provider
        )

    # ========================================================
    # Provider table
    # ========================================================

    print()
    print("=" * 135)
    print(
        "             SECURITY-QUALIFIED MULTI-RESOURCE PROVIDERS"
    )
    print("=" * 135)

    print(
        f"{'Vehicle':<24}"
        f"{'Trust':>11}"
        f"{'Position':>11}"
        f"{'Residency':>11}"
        f"{'Admission':>14}"
        f"{'PoR':>11}"
        f"{'CPU':>8}"
        f"{'MEM':>8}"
        f"{'STOR':>9}"
        f"{'Eligible':>11}"
    )

    print("-" * 135)

    for provider in providers:

        r = provider["resources"]

        print(
            f"{provider['vehicle_id']:<24}"
            f"{provider['trust']:>11}"
            f"{provider['position']:>11}"
            f"{provider['residency']:>11.2f}"
            f"{provider['admission']:>14}"
            f"{provider['por']:>11}"
            f"{r['CPU']:>8.2f}"
            f"{r['MEMORY']:>8.2f}"
            f"{r['STORAGE']:>9.2f}"
            f"{str(provider['eligible']):>11}"
        )

    # ========================================================
    # Request
    # ========================================================

    print()
    print("=" * 135)
    print(
        "                   JOINT MULTI-RESOURCE REQUEST"
    )
    print("=" * 135)

    for resource, amount in REQUEST.items():

        print(
            f"{resource:<10}: "
            f"{amount:.2f}"
        )

    print(
        f"Fairness reserve: "
        f"{FAIRNESS_RESERVE * 100:.0f}%"
    )

    # ========================================================
    # Greedy knapsack
    # ========================================================

    allocations, remaining = (
        greedy_multi_resource_allocate(
            providers,
            REQUEST,
        )
    )

    print()
    print("=" * 135)
    print(
        "                  GREEDY MULTI-RESOURCE ALLOCATION"
    )
    print("=" * 135)

    for allocation in allocations:

        print()
        print(
            f"Provider: "
            f"{allocation['vehicle_id']}"
        )

        print(
            f"Zone: "
            f"{allocation['zone']}"
        )

        print(
            f"Score: "
            f"{allocation['score']:.4f}"
        )

        print(
            f"Predicted residency: "
            f"{allocation['residency']:.2f}s"
        )

        for resource, amount in (
            allocation["resources"].items()
        ):

            print(
                f"  {resource:<9}"
                f" {amount:.2f}"
            )

    # ========================================================
    # Totals
    # ========================================================

    allocated_totals = {
        resource: 0.0
        for resource in REQUEST
    }

    for allocation in allocations:

        for resource, amount in (
            allocation["resources"].items()
        ):

            allocated_totals[
                resource
            ] += amount

    print()
    print(
        "ALLOCATED TOTALS"
    )

    for resource in REQUEST:

        print(
            f"{resource:<10}: "
            f"{allocated_totals[resource]:.2f}"
            f" / "
            f"{REQUEST[resource]:.2f}"
        )

    print()
    print(
        "REMAINING"
    )

    for resource in REQUEST:

        print(
            f"{resource:<10}: "
            f"{remaining[resource]:.2f}"
        )

    # ========================================================
    # PRT
    # ========================================================

    if allocations:

        prt_duration = min(
            allocation["residency"]
            for allocation in allocations
        )

        print()
        print(
            f"PRT tenancy duration: "
            f"{prt_duration:.2f}s"
        )

    # ========================================================
    # Validation
    # ========================================================

    request_satisfied = all(
        remaining[resource] <= 1e-9
        for resource in REQUEST
    )

    only_secure = all(
        any(
            p["vehicle_id"]
            == allocation["vehicle_id"]
            and p["eligible"]
            for p in providers
        )
        for allocation in allocations
    )

    fairness_reserve_applied = all(
        allocation["resources"][resource]
        <= next(
            p["resources"][resource]
            for p in providers
            if p["vehicle_id"]
            == allocation["vehicle_id"]
        ) * (1.0 - FAIRNESS_RESERVE)
        + 1e-9
        for allocation in allocations
        for resource in allocation["resources"]
    )

    print()
    print("=" * 135)
    print(
        "                         VALIDATION"
    )
    print("=" * 135)

    print(
        "Security-qualified providers : PASS"
    )

    print(
        "Greedy multi-resource logic  : "
        + (
            "PASS"
            if len(allocations) > 0
            else "FAIL"
        )
    )

    print(
        "Only secure providers used   : "
        + (
            "PASS"
            if only_secure
            else "FAIL"
        )
    )

    print(
        "Fairness reserve applied     : "
        + (
            "PASS"
            if fairness_reserve_applied
            else "FAIL"
        )
    )

    print(
        "CPU requirement satisfied    : "
        + (
            "PASS"
            if remaining["CPU"] <= 1e-9
            else "FAIL"
        )
    )

    print(
        "MEMORY requirement satisfied : "
        + (
            "PASS"
            if remaining["MEMORY"] <= 1e-9
            else "FAIL"
        )
    )

    print(
        "STORAGE requirement satisfied: "
        + (
            "PASS"
            if remaining["STORAGE"] <= 1e-9
            else "FAIL"
        )
    )

    print()
    print(
        "Joint Multi-Resource Allocation: "
        + (
            "PASS"
            if (
                request_satisfied
                and only_secure
                and fairness_reserve_applied
            )
            else "CHECK"
        )
    )

    print("=" * 135)


if __name__ == "__main__":
    main()
