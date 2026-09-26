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

X_THRESHOLD = 25.0
TMIN_THRESHOLD = 10.0

TRUST_THRESHOLD = 0.70
POR_TOLERANCE = 0.05

FAIRNESS_RESERVE = 0.10

REQUEST_RESOURCE = "CPU"
REQUEST_AMOUNT = 20.0


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
# 10-second sequence
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
# Trust
# ============================================================

def trust_status(vehicle_id):

    # Deliberate security test.
    if vehicle_id == "eastbound_flow.21":
        return 0.40, "UNTRUSTED"

    return 0.90, "TRUSTED"


# ============================================================
# Position / RSU validation
# ============================================================

def position_status(
    vehicle_id,
    actual_zone,
    x,
):

    # Deliberate false-zone test.
    claimed_zone = actual_zone

    if vehicle_id == "eastbound_flow.18":
        claimed_zone = 1

    zone_start = claimed_zone * 1000.0
    zone_end = zone_start + 1000.0

    margin = 50.0

    valid = (
        zone_start - margin
        <= x
        <= zone_end + margin
    )

    return (
        "VALID"
        if valid
        else "INVALID"
    )


# ============================================================
# Security-qualified provider
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

    # ------------------------------
    # S1 Trust
    # ------------------------------

    trust_score, trust = trust_status(
        vehicle_id
    )

    # ------------------------------
    # S4 Position
    # ------------------------------

    position = position_status(
        vehicle_id,
        zone_id,
        float(current["x"]),
    )

    # ------------------------------
    # LSTM
    # ------------------------------

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

    # ------------------------------
    # S2 PoR
    # ------------------------------

    resources = resource_profile(
        vehicle_index
    )

    claimed_cpu = resources["CPU"]

    measured_cpu = claimed_cpu

    # Deliberate false resource test.
    if vehicle_id == "eastbound_flow.20":
        measured_cpu = claimed_cpu * 0.40

    claim = ResourceClaim(
        vehicle_id=vehicle_id,
        resource_type="CPU",
        claimed_amount=claimed_cpu,
    )

    proof = por.verify(
        claim,
        measured_cpu,
    )

    # ------------------------------
    # Final security gate
    # ------------------------------

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
        "trust_score": trust_score,
        "trust": trust,
        "position": position,
        "residency": residency,
        "admission": admission.value,
        "claimed_cpu": claimed_cpu,
        "measured_cpu": measured_cpu,
        "por": proof.status.value,
        "eligible": eligible,
    }


# ============================================================
# Cluster construction
# ============================================================

def build_clusters(providers):

    clusters = defaultdict(list)

    for provider in providers:

        if not provider["eligible"]:
            continue

        key = (
            provider["zone"],
            "CPU",
        )

        clusters[key].append(
            provider
        )

    return clusters


# ============================================================
# Provider scoring
# ============================================================

def provider_score(provider):

    residency_score = min(
        provider["residency"] / 60.0,
        1.0,
    )

    capacity_score = min(
        provider["measured_cpu"] / 16.0,
        1.0,
    )

    trust_score = provider[
        "trust_score"
    ]

    return (
        0.50 * residency_score
        + 0.30 * trust_score
        + 0.20 * capacity_score
    )


# ============================================================
# Intra-resource search + greedy allocation
# ============================================================

def allocate_from_cluster(
    cluster,
    requested_amount,
):

    candidates = []

    for provider in cluster:

        usable_cpu = (
            provider["measured_cpu"]
            * (1.0 - FAIRNESS_RESERVE)
        )

        if usable_cpu <= 0:
            continue

        score = provider_score(
            provider
        )

        candidates.append(
            (
                provider,
                usable_cpu,
                score,
            )
        )

    candidates.sort(
        key=lambda x: x[2],
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
            {
                "vehicle_id":
                    provider["vehicle_id"],
                "amount":
                    amount,
                "residency":
                    provider["residency"],
                "score":
                    score,
            }
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
    # Security evaluation
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

        result = evaluate_provider(
            vehicle_index,
            vehicle_id,
            current,
            sequence,
            model,
            scaler,
            device,
            por,
        )

        providers.append(result)

    # ========================================================
    # Provider security table
    # ========================================================

    print()
    print("=" * 125)
    print(
        "             SECURITY-QUALIFIED RESOURCE PROVIDERS"
    )
    print("=" * 125)

    print(
        f"{'Vehicle':<24}"
        f"{'Trust':>11}"
        f"{'Position':>12}"
        f"{'Residency':>12}"
        f"{'Admission':>14}"
        f"{'PoR':>12}"
        f"{'CPU':>9}"
        f"{'Eligible':>12}"
    )

    print("-" * 125)

    for p in providers:

        print(
            f"{p['vehicle_id']:<24}"
            f"{p['trust']:>11}"
            f"{p['position']:>12}"
            f"{p['residency']:>12.2f}"
            f"{p['admission']:>14}"
            f"{p['por']:>12}"
            f"{p['measured_cpu']:>9.2f}"
            f"{str(p['eligible']):>12}"
        )

    # ========================================================
    # Build resource clusters
    # ========================================================

    clusters = build_clusters(
        providers
    )

    print()
    print("=" * 125)
    print(
        "                    SECURE RESOURCE CLUSTERS"
    )
    print("=" * 125)

    for (zone, resource_type), members in clusters.items():

        total_cpu = sum(
            p["measured_cpu"]
            for p in members
        )

        print()
        print(
            f"Cluster: "
            f"(Zone {zone}, {resource_type})"
        )

        print(
            f"Members       : "
            f"{len(members)}"
        )

        print(
            f"Total CPU     : "
            f"{total_cpu:.2f}"
        )

        print(
            "Vehicles      : "
            + ", ".join(
                p["vehicle_id"]
                for p in members
            )
        )

    # ========================================================
    # Resource search
    # ========================================================

    requested_cluster = (
        TARGET_ZONE,
        REQUEST_RESOURCE,
    )

    cluster = clusters.get(
        requested_cluster,
        [],
    )

    print()
    print("=" * 125)
    print(
        "                     INTRA-RESOURCE SEARCH"
    )
    print("=" * 125)

    print(
        f"Requested resource : "
        f"{REQUEST_RESOURCE}"
    )

    print(
        f"Requested amount   : "
        f"{REQUEST_AMOUNT:.2f}"
    )

    print(
        f"Search cluster     : "
        f"(Zone {TARGET_ZONE}, "
        f"{REQUEST_RESOURCE})"
    )

    print(
        f"Eligible providers : "
        f"{len(cluster)}"
    )

    # ========================================================
    # Allocation
    # ========================================================

    allocations, remaining = (
        allocate_from_cluster(
            cluster,
            REQUEST_AMOUNT,
        )
    )

    print()
    print("=" * 125)
    print(
        "                       GREEDY ALLOCATION"
    )
    print("=" * 125)

    if not allocations:

        print(
            "Allocation failed."
        )

        return

    total_allocated = sum(
        a["amount"]
        for a in allocations
    )

    for allocation in allocations:

        print(
            f"{allocation['vehicle_id']:<24}"
            f"allocated={allocation['amount']:>7.2f}"
            f" | residency="
            f"{allocation['residency']:.2f}s"
            f" | score="
            f"{allocation['score']:.4f}"
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

    # ========================================================
    # PRT
    # ========================================================

    if allocations:

        prt_duration = min(
            a["residency"]
            for a in allocations
        )

        print()
        print(
            f"PRT tenancy duration : "
            f"{prt_duration:.2f}s"
        )

    # ========================================================
    # Validation
    # ========================================================

    eligible_count = sum(
        p["eligible"]
        for p in providers
    )

    blocked_count = (
        len(providers)
        - eligible_count
    )

    allocation_pass = (
        abs(
            total_allocated
            - REQUEST_AMOUNT
        ) < 1e-6
    )

    only_secure_providers = all(
        any(
            p["vehicle_id"]
            == a["vehicle_id"]
            and p["eligible"]
            for p in providers
        )
        for a in allocations
    )

    print()
    print("=" * 125)
    print(
        "                           VALIDATION"
    )
    print("=" * 125)

    print(
        f"Security-qualified providers : "
        f"{eligible_count}"
    )

    print(
        f"Blocked providers            : "
        f"{blocked_count}"
    )

    print(
        "Secure cluster created       : "
        + (
            "PASS"
            if len(cluster) > 0
            else "FAIL"
        )
    )

    print(
        "Intra-resource search        : "
        + (
            "PASS"
            if len(cluster) > 0
            else "FAIL"
        )
    )

    print(
        "Only secure providers used  : "
        + (
            "PASS"
            if only_secure_providers
            else "FAIL"
        )
    )

    print(
        "20 CPU allocation            : "
        + (
            "PASS"
            if allocation_pass
            else "FAIL"
        )
    )

    print()
    print(
        "Secure RCSA Pipeline         : "
        + (
            "PASS"
            if (
                len(cluster) > 0
                and only_secure_providers
                and allocation_pass
            )
            else "CHECK"
        )
    )

    print("=" * 125)


if __name__ == "__main__":
    main()
