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

REQUEST_RESOURCE = "CPU"

# Deliberately request more than the local secure
# Zone-0 CPU cluster can provide.
REQUEST_AMOUNT = 30.0

FAIRNESS_RESERVE = 0.10

X_THRESHOLD = 25.0
TMIN_THRESHOLD = 10.0

TRUST_THRESHOLD = 0.70
POR_TOLERANCE = 0.05

# Maximum V2V/V2V-resource-cluster search hops.
MAX_HOPS = 3


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
# Residency prediction
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

    zone_start = (
        claimed_zone * 1000.0
    )

    zone_end = (
        zone_start + 1000.0
    )

    margin = 50.0

    if (
        zone_start - margin
        <= x
        <= zone_end + margin
    ):
        return "VALID"

    return "INVALID"


# ============================================================
# Evaluate provider
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

    claimed_cpu = resources["CPU"]

    measured_cpu = claimed_cpu

    if vehicle_id == "eastbound_flow.20":
        measured_cpu = (
            claimed_cpu * 0.40
        )

    claim = ResourceClaim(
        vehicle_id=vehicle_id,
        resource_type="CPU",
        claimed_amount=claimed_cpu,
    )

    proof = por.verify(
        claim,
        measured_cpu,
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
        "cpu": measured_cpu,
        "eligible": eligible,
    }


# ============================================================
# Build secure clusters
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
# Cluster capacity
# ============================================================

def usable_capacity(cluster):

    return sum(
        p["cpu"] * (1.0 - FAIRNESS_RESERVE)
        for p in cluster
    )


# ============================================================
# Greedy allocation from a cluster
# ============================================================

def allocate_from_cluster(
    cluster,
    requested_amount,
):

    candidates = []

    for provider in cluster:

        usable = (
            provider["cpu"]
            * (1.0 - FAIRNESS_RESERVE)
        )

        if usable <= 0:
            continue

        residency_score = min(
            provider["residency"] / 60.0,
            1.0,
        )

        capacity_score = min(
            provider["cpu"] / 16.0,
            1.0,
        )

        score = (
            0.50 * residency_score
            + 0.30 * provider["trust_score"]
            + 0.20 * capacity_score
        )

        candidates.append(
            (
                provider,
                usable,
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
                "zone":
                    provider["zone"],
                "amount":
                    amount,
                "residency":
                    provider["residency"],
                "score":
                    score,
                "hop":
                    None,
            }
        )

        remaining -= amount

    return allocations, remaining


# ============================================================
# Inter-resource search
# ============================================================

def inter_resource_search(
    clusters,
    source_zone,
    resource_type,
    requested_amount,
):

    allocations = []
    remaining = requested_amount

    # --------------------------------------------------------
    # Hop 0 = local cluster
    # --------------------------------------------------------

    local_key = (
        source_zone,
        resource_type,
    )

    local_cluster = clusters.get(
        local_key,
        [],
    )

    local_allocations, remaining = (
        allocate_from_cluster(
            local_cluster,
            remaining,
        )
    )

    for allocation in local_allocations:
        allocation["hop"] = 0
        allocations.append(
            allocation
        )

    if remaining <= 0:
        return allocations, remaining, "LOCAL"

    # --------------------------------------------------------
    # Remote trusted clusters
    # --------------------------------------------------------

    remote_zones = sorted(
        {
            zone
            for zone, rtype
            in clusters.keys()
            if rtype == resource_type
            and zone != source_zone
        },
        key=lambda z: abs(
            z - source_zone
        ),
    )

    for zone in remote_zones:

        # Each remote cluster is treated as one
        # resource-search hop.
        hop = abs(
            zone - source_zone
        )

        if hop > MAX_HOPS:
            continue

        key = (
            zone,
            resource_type,
        )

        cluster = clusters.get(
            key,
            [],
        )

        if not cluster:
            continue

        remote_allocations, remaining = (
            allocate_from_cluster(
                cluster,
                remaining,
            )
        )

        for allocation in remote_allocations:
            allocation["hop"] = hop
            allocations.append(
                allocation
            )

        if remaining <= 0:
            return (
                allocations,
                remaining,
                "INTER_RESOURCE",
            )

    # --------------------------------------------------------
    # Hop budget exhausted
    # --------------------------------------------------------

    if remaining > 0:
        return (
            allocations,
            remaining,
            "RSU_ASSISTED_REQUIRED",
        )

    return (
        allocations,
        remaining,
        "INTER_RESOURCE",
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
    # Secure clusters
    # ========================================================

    clusters = build_clusters(
        providers
    )

    print()
    print("=" * 115)
    print(
        "                  SECURE RESOURCE CLUSTER MAP"
    )
    print("=" * 115)

    for (zone, resource_type), cluster in sorted(
        clusters.items()
    ):

        print(
            f"Zone {zone} | {resource_type}"
            f" | members={len(cluster)}"
            f" | usable="
            f"{usable_capacity(cluster):.2f}"
        )

    # ========================================================
    # Search
    # ========================================================

    print()
    print("=" * 115)
    print(
        "                    RESOURCE SEARCH"
    )
    print("=" * 115)

    print(
        f"Request resource : {REQUEST_RESOURCE}"
    )

    print(
        f"Request amount   : {REQUEST_AMOUNT:.2f}"
    )

    print(
        f"Source zone      : {TARGET_ZONE}"
    )

    print(
        f"Maximum hops     : {MAX_HOPS}"
    )

    allocations, remaining, search_mode = (
        inter_resource_search(
            clusters,
            TARGET_ZONE,
            REQUEST_RESOURCE,
            REQUEST_AMOUNT,
        )
    )

    print()
    print(
        f"Search mode      : {search_mode}"
    )

    # ========================================================
    # Allocation
    # ========================================================

    print()
    print("=" * 115)
    print(
        "                   SEARCH + ALLOCATION RESULT"
    )
    print("=" * 115)

    if allocations:

        for allocation in allocations:

            print(
                f"{allocation['vehicle_id']:<24}"
                f"zone={allocation['zone']}"
                f" | amount="
                f"{allocation['amount']:.2f}"
                f" | hop="
                f"{allocation['hop']}"
                f" | residency="
                f"{allocation['residency']:.2f}s"
            )

    total_allocated = sum(
        a["amount"]
        for a in allocations
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

        print(
            f"PRT duration    : "
            f"{prt_duration:.2f}s"
        )

    # ========================================================
    # Validation
    # ========================================================

    all_secure = all(
        any(
            p["vehicle_id"]
            == a["vehicle_id"]
            and p["eligible"]
            for p in providers
        )
        for a in allocations
    )

    if REQUEST_AMOUNT > 24.0:

        local_capacity_insufficient = (
            usable_capacity(
                clusters.get(
                    (
                        TARGET_ZONE,
                        REQUEST_RESOURCE,
                    ),
                    [],
                )
            )
            < REQUEST_AMOUNT
        )

    else:

        local_capacity_insufficient = False

    inter_resource_test = any(
        a["hop"] > 0
        for a in allocations
    )

    rsu_required = (
        remaining > 0
        and search_mode
        == "RSU_ASSISTED_REQUIRED"
    )

    print()
    print("=" * 115)
    print(
        "                         VALIDATION"
    )
    print("=" * 115)

    print(
        "Local capacity check      : "
        + (
            "PASS"
            if local_capacity_insufficient
            else "CHECK"
        )
    )

    print(
        "Inter-resource search     : "
        + (
            "PASS"
            if inter_resource_test
            else "NOT NEEDED"
        )
    )

    print(
        "Hop budget enforced       : "
        + (
            "PASS"
            if all(
                a["hop"] is not None
                and a["hop"] <= MAX_HOPS
                for a in allocations
            )
            else "FAIL"
        )
    )

    print(
        "Only secure providers     : "
        + (
            "PASS"
            if all_secure
            else "FAIL"
        )
    )

    print(
        "RSU fallback detection    : "
        + (
            "PASS"
            if rsu_required
            else "NOT NEEDED"
        )
    )

    print("=" * 115)


if __name__ == "__main__":
    main()
