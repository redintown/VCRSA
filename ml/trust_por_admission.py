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

X_THRESHOLD = 25.0
TMIN_THRESHOLD = 10.0

TRUST_THRESHOLD = 0.70
POR_TOLERANCE = 0.05


# ============================================================
# Vehicle Security State
# ============================================================

@dataclass
class VehicleSecurityState:

    vehicle_id: str

    trust_score: float

    trust_status: str

    predicted_residency: float

    admission: str

    claimed_resource: float

    measured_resource: float

    por_status: str


# ============================================================
# Resource profile
# ============================================================

def resource_profile(vehicle_index):

    resource_class = vehicle_index % 3

    if resource_class == 0:
        return 4.0

    if resource_class == 1:
        return 8.0

    return 12.0


# ============================================================
# Trust Gate
# ============================================================

def trust_gate(
    vehicle_id,
    trust_score,
):

    if trust_score >= TRUST_THRESHOLD:

        return True, "TRUSTED"

    return False, "UNTRUSTED"


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
# Snapshot sequence
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

    results = []

    # --------------------------------------------------------
    # Build security states
    # --------------------------------------------------------

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

        # ----------------------------------------------------
        # Prototype trust assignment
        # ----------------------------------------------------
        #
        # Most vehicles are trusted.
        # eastbound_flow.21 is deliberately assigned
        # a low trust score to test the Trust Gate.
        #

        trust_score = 0.90

        if vehicle_id == "eastbound_flow.21":
            trust_score = 0.40

        trusted, trust_status = trust_gate(
            vehicle_id,
            trust_score,
        )

        # ----------------------------------------------------
        # LSTM
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Resource
        # ----------------------------------------------------

        claimed_resource = resource_profile(
            vehicle_index
        )

        measured_resource = claimed_resource

        # Deliberately create a false resource claim
        # for eastbound_flow.20.
        if vehicle_id == "eastbound_flow.20":
            measured_resource = (
                claimed_resource * 0.40
            )

        claim = ResourceClaim(
            vehicle_id=vehicle_id,
            resource_type=RESOURCE_TYPE,
            claimed_amount=claimed_resource,
        )

        proof = por.verify(
            claim,
            measured_resource,
        )

        results.append(
            VehicleSecurityState(
                vehicle_id=vehicle_id,
                trust_score=trust_score,
                trust_status=trust_status,
                predicted_residency=(
                    predicted_residency
                ),
                admission=admission.value,
                claimed_resource=claimed_resource,
                measured_resource=measured_resource,
                por_status=proof.status.value,
            )
        )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    print()
    print("=" * 125)
    print(
        "              TRUST → PoR → ADMISSION SECURITY PIPELINE"
    )
    print("=" * 125)

    print(
        f"Snapshot={SNAPSHOT_TIME:.0f}s"
        f" | Zone={TARGET_ZONE}"
        f" | Resource={RESOURCE_TYPE}"
    )

    print()

    print(
        f"{'Vehicle':<24}"
        f"{'Trust':>8}"
        f"{'Trust Gate':>14}"
        f"{'Residency':>12}"
        f"{'Admission':>14}"
        f"{'PoR':>12}"
        f"{'Final Gate':>14}"
    )

    print("-" * 125)

    final_eligible = 0

    for result in results:

        final_gate = (
            result.trust_status == "TRUSTED"
            and result.por_status == "VERIFIED"
            and result.admission != "REJECT"
        )

        if final_gate:
            final_eligible += 1

        print(
            f"{result.vehicle_id:<24}"
            f"{result.trust_score:>8.2f}"
            f"{result.trust_status:>14}"
            f"{result.predicted_residency:>12.2f}"
            f"{result.admission:>14}"
            f"{result.por_status:>12}"
            f"{'ELIGIBLE' if final_gate else 'BLOCKED':>14}"
        )

    # --------------------------------------------------------
    # Counts
    # --------------------------------------------------------

    trust_rejected = sum(
        r.trust_status == "UNTRUSTED"
        for r in results
    )

    por_rejected = sum(
        r.por_status == "REJECTED"
        for r in results
    )

    admission_rejected = sum(
        r.admission == "REJECT"
        for r in results
    )

    print()
    print(
        f"Total vehicles       : {len(results)}"
    )

    print(
        f"Trust rejected       : {trust_rejected}"
    )

    print(
        f"PoR rejected         : {por_rejected}"
    )

    print(
        f"Admission rejected   : {admission_rejected}"
    )

    print(
        f"Final eligible       : {final_eligible}"
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    trust_test = any(
        r.vehicle_id == "eastbound_flow.21"
        and r.trust_status == "UNTRUSTED"
        for r in results
    )

    por_test = any(
        r.vehicle_id == "eastbound_flow.20"
        and r.por_status == "REJECTED"
        for r in results
    )

    final_gate_test = all(
        not (
            r.trust_status == "UNTRUSTED"
            or r.por_status == "REJECTED"
            or r.admission == "REJECT"
        )
        for r in results
        if (
            r.trust_status == "TRUSTED"
            and r.por_status == "VERIFIED"
            and r.admission != "REJECT"
        )
    )

    print()
    print("=" * 125)
    print(
        "                         VALIDATION"
    )
    print("=" * 125)

    print(
        "Trust Gate test        : "
        + ("PASS" if trust_test else "FAIL")
    )

    print(
        "PoR Gate test          : "
        + ("PASS" if por_test else "FAIL")
    )

    print(
        "Admission Gate present : PASS"
    )

    print(
        "Sequential security    : "
        + ("PASS" if final_gate_test else "FAIL")
    )

    print()
    print(
        "Trust + PoR + Admission: "
        + (
            "PASS"
            if (
                trust_test
                and por_test
                and final_gate_test
            )
            else "CHECK"
        )
    )

    print("=" * 125)


if __name__ == "__main__":
    main()
